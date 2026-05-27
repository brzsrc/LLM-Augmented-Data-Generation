import argparse
import json
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler


# ============================================================================
# CLI
# ============================================================================
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="./outputs/kfold")

    # K-fold + parallelism
    p.add_argument("--n_folds", type=int, default=3)
    p.add_argument("--devices", default="cuda:0,cuda:1",
                   help="Comma-separated torch devices. Folds are round-robined "
                        "across them, one process per device. Use a single value "
                        "(e.g. 'cuda:0' or 'cpu') to run in-process.")

    # DDQN (per fold)
    p.add_argument("--ddqn_iters", type=int, default=80000)
    p.add_argument("--ddqn_eval_every", type=int, default=6000)
    p.add_argument("--ddqn_batch", type=int, default=512)

    # FQE (per fold, repeated over seeds)
    p.add_argument("--fqe_iters", type=int, default=20000)
    p.add_argument("--fqe_batch", type=int, default=512)
    p.add_argument("--fqe_seeds", type=int, default=5,
                   help="FQE fits per fold; per-patient value = median over these.")

    p.add_argument("--bootstrap_B", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true",
                   help="Disable bf16 autocast (default: enabled on cuda).")
    return p.parse_args()


# ============================================================================
# Constants
# ============================================================================
STATE_COLS = ['study_day', 'weekday', 'slot', 'weather', 'temp', 'loc', 'resp', 'steps30pre']

NUM_ACTIONS = 3
GAMMA = 0.9
DDQN_LR = 5.5e-5
DDQN_HIDDEN = 128

POLICY_NAMES = ["Original (cross-fit)", "No message", "Send a=1", "Send a=2"]
K_POLICIES = len(POLICY_NAMES)


def build_transitions(df):
    df = df.sort_values(['uid', 'study_day', 'slot']).reset_index(drop=True)
    transitions = []
    for uid, g in df.groupby('uid', sort=False):
        g = g.reset_index(drop=True)
        S = g[STATE_COLS].values
        A = g["send"].values
        R = g["reward"].values
        n = len(g)
        for t in range(n):
            ns = S[t + 1] if t < n - 1 else S[t]
            done = 0 if t < n - 1 else 1
            transitions.append({
                "patient_id": uid,
                "s": S[t], "a": int(A[t]), "r": float(R[t]),
                "ns": ns, "done": done,
            })
    return transitions


def to_arrays(trans):
    s = np.array([t["s"] for t in trans], dtype=np.float32)
    a = np.array([t["a"] for t in trans], dtype=np.int64)
    r = np.array([t["r"] for t in trans], dtype=np.float32)
    ns = np.array([t["ns"] for t in trans], dtype=np.float32)
    d = np.array([t["done"] for t in trans], dtype=np.float32)
    pid = np.array([t["patient_id"] for t in trans])
    return s, a, r, ns, d, pid


def filter_trans(trans, pats):
    return [t for t in trans if t["patient_id"] in pats]


def get_initial_states_with_ids(trans):
    """One initial state per patient, with aligned patient-id list.
    `trans` is assumed time-sorted within patient (build_transitions guarantees
    this and filter_trans preserves order)."""
    by_pid = defaultdict(list)
    for t in trans:
        by_pid[t["patient_id"]].append(t)
    pids = list(by_pid.keys())
    init = np.array([by_pid[p][0]["s"] for p in pids], dtype=np.float32)
    return pids, init


def make_folds(patients, n_folds, seed):
    """Partition patients into n_folds disjoint test sets."""
    rng = np.random.default_rng(seed)
    pats = list(patients)
    rng.shuffle(pats)
    return [set(pats[i::n_folds]) for i in range(n_folds)]


# ============================================================================
# Networks  (identical to baseline)
# ============================================================================
class QNet(nn.Module):
    def __init__(self, state_dim, num_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class MultiQNet(nn.Module):
    def __init__(self, n_policies, state_dim, num_actions, hidden=64):
        super().__init__()
        K = n_policies
        self.W1 = nn.Parameter(torch.empty(K, state_dim, hidden))
        self.b1 = nn.Parameter(torch.zeros(K, 1, hidden))
        self.W2 = nn.Parameter(torch.empty(K, hidden, hidden))
        self.b2 = nn.Parameter(torch.zeros(K, 1, hidden))
        self.W3 = nn.Parameter(torch.empty(K, hidden, num_actions))
        self.b3 = nn.Parameter(torch.zeros(K, 1, num_actions))
        for W in [self.W1, self.W2, self.W3]:
            nn.init.kaiming_uniform_(W, a=np.sqrt(5))

    def forward(self, x):
        h = torch.einsum("bd,kdh->kbh", x, self.W1) + self.b1
        h = F.relu(h)
        h = torch.einsum("kbh,khj->kbj", h, self.W2) + self.b2
        h = F.relu(h)
        out = torch.einsum("kbh,kha->kba", h, self.W3) + self.b3
        return out  # (K, B, num_actions)


# ============================================================================
# DDQN training  (identical to baseline, verbose off by default)
# ============================================================================
def train_ddqn(train_trans, val_trans, scaler, device,
               n_iters=80000, eval_every=6000,
               batch=512, lr=DDQN_LR, hidden=DDQN_HIDDEN,
               gamma=GAMMA, tau=1e-4, use_amp=True, verbose=False):
    s, a, r, ns, d, _ = to_arrays(train_trans)
    s = scaler.transform(s); ns = scaler.transform(ns)
    s = torch.tensor(s, dtype=torch.float32, device=device)
    a = torch.tensor(a, dtype=torch.long, device=device)
    r = torch.tensor(r, dtype=torch.float32, device=device)
    ns = torch.tensor(ns, dtype=torch.float32, device=device)
    d = torch.tensor(d, dtype=torch.float32, device=device)

    vs, _, _, _, _, _ = to_arrays(val_trans)
    vs = torch.tensor(scaler.transform(vs), dtype=torch.float32, device=device)

    state_dim = s.shape[1]
    online = QNet(state_dim, NUM_ACTIONS, hidden).to(device)
    target = QNet(state_dim, NUM_ACTIONS, hidden).to(device)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=lr)

    N = len(s)
    best_val_q = -np.inf
    best_state = {k: v.detach().clone() for k, v in online.state_dict().items()}
    autocast = (use_amp and device.type == "cuda")

    for it in range(n_iters):
        idx = torch.randint(0, N, (batch,), device=device)
        sb, ab, rb, nsb, db = s[idx], a[idx], r[idx], ns[idx], d[idx]
        with torch.amp.autocast(device_type="cuda", enabled=autocast, dtype=torch.bfloat16):
            with torch.no_grad():
                next_a = online(nsb).argmax(dim=1, keepdim=True)
                next_q = target(nsb).gather(1, next_a).squeeze(1)
                y = rb + gamma * (1.0 - db) * next_q
            q_sa = online(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q_sa, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
        opt.step()
        with torch.no_grad():
            for tp, op in zip(target.parameters(), online.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * op.data)

        if (it + 1) % eval_every == 0:
            with torch.no_grad():
                v = online(vs).max(dim=1).values.mean().item()
            if v > best_val_q:
                best_val_q = v
                best_state = {k: vv.detach().clone() for k, vv in online.state_dict().items()}
            if verbose:
                print(f"      [ddqn] iter {it+1:6d} loss={loss.item():.4f} val_max_q={v:.4f}")

    online.load_state_dict(best_state)
    return online


# ============================================================================
# FQE — train K policies simultaneously, with VALUE CLIPPING
# ============================================================================
def train_fqe_multi(policy_actions_at_ns, eval_trans, scaler, state_dim,
                    n_policies, device,
                    n_iters=20000, batch=512, lr=4e-3, gamma=GAMMA, tau=0.009,
                    hidden=64, use_amp=True, vmin=None, vmax=None):
    s, a, r, ns, d, _ = to_arrays(eval_trans)
    s = scaler.transform(s); ns = scaler.transform(ns)
    s = torch.tensor(s, dtype=torch.float32, device=device)
    a = torch.tensor(a, dtype=torch.long, device=device)
    r = torch.tensor(r, dtype=torch.float32, device=device)
    ns = torch.tensor(ns, dtype=torch.float32, device=device)
    d = torch.tensor(d, dtype=torch.float32, device=device)

    online = MultiQNet(n_policies, state_dim, NUM_ACTIONS, hidden).to(device)
    target = MultiQNet(n_policies, state_dim, NUM_ACTIONS, hidden).to(device)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=lr)

    N = len(s)
    autocast = (use_amp and device.type == "cuda")
    K = n_policies

    for it in range(n_iters):
        idx = torch.randint(0, N, (batch,), device=device)
        sb = s[idx]; ab = a[idx]; rb = r[idx]; nsb = ns[idx]; db = d[idx]
        ns_a_all = policy_actions_at_ns[:, idx]   # (K, B)

        with torch.amp.autocast(device_type="cuda", enabled=autocast, dtype=torch.bfloat16):
            with torch.no_grad():
                tgt_q_all = target(nsb)
                next_q = tgt_q_all.gather(2, ns_a_all.unsqueeze(-1)).squeeze(-1)
                y = rb.unsqueeze(0) + gamma * (1.0 - db).unsqueeze(0) * next_q
                if vmin is not None:
                    y = y.clamp(vmin, vmax)             # <-- divergence guard
            q_all = online(sb)
            ab_exp = ab.unsqueeze(0).expand(K, -1).unsqueeze(-1)
            q_sa = q_all.gather(2, ab_exp).squeeze(-1)
            loss = F.smooth_l1_loss(q_sa, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
        opt.step()
        with torch.no_grad():
            for tp, op in zip(target.parameters(), online.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * op.data)

    return online


def per_patient_values_multi(qnet_eval, policy_actions_at_init, init_states_scaled,
                             device, vmin=None, vmax=None):
    """Return (K, n_patients) — Q^pi(s0, pi(s0)) for each policy at each test
    patient's initial state."""
    s = torch.tensor(init_states_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        q_all = qnet_eval(s)
        gathered = q_all.gather(2, policy_actions_at_init.unsqueeze(-1)).squeeze(-1)
        if vmin is not None:
            gathered = gathered.clamp(vmin, vmax)
    return gathered.cpu().numpy()  # (K, n)


def policy_actions_for_states(qnet, states_tensor, K, device):
    """(K, N) action table: row 0 = learned argmax, rows 1-3 = fixed 0/1/2."""
    with torch.no_grad():
        actions = torch.zeros(K, states_tensor.shape[0], dtype=torch.long, device=device)
        actions[0] = qnet(states_tensor).argmax(dim=1)
        actions[1] = 0
        actions[2] = 1
        actions[3] = 2
    return actions


# ============================================================================
# One fold: train DDQN on train pats, FQE-evaluate held-out test pats
# ============================================================================
def one_fold(fold_idx, train_pats, test_pats, orig_trans, cfg, device,
             save_dir=None):
    torch.manual_seed(cfg["seed"] + fold_idx)
    np.random.seed(cfg["seed"] + fold_idx)
    use_amp = (not cfg["no_amp"]) and device.type == "cuda"

    train_trans = filter_trans(orig_trans, train_pats)
    test_trans = filter_trans(orig_trans, test_pats)

    # internal val split for DDQN early-stop
    rng = np.random.default_rng(cfg["seed"] + fold_idx)
    tp = sorted(train_pats)
    rng.shuffle(tp)
    n_val = max(2, len(tp) // 4)
    val_pats = set(tp[:n_val])
    tr_pats = set(tp[n_val:])
    tr = filter_trans(train_trans, tr_pats)
    vl = filter_trans(train_trans, val_pats)

    s_tr, _, _, _, _, _ = to_arrays(tr)
    scaler = MinMaxScaler().fit(s_tr)

    qnet = train_ddqn(tr, vl, scaler, device,
                      n_iters=cfg["ddqn_iters"], eval_every=cfg["ddqn_eval_every"],
                      batch=cfg["ddqn_batch"], lr=DDQN_LR, hidden=DDQN_HIDDEN,
                      use_amp=use_amp, verbose=False)
    if save_dir:
        torch.save(qnet.state_dict(), os.path.join(save_dir, f"ddqn_fold{fold_idx}.pt"))

    # eval prep on held-out test fold
    pids, init = get_initial_states_with_ids(test_trans)
    init_s = scaler.transform(init)
    init_t = torch.tensor(init_s, dtype=torch.float32, device=device)
    s_full, _, _, ns_full, _, _ = to_arrays(test_trans)
    ns_t = torch.tensor(scaler.transform(ns_full), dtype=torch.float32, device=device)

    K = K_POLICIES
    pol_a_ns = policy_actions_for_states(qnet, ns_t, K, device)
    pol_a_init = policy_actions_for_states(qnet, init_t, K, device)

    # multi-seed FQE -> per-patient median
    seed_vals = []
    for sd in range(cfg["fqe_seeds"]):
        torch.manual_seed(cfg["seed"] + fold_idx * 100 + sd)
        qe = train_fqe_multi(pol_a_ns, test_trans, scaler, cfg["state_dim"], K, device,
                             n_iters=cfg["fqe_iters"], batch=cfg["fqe_batch"],
                             use_amp=use_amp, vmin=cfg["vmin"], vmax=cfg["vmax"])
        vals = per_patient_values_multi(qe, pol_a_init, init_s, device,
                                        vmin=cfg["vmin"], vmax=cfg["vmax"])  # (K, n)
        seed_vals.append(vals)
    med = np.median(np.stack(seed_vals, axis=0), axis=0)  # (K, n)

    # cross-fitted action distribution on this test fold
    with torch.no_grad():
        ts_t = torch.tensor(scaler.transform(s_full), dtype=torch.float32, device=device)
        ac = qnet(ts_t).argmax(dim=1).cpu().numpy()
    counts = [int((ac == k).sum()) for k in range(NUM_ACTIONS)]

    result = {pid: med[:, i].astype(float) for i, pid in enumerate(pids)}
    return result, counts


def run_folds_on_device(device_str, fold_specs, orig_trans, cfg, save_dir, out_q):
    """Worker: process this device's folds sequentially, push results to queue."""
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    for (fi, train_pats, test_pats) in fold_specs:
        t0 = time.time()
        res, counts = one_fold(fi, train_pats, test_pats, orig_trans, cfg, device,
                               save_dir=save_dir)
        print(f"    [{device_str}] fold {fi} done in {time.time()-t0:.1f}s "
              f"({len(res)} test patients)")
        out_q.put((fi, res, counts))


# ============================================================================
# Bootstrap over patients (cheap: resamples precomputed per-patient values)
# ============================================================================
def bootstrap_over_patients(V, B, seed):
    """V: (n_patients, K). Returns boot matrix (B, K) of resampled means."""
    n, K = V.shape
    rng = np.random.default_rng(seed)
    boot = np.empty((B, K), dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boot[b] = V[idx].mean(axis=0)
    return boot


# ============================================================================
# Main
# ============================================================================
def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fold_dir = os.path.join(args.out_dir, "per_fold_ddqn")
    os.makedirs(fold_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    print(f"\n>>> Devices: {devices} | n_folds={args.n_folds} | fqe_seeds={args.fqe_seeds}")

    # ----- Load + preprocess -----
    print("\n>>> Loading real data ...")
    orig = pd.read_csv('../data/data_eval.csv')
    print(f"orig: {len(orig)} rows, {orig['uid'].nunique()} patients")

    orig_trans = build_transitions(orig)
    print(f"transitions: {len(orig_trans)}")

    # ----- Value-clip bounds from reward range -----
    r_all = orig["reward"].astype(float).values
    vmin = float(r_all.min() / (1.0 - GAMMA))
    vmax = float(r_all.max() / (1.0 - GAMMA))
    print(f"    reward range [{r_all.min():.3f}, {r_all.max():.3f}] "
          f"-> FQE value clip [{vmin:.2f}, {vmax:.2f}]")

    # ----- Build folds over patients -----
    all_pats = sorted(set(t["patient_id"] for t in orig_trans))
    folds = make_folds(all_pats, args.n_folds, args.seed)
    print(f"\n>>> Folds (sizes): {[len(f) for f in folds]}")
    fold_specs = []
    for i, test_pats in enumerate(folds):
        train_pats = set(all_pats) - test_pats
        fold_specs.append((i, train_pats, test_pats))

    cfg = {
        "seed": args.seed, "no_amp": args.no_amp,
        "ddqn_iters": args.ddqn_iters, "ddqn_eval_every": args.ddqn_eval_every,
        "ddqn_batch": args.ddqn_batch,
        "fqe_iters": args.fqe_iters, "fqe_batch": args.fqe_batch,
        "fqe_seeds": args.fqe_seeds,
        "state_dim": len(STATE_COLS), "vmin": vmin, "vmax": vmax,
    }

    # ----- Run folds (parallel across devices, or in-process if single device) -----
    print("\n>>> Running folds ...")
    t0 = time.time()
    results = {}          # pid -> (K,) value vector
    action_counts = [0, 0, 0]

    if len(devices) == 1:
        # in-process, sequential (easy debugging / single GPU / CPU)
        dev = torch.device(devices[0])
        if dev.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        for (fi, trp, tep) in fold_specs:
            tf = time.time()
            res, counts = one_fold(fi, trp, tep, orig_trans, cfg, dev, save_dir=fold_dir)
            print(f"    [{devices[0]}] fold {fi} done in {time.time()-tf:.1f}s "
                  f"({len(res)} test patients)")
            results.update(res)
            for k in range(NUM_ACTIONS):
                action_counts[k] += counts[k]
    else:
        # one process per device; folds round-robined across devices
        ctx = mp.get_context("spawn")
        out_q = ctx.Queue()
        per_dev = defaultdict(list)
        for j, spec in enumerate(fold_specs):
            per_dev[devices[j % len(devices)]].append(spec)
        procs = []
        for dev_str, specs in per_dev.items():
            p = ctx.Process(target=run_folds_on_device,
                            args=(dev_str, specs, orig_trans, cfg, fold_dir, out_q))
            p.start()
            procs.append(p)
        for _ in range(len(fold_specs)):
            fi, res, counts = out_q.get()   # blocks until a fold finishes
            results.update(res)
            for k in range(NUM_ACTIONS):
                action_counts[k] += counts[k]
        for p in procs:
            p.join()

    print(f"    all folds done in {time.time()-t0:.1f}s; "
          f"{len(results)} patients evaluated")

    # ----- Assemble per-patient value matrix -----
    pids = sorted(results.keys())
    V = np.array([results[p] for p in pids], dtype=float)   # (n_pat, K)
    point = V.mean(axis=0)
    print("\n>>> Cross-fitted point estimates (mean over patients):")
    for name, v in zip(POLICY_NAMES, point):
        print(f"    {name:25s} V_hat = {v:.4f}")

    # ----- Bootstrap over patients -----
    boot = bootstrap_over_patients(V, args.bootstrap_B, args.seed)  # (B, K)

    rows = []
    print("\n>>> Per-policy bootstrap CIs (B over patients):")
    for k, name in enumerate(POLICY_NAMES):
        vals = boot[:, k]
        Vhat = float(point[k])
        median = float(np.median(vals))
        errors = vals - Vhat
        ci_lo = Vhat - np.quantile(errors, 0.975)
        ci_hi = Vhat - np.quantile(errors, 0.025)
        pct_lo = float(np.quantile(vals, 0.025))
        pct_hi = float(np.quantile(vals, 0.975))
        rows.append({"policy": name, "point_estimate": Vhat, "median": median,
                     "ci_low": ci_lo, "ci_high": ci_hi,
                     "pct_ci_low": pct_lo, "pct_ci_high": pct_hi})
        print(f"    {name:25s} Vhat={Vhat:7.3f} median={median:7.3f} "
              f"CI=[{ci_lo:7.3f}, {ci_hi:7.3f}]")
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "kfold_summary.csv"), index=False)

    # ----- Paired bootstrap differences (replaces Wilcoxon) -----
    print("\n>>> Paired bootstrap differences (A vs B, diff = A - B):")
    order = sorted(range(K_POLICIES), key=lambda k: np.median(boot[:, k]), reverse=True)
    pair_rows = []
    for ii in range(len(order)):
        for jj in range(ii + 1, len(order)):
            ka, kb = order[ii], order[jj]
            d = boot[:, ka] - boot[:, kb]
            p_gt = float(np.mean(d > 0))
            lo, hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
            crosses0 = bool(lo < 0.0 < hi)
            verdict = "no detectable diff" if crosses0 else f"{POLICY_NAMES[ka]} > {POLICY_NAMES[kb]}"
            pair_rows.append({"A": POLICY_NAMES[ka], "B": POLICY_NAMES[kb],
                              "P(A>B)": p_gt, "median_diff": float(np.median(d)),
                              "ci_low": lo, "ci_high": hi, "crosses_0": crosses0})
            print(f"    {POLICY_NAMES[ka]:20s} vs {POLICY_NAMES[kb]:20s} "
                  f"P={p_gt:.3f} diffCI=[{lo:+.2f},{hi:+.2f}] -> {verdict}")
    pd.DataFrame(pair_rows).to_csv(os.path.join(args.out_dir, "paired_diff.csv"), index=False)

    # ----- Acceptance gate: can we distinguish 'Send a=1'? -----
    print("\n>>> Power check — can the pipeline separate 'Send a=1'?")
    k_a1 = POLICY_NAMES.index("Send a=1")
    separated = []
    for k, name in enumerate(POLICY_NAMES):
        if k == k_a1:
            continue
        d = boot[:, k] - boot[:, k_a1]
        lo, hi = np.quantile(d, 0.025), np.quantile(d, 0.975)
        ok = not (lo < 0 < hi)
        separated.append(ok)
        flag = "YES" if ok else "no"
        print(f"    {name:25s} > Send a=1 ? {flag}  (diffCI=[{lo:+.2f},{hi:+.2f}])")
    print(f"    => pipeline {'HAS' if any(separated) else 'does NOT yet have'} "
          f"power to separate the extreme policy.")

    # ----- Save raw + action distribution -----
    with open(os.path.join(args.out_dir, "boot_values.pkl"), "wb") as f:
        pickle.dump({"per_patient_values": V, "patient_ids": pids,
                     "boot_matrix": boot, "policy_names": POLICY_NAMES,
                     "point_estimates": dict(zip(POLICY_NAMES, point.tolist()))}, f)

    tot = sum(action_counts)
    dist = {f"a={k}": (action_counts[k] / tot if tot else 0.0) for k in range(NUM_ACTIONS)}
    print(f"\n>>> Cross-fitted DDQN action distribution: {dist}")
    with open(os.path.join(args.out_dir, "action_distributions.json"), "w") as f:
        json.dump({"original_cross_fit": dist}, f, indent=2)

    print(f"\n✅ DONE. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
