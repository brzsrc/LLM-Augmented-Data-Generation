"""
Policy Utility Evaluation — K-FOLD CROSS-FITTED, DUAL-GPU
========================================================
Stabilised version. On top of the K-fold + value-clipped FQE + multi-seed median
design, this revision fixes the run-to-run instability we saw (Original V_hat
swinging by ~10 between runs):

  Fix 2  CUDA determinism: cuDNN/TF32 turned off, deterministic algos on,
         CUBLAS_WORKSPACE_CONFIG set. AMP defaults to OFF (--amp opt-in).
  Fix 3  DDQN early-stop dropped (max-Q on a 7-patient val set was a noisy,
         positively-biased selection criterion). Replaced with SWA: parameters
         averaged across the last `--ddqn_swa_keep` checkpoints.
  Fix 4  DDQN now runs `--ddqn_seeds` independent seeds per fold. Per-patient
         V_hat = median over (DDQN_seed x FQE_seed) realisations, so the
         "policy being evaluated" is itself averaged, not a lucky single net.
  Fix 5  No val split inside each fold (Fix 3 made it unnecessary); the ~7
         val patients return to DDQN training, slightly more data per fold.
  Fix 6  GAMMA bumped 0.9 -> 0.95 (effective horizon 10 -> 20 steps), so
         delayed costs (notification fatigue) get reflected in the Bellman
         backup. Value-clip bounds widen accordingly (2x).

Everything else (K-fold over patients, paired-diff CIs, dual-GPU) unchanged.
"""
import argparse
import json
import os

# Must be set BEFORE the first CUDA op for `torch.use_deterministic_algorithms`
# to work with cuBLAS. spawn'd children inherit os.environ.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
    p.add_argument("--n_folds", type=int, default=3)
    p.add_argument("--devices", default="cuda:0,cuda:1",
                   help="Comma-separated torch devices. Folds are round-robined "
                        "across them, one process per device. Use a single value "
                        "(e.g. 'cuda:0' or 'cpu') to run in-process.")

    # DDQN (per fold)
    p.add_argument("--ddqn_iters", type=int, default=80000)
    p.add_argument("--ddqn_eval_every", type=int, default=6000,
                   help="How often to snapshot a DDQN checkpoint for SWA averaging.")
    p.add_argument("--ddqn_batch", type=int, default=512)
    p.add_argument("--ddqn_seeds", type=int, default=3,
                   help="Independent DDQN training seeds per fold (Fix 4).")
    p.add_argument("--ddqn_swa_keep", type=int, default=3,
                   help="Average parameters of the last N checkpoints (Fix 3).")
    p.add_argument("--cql_alpha", type=float, default=1.0,
                   help="CQL conservative-penalty weight. 0 = vanilla DDQN. "
                        "Typical: 0.5–5.0; higher = more conservative (closer to BC). "
                        "Fixes offline-RL extrapolation by pushing down Q on actions "
                        "not in the data.")
    p.add_argument("--gamma", type=float, default=0.95,
                   help="Discount factor for both DDQN and FQE. Effective horizon "
                        "≈ 1/(1-gamma). 0.9 -> 10 steps, 0.95 -> 20 steps, "
                        "0.99 -> 100 steps. Higher = delayed costs (e.g. notification "
                        "fatigue) get more weight; FQE value-clip bounds scale "
                        "automatically with 1/(1-gamma).")

    # FQE (per fold, repeated over seeds)
    p.add_argument("--fqe_iters", type=int, default=20000)
    p.add_argument("--fqe_batch", type=int, default=512)
    p.add_argument("--fqe_seeds", type=int, default=5)

    p.add_argument("--bootstrap_B", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true",
                   help="Enable bf16 autocast (off by default for determinism, Fix 2).")
    p.add_argument("--no_amp", action="store_true",
                   help="(Deprecated; AMP is already off by default. Kept for back-compat.)")
    return p.parse_args()


# ============================================================================
# Constants
# ============================================================================
STATE_COLS = ['study_day', 'weekday', 'slot', 'weather', 'temp', 'loc', 'resp',
              'steps30pre', 'dosage']
NUM_ACTIONS = 3
DDQN_LR = 5.5e-5
DDQN_HIDDEN = 128

POLICY_NAMES = ["Original (cross-fit)", "No message", "Send a=1", "Send a=2"]
K_POLICIES = len(POLICY_NAMES)


# ============================================================================
# Determinism helper (Fix 2)
# ============================================================================
def setup_determinism(seed):
    """Lock all RNGs + force deterministic CUDA. Call in main and in each worker."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# ============================================================================
# Data / MDP
# ============================================================================
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
    by_pid = defaultdict(list)
    for t in trans:
        by_pid[t["patient_id"]].append(t)
    pids = list(by_pid.keys())
    init = np.array([by_pid[p][0]["s"] for p in pids], dtype=np.float32)
    return pids, init


def make_folds(patients, n_folds, seed):
    rng = np.random.default_rng(seed)
    pats = list(patients)
    rng.shuffle(pats)
    return [set(pats[i::n_folds]) for i in range(n_folds)]


# ============================================================================
# Networks
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
        return out


# ============================================================================
# DDQN training — no early-stop, SWA over last N checkpoints  (Fix 3 + Fix 5)
# ============================================================================
def train_ddqn(train_trans, scaler, device,
               n_iters=80000, eval_every=6000,
               batch=512, lr=DDQN_LR, hidden=DDQN_HIDDEN,
               gamma=0.95, tau=1e-4, use_amp=False, verbose=False,
               swa_keep=5, cql_alpha=0.0):
    """Train DDQN, then return a QNet whose parameters are the *average* of the
    last `swa_keep` checkpoints. No val set, no early-stop — both were sources
    of run-to-run noise. SWA across late training is a far less biased way to
    pick weights than max-Q on a 7-patient val.

    If `cql_alpha > 0`, adds the Conservative Q-Learning penalty
        L_cql = E_s[logsumexp_a Q(s,a)] - E_(s,a)~data[Q(s,a)]
    which pushes Q down on actions not seen in the data, preventing the
    offline-RL extrapolation pathology where argmax picks OOD actions that
    only look good because their Q was never grounded by data."""
    s, a, r, ns, d, _ = to_arrays(train_trans)
    s = scaler.transform(s); ns = scaler.transform(ns)
    s = torch.tensor(s, dtype=torch.float32, device=device)
    a = torch.tensor(a, dtype=torch.long, device=device)
    r = torch.tensor(r, dtype=torch.float32, device=device)
    ns = torch.tensor(ns, dtype=torch.float32, device=device)
    d = torch.tensor(d, dtype=torch.float32, device=device)

    state_dim = s.shape[1]
    online = QNet(state_dim, NUM_ACTIONS, hidden).to(device)
    target = QNet(state_dim, NUM_ACTIONS, hidden).to(device)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=lr)

    N = len(s)
    autocast = (use_amp and device.type == "cuda")
    ckpts = []   # rolling buffer of last `swa_keep` parameter snapshots

    for it in range(n_iters):
        idx = torch.randint(0, N, (batch,), device=device)
        sb, ab, rb, nsb, db = s[idx], a[idx], r[idx], ns[idx], d[idx]
        with torch.amp.autocast(device_type="cuda", enabled=autocast, dtype=torch.bfloat16):
            with torch.no_grad():
                next_a = online(nsb).argmax(dim=1, keepdim=True)
                next_q = target(nsb).gather(1, next_a).squeeze(1)
                y = rb + gamma * (1.0 - db) * next_q
            q_all_at_sb = online(sb)                                     # (B, num_actions)
            q_sa = q_all_at_sb.gather(1, ab.unsqueeze(1)).squeeze(1)
            loss_td = F.smooth_l1_loss(q_sa, y)
            if cql_alpha > 0.0:
                # CQL: penalise high Q on unseen actions while leaving Q
                # at the observed (s, a) intact (cancels out in the diff).
                logsumexp_q = torch.logsumexp(q_all_at_sb, dim=1)
                loss_cql = (logsumexp_q - q_sa).mean()
                loss = loss_td + cql_alpha * loss_cql
            else:
                loss = loss_td
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
        opt.step()
        with torch.no_grad():
            for tp, op in zip(target.parameters(), online.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * op.data)

        if (it + 1) % eval_every == 0:
            ckpts.append({k: v.detach().clone() for k, v in online.state_dict().items()})
            if len(ckpts) > swa_keep:
                ckpts.pop(0)
            if verbose:
                print(f"      [ddqn] iter {it+1:6d} loss={loss.item():.4f} "
                      f"(snapshot, kept={len(ckpts)})")

    # SWA: average parameters of the last `swa_keep` checkpoints
    if not ckpts:
        return online
    avg_state = {}
    for k in ckpts[0].keys():
        stacked = torch.stack([c[k].float() for c in ckpts])
        avg_state[k] = stacked.mean(dim=0).to(ckpts[0][k].dtype)
    online.load_state_dict(avg_state)
    return online


# ============================================================================
# FQE — value-clipped, multi-policy
# ============================================================================
def train_fqe_multi(policy_actions_at_ns, eval_trans, scaler, state_dim,
                    n_policies, device,
                    n_iters=20000, batch=512, lr=4e-3, gamma=0.95, tau=0.009,
                    hidden=64, use_amp=False, vmin=None, vmax=None):
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
        ns_a_all = policy_actions_at_ns[:, idx]

        with torch.amp.autocast(device_type="cuda", enabled=autocast, dtype=torch.bfloat16):
            with torch.no_grad():
                tgt_q_all = target(nsb)
                next_q = tgt_q_all.gather(2, ns_a_all.unsqueeze(-1)).squeeze(-1)
                y = rb.unsqueeze(0) + gamma * (1.0 - db).unsqueeze(0) * next_q
                if vmin is not None:
                    y = y.clamp(vmin, vmax)
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
    s = torch.tensor(init_states_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        q_all = qnet_eval(s)
        gathered = q_all.gather(2, policy_actions_at_init.unsqueeze(-1)).squeeze(-1)
        if vmin is not None:
            gathered = gathered.clamp(vmin, vmax)
    return gathered.cpu().numpy()


def policy_actions_for_states(qnet, states_tensor, K, device):
    with torch.no_grad():
        actions = torch.zeros(K, states_tensor.shape[0], dtype=torch.long, device=device)
        actions[0] = qnet(states_tensor).argmax(dim=1)
        actions[1] = 0
        actions[2] = 1
        actions[3] = 2
    return actions


# ============================================================================
# One fold: multi-seed DDQN x multi-seed FQE, median over all realisations
# (Fix 4 + Fix 5)
# ============================================================================
def one_fold(fold_idx, train_pats, test_pats, orig_trans, cfg, device,
             save_dir=None):
    setup_determinism(cfg["seed"] + fold_idx)
    use_amp = cfg["use_amp"] and device.type == "cuda"

    train_trans = filter_trans(orig_trans, train_pats)
    test_trans = filter_trans(orig_trans, test_pats)

    # scaler fit on ALL of this fold's train transitions (Fix 5: no val split)
    s_tr, _, _, _, _, _ = to_arrays(train_trans)
    scaler = MinMaxScaler().fit(s_tr)

    # eval-side tensors (don't depend on which DDQN seed)
    pids, init = get_initial_states_with_ids(test_trans)
    init_s = scaler.transform(init)
    init_t = torch.tensor(init_s, dtype=torch.float32, device=device)
    s_full, _, _, ns_full, _, _ = to_arrays(test_trans)
    ns_t = torch.tensor(scaler.transform(ns_full), dtype=torch.float32, device=device)

    K = K_POLICIES
    all_vals = []                    # one (K, n_pat) array per (ddqn_seed x fqe_seed)
    action_counts = [0, 0, 0]        # accumulated across all DDQN seeds in this fold

    for ds in range(cfg["ddqn_seeds"]):
        # distinct seed per (fold, ddqn_seed)
        torch.manual_seed(cfg["seed"] + fold_idx * 1000 + ds * 100)
        qnet = train_ddqn(train_trans, scaler, device,
                          n_iters=cfg["ddqn_iters"], eval_every=cfg["ddqn_eval_every"],
                          batch=cfg["ddqn_batch"], lr=DDQN_LR, hidden=DDQN_HIDDEN,
                          gamma=cfg["gamma"], use_amp=use_amp, verbose=False,
                          swa_keep=cfg["ddqn_swa_keep"],
                          cql_alpha=cfg["cql_alpha"])
        if save_dir:
            torch.save(qnet.state_dict(),
                       os.path.join(save_dir, f"ddqn_fold{fold_idx}_seed{ds}.pt"))

        # action distribution from THIS DDQN seed
        with torch.no_grad():
            ts_t = torch.tensor(scaler.transform(s_full), dtype=torch.float32, device=device)
            ac = qnet(ts_t).argmax(dim=1).cpu().numpy()
        for k in range(NUM_ACTIONS):
            action_counts[k] += int((ac == k).sum())

        # this DDQN's policy actions at test next-states / init states
        pol_a_ns = policy_actions_for_states(qnet, ns_t, K, device)
        pol_a_init = policy_actions_for_states(qnet, init_t, K, device)

        for fs in range(cfg["fqe_seeds"]):
            torch.manual_seed(cfg["seed"] + fold_idx * 1000 + ds * 100 + fs + 1)
            qe = train_fqe_multi(pol_a_ns, test_trans, scaler, cfg["state_dim"], K, device,
                                 n_iters=cfg["fqe_iters"], batch=cfg["fqe_batch"],
                                 gamma=cfg["gamma"], use_amp=use_amp,
                                 vmin=cfg["vmin"], vmax=cfg["vmax"])
            vals = per_patient_values_multi(qe, pol_a_init, init_s, device,
                                            vmin=cfg["vmin"], vmax=cfg["vmax"])
            all_vals.append(vals)

    # median over ALL (ddqn_seed x fqe_seed) realisations -> robust per-patient V
    med = np.median(np.stack(all_vals, axis=0), axis=0)   # (K, n_pat)
    result = {pid: med[:, i].astype(float) for i, pid in enumerate(pids)}
    return result, action_counts


def run_folds_on_device(device_str, fold_specs, orig_trans, cfg, save_dir, out_q):
    """Worker: deterministic setup, then process its folds sequentially."""
    setup_determinism(cfg["seed"])
    device = torch.device(device_str)
    for (fi, train_pats, test_pats) in fold_specs:
        t0 = time.time()
        res, counts = one_fold(fi, train_pats, test_pats, orig_trans, cfg, device,
                               save_dir=save_dir)
        print(f"    [{device_str}] fold {fi} done in {time.time()-t0:.1f}s "
              f"({len(res)} test patients)")
        out_q.put((fi, res, counts))


# ============================================================================
# Bootstrap over patients (cheap; resamples precomputed per-patient values)
# ============================================================================
def bootstrap_over_patients(V, B, seed):
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
    setup_determinism(args.seed)

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    print(f"\n>>> Devices: {devices} | n_folds={args.n_folds} "
          f"| ddqn_seeds={args.ddqn_seeds} | fqe_seeds={args.fqe_seeds} "
          f"| cql_alpha={args.cql_alpha} "
          f"| gamma={args.gamma} | amp={args.amp}")
    print(">>> Determinism: cuDNN.det=True, cuDNN.bench=False, TF32 off, "
          "CUBLAS_WORKSPACE_CONFIG set.")

    # ----- Load + preprocess -----
    print("\n>>> Loading real data ...")
    orig = pd.read_csv('../data/data_eval.csv')
    print(f"orig: {len(orig)} rows, {orig['uid'].nunique()} patients")

    orig_trans = build_transitions(orig)
    print(f"transitions: {len(orig_trans)}")

    # ----- Value-clip bounds from reward range (scales with 1/(1-gamma)) -----
    r_all = orig["reward"].astype(float).values
    vmin = float(r_all.min() / (1.0 - args.gamma))
    vmax = float(r_all.max() / (1.0 - args.gamma))
    print(f"    reward range [{r_all.min():.3f}, {r_all.max():.3f}] "
          f"-> FQE value clip [{vmin:.2f}, {vmax:.2f}] (gamma={args.gamma})")

    # ----- Folds -----
    all_pats = sorted(set(t["patient_id"] for t in orig_trans))
    folds = make_folds(all_pats, args.n_folds, args.seed)
    print(f"\n>>> Folds (sizes): {[len(f) for f in folds]}")
    fold_specs = []
    for i, test_pats in enumerate(folds):
        train_pats = set(all_pats) - test_pats
        fold_specs.append((i, train_pats, test_pats))

    cfg = {
        "seed": args.seed, "use_amp": args.amp,
        "ddqn_iters": args.ddqn_iters, "ddqn_eval_every": args.ddqn_eval_every,
        "ddqn_batch": args.ddqn_batch,
        "ddqn_seeds": args.ddqn_seeds, "ddqn_swa_keep": args.ddqn_swa_keep,
        "cql_alpha": args.cql_alpha,
        "fqe_iters": args.fqe_iters, "fqe_batch": args.fqe_batch,
        "fqe_seeds": args.fqe_seeds,
        "state_dim": len(STATE_COLS), "vmin": vmin, "vmax": vmax,
        "gamma": args.gamma,
    }

    # ----- Run folds -----
    print("\n>>> Running folds ...")
    t0 = time.time()
    results = {}
    action_counts = [0, 0, 0]

    if len(devices) == 1:
        dev = torch.device(devices[0])
        for (fi, trp, tep) in fold_specs:
            tf = time.time()
            res, counts = one_fold(fi, trp, tep, orig_trans, cfg, dev, save_dir=fold_dir)
            print(f"    [{devices[0]}] fold {fi} done in {time.time()-tf:.1f}s "
                  f"({len(res)} test patients)")
            results.update(res)
            for k in range(NUM_ACTIONS):
                action_counts[k] += counts[k]
    else:
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
            fi, res, counts = out_q.get()
            results.update(res)
            for k in range(NUM_ACTIONS):
                action_counts[k] += counts[k]
        for p in procs:
            p.join()

    print(f"    all folds done in {time.time()-t0:.1f}s; "
          f"{len(results)} patients evaluated")

    # ----- Per-patient value matrix -----
    pids = sorted(results.keys())
    V = np.array([results[p] for p in pids], dtype=float)
    point = V.mean(axis=0)
    print("\n>>> Cross-fitted point estimates (mean over patients):")
    for name, v in zip(POLICY_NAMES, point):
        print(f"    {name:25s} V_hat = {v:.4f}")

    # ----- Bootstrap CI -----
    boot = bootstrap_over_patients(V, args.bootstrap_B, args.seed)
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

    # ----- Paired diff -----
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

    # ----- Power gate -----
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
                     "point_estimates": dict(zip(POLICY_NAMES, point.tolist())),
                     "config": {"gamma": args.gamma, "ddqn_seeds": args.ddqn_seeds,
                                "fqe_seeds": args.fqe_seeds,
                                "ddqn_swa_keep": args.ddqn_swa_keep,
                                "cql_alpha": args.cql_alpha,
                                "amp": args.amp, "seed": args.seed,
                                "n_folds": args.n_folds}}, f)

    tot = sum(action_counts)
    dist = {f"a={k}": (action_counts[k] / tot if tot else 0.0) for k in range(NUM_ACTIONS)}
    print(f"\n>>> Cross-fitted DDQN action distribution (averaged over {args.ddqn_seeds} seeds): {dist}")
    with open(os.path.join(args.out_dir, "action_distributions.json"), "w") as f:
        json.dump({"original_cross_fit": dist,
                   "averaged_over_ddqn_seeds": args.ddqn_seeds}, f, indent=2)

    print(f"\n✅ DONE. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
