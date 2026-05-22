"""
Policy Utility Evaluation — TRAIN-SET-ONLY BASELINE
====================================================
Baseline version of policy_utility_fast.py:
- Trains ONE DDQN on the real training set (no synthetic data)
- Compares against 3 fixed policies (No message / Send a=1 / Send a=2)
- FQE-based off-policy evaluation on the held-out real test set
- B=150 bootstrap for CIs

This is a standalone file — no dependency on policy_utility_fast.py.
The DDQN / FQE training code is identical (copy-paste), so any future
improvement made there should be ported here.

OUTPUTS
-------
- ddqn_original.pt              — trained DDQN weights
- fqe_summary.csv               — point estimates + CI for all 4 policies
- boot_values.pkl               — raw bootstrap values
- wilcoxon.csv                  — pairwise significance (best vs others)
- action_distributions.json     — DDQN action distribution on test set

USAGE
-----
    python policy_utility_baseline.py \
        --orig_csv data/cleaned_output.csv \
        --train_uids data/train_uids.json \
        --test_uids data/test_uids.json \
        --out_dir outputs/baseline
"""
import argparse
import json
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sstats
from sklearn.preprocessing import MinMaxScaler


# ============================================================================
# CLI
# ============================================================================
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--orig_csv", required=True,
                   help="Real HeartSteps CSV (has uid, datetime, send, jbsteps30, etc.)")
    p.add_argument("--out_dir", default="./outputs/baseline")
    p.add_argument("--train_uids", default="../data/train_uids.json")
    p.add_argument("--test_uids", default="../data/test_uids.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # DDQN
    p.add_argument("--ddqn_iters", type=int, default=80000)
    p.add_argument("--ddqn_eval_every", type=int, default=6000)
    p.add_argument("--ddqn_batch", type=int, default=512)

    # FQE — point estimate (full test set)
    p.add_argument("--fqe_iters_point", type=int, default=20000)
    # FQE — bootstrap (smaller samples → fewer iters)
    p.add_argument("--fqe_iters_boot", type=int, default=20000)
    p.add_argument("--fqe_batch", type=int, default=512)

    p.add_argument("--bootstrap_B", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--no_amp", action="store_true",
                   help="Disable bf16 autocast (default: enabled on cuda).")
    return p.parse_args()


# ============================================================================
# Constants
# ============================================================================
STATE_COLS = ["slot", "study_day", "weekend", "dosage", "weather",
              "temperature", "jbsteps30pre", "loc_enc"]
NUM_ACTIONS = 3
GAMMA = 0.9

# DDQN hyperparams for the 'original' (real-data-only) policy — same as
# policy_utility_fast.py 'original' entry, kept here for self-containment.
DDQN_LR = 5.5e-5
DDQN_HIDDEN = 128


# ============================================================================
# Data loading & MDP construction
# ============================================================================
def load_and_preprocess(orig_csv):
    """Same preprocessing as policy_utility_fast.py but for the real CSV only.
    Produces dosage, weekend, slot, loc_enc, act_enc, reward_calc columns.
    """
    orig = pd.read_csv(orig_csv)

    orig["avail"] = orig["avail"].astype(int)
    orig["date_dt"] = pd.to_datetime(orig["date"])
    orig["study_day"] = orig.groupby("uid")["date_dt"].transform(
        lambda s: (s - s.min()).dt.days + 1)
    orig["weekend"] = (orig["date_dt"].dt.dayofweek >= 5).astype(int)
    orig["slot"] = orig["day_slot"]

    orig = orig.sort_values(["uid", "datetime"]).reset_index(drop=True)
    dosages = []
    for _, g in orig.groupby("uid", sort=False):
        d, prev_a = 0.0, 0
        out = []
        for a in g["send"].values:
            d = 0.95 * d + (1 if prev_a > 0 else 0)
            out.append(d)
            prev_a = a
        dosages.extend(out)
    orig["dosage"] = dosages

    orig["reward_calc"] = np.log(orig["jbsteps30"].astype(float) + 0.5)

    # Temperature: ordinal encoding (freezing<cold<cool<mild<warm<hot)
    # Required because STATE_COLS includes "temperature" but the CSV stores it
    # as a string. If your real preprocessing differs, edit TEMP_ORDER below.
    TEMP_ORDER = {"temp_freezing": 0, "temp_cold": 1, "temp_cool": 2,
                  "temp_mild": 3, "temp_warm": 4, "temp_hot": 5}
    if not pd.api.types.is_numeric_dtype(orig["temperature"]):
        unknown_temps = set(orig["temperature"].unique()) - set(TEMP_ORDER.keys())
        if unknown_temps:
            print(f"    WARNING: unknown temperature labels: {unknown_temps} (mapped to NaN)")
        orig["temperature"] = orig["temperature"].map(TEMP_ORDER).astype(int)

    WEATHER_ORDER = {"Clear": 0,"MostlyCloudy": 1, "PartlyCloudy": 2,"Overcast": 3, "Fog": 4, "Rain": 5,"Snow": 6}
    if not pd.api.types.is_numeric_dtype(orig["weather"]):
        unknown_weather = set(orig["weather"].unique()) - set(WEATHER_ORDER.keys())
        if unknown_weather:
            print(f"    WARNING: unknown weather labels: {unknown_weather}")
        orig["weather"] = orig["weather"].map(WEATHER_ORDER).astype(int)

    # Categorical encoding (just on real data — no sim to merge)
    loc_cats = sorted(set(orig["location"].astype(str)))
    act_cats = sorted(set(orig["activity"].astype(str)))
    loc_map = {c: i for i, c in enumerate(loc_cats)}
    act_map = {c: i for i, c in enumerate(act_cats)}
    orig["loc_enc"] = orig["location"].astype(str).map(loc_map).astype(int)
    orig["act_enc"] = orig["activity"].astype(str).map(act_map).astype(int)

    return orig


def build_transitions(df, user_col, sort_cols):
    df = df.sort_values([user_col] + sort_cols).reset_index(drop=True)
    transitions = []
    for uid, g in df.groupby(user_col, sort=False):
        g = g.reset_index(drop=True)
        S = g[STATE_COLS].astype(float).values
        A = g["send"].astype(int).values
        R = g["reward_calc"].astype(float).values
        n = len(g)
        for t in range(n):
            ns = S[t + 1] if t < n - 1 else S[t]
            done = 0.0 if t < n - 1 else 1.0
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


def make_splits(orig_trans, train_uids, test_uids, seed=42):
    """Train DDQN uses 75% of train_uids; remaining 25% is val for early-stop.
    Test set = test_uids ∩ data (for FQE)."""
    rng = np.random.default_rng(seed)
    orig_pats_all = set(t["patient_id"] for t in orig_trans)
    expected = train_uids | test_uids
    extra_in_data = orig_pats_all - expected
    extra_in_split = expected - orig_pats_all
    if extra_in_data:
        print(f"    WARNING: orig patients not in split file: {sorted(extra_in_data)}")
    if extra_in_split:
        print(f"    WARNING: split-file uids not in data:     {sorted(extra_in_split)}")

    orig_train_all = sorted(train_uids & orig_pats_all)
    rng.shuffle(orig_train_all)
    n_val = max(3, len(orig_train_all) // 4)
    orig_val_pats = set(orig_train_all[:n_val])
    orig_train_pats = set(orig_train_all[n_val:])

    def f(trans, pats):
        return [t for t in trans if t["patient_id"] in pats]

    test_pats = test_uids & orig_pats_all
    train_set = f(orig_trans, orig_train_pats)
    val_set = f(orig_trans, orig_val_pats)
    test_set = f(orig_trans, test_pats)
    return train_set, val_set, test_set, test_pats


# ============================================================================
# Networks
# ============================================================================
class QNet(nn.Module):
    """Single Q-network for DDQN training."""
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
    """K independent Q-nets stacked into one tensor (parameter dim 0 = policy index).

    Same architecture as policy_utility_fast.py — we keep it so that the
    multi-policy FQE training pass works for K=4 in one shot.
    """
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
# DDQN training
# ============================================================================
def train_ddqn(train_trans, val_trans, scaler, device,
               n_iters=80000, eval_every=6000,
               batch=512, lr=DDQN_LR, hidden=DDQN_HIDDEN,
               gamma=GAMMA, tau=1e-4, use_amp=True, verbose=True):
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
                print(f"    iter {it+1:6d}  loss={loss.item():.4f}  val_max_q={v:.4f}")

    online.load_state_dict(best_state)
    return online


# ============================================================================
# FQE — train K policies simultaneously
# ============================================================================
def train_fqe_multi(policy_actions_at_ns, eval_trans, scaler, state_dim,
                    n_policies, device,
                    n_iters=20000, batch=512, lr=4e-3, gamma=GAMMA, tau=0.009,
                    hidden=64, use_amp=True):
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
        sb = s[idx]
        ab = a[idx]
        rb = r[idx]
        nsb = ns[idx]
        db = d[idx]
        ns_a_all = policy_actions_at_ns[:, idx]   # (K, B)

        with torch.amp.autocast(device_type="cuda", enabled=autocast, dtype=torch.bfloat16):
            with torch.no_grad():
                tgt_q_all = target(nsb)
                next_q = tgt_q_all.gather(2, ns_a_all.unsqueeze(-1)).squeeze(-1)
                y = rb.unsqueeze(0) + gamma * (1.0 - db).unsqueeze(0) * next_q
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


def value_estimate_multi(qnet_eval, policy_actions_at_init, init_states_scaled, device):
    s = torch.tensor(init_states_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        q_all = qnet_eval(s)
        gathered = q_all.gather(2, policy_actions_at_init.unsqueeze(-1)).squeeze(-1)
    return gathered.mean(dim=1).cpu().numpy()


def get_initial_states(trans):
    by_pid = defaultdict(list)
    for t in trans:
        by_pid[t["patient_id"]].append(t)
    return np.array([ts[0]["s"] for ts in by_pid.values()], dtype=np.float32)


# ============================================================================
# Main
# ============================================================================
def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"\n>>> Using device: {device} | amp(bf16): {use_amp}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ----- Load + preprocess (real data only) -----
    print("\n>>> Loading real data ...")
    orig = load_and_preprocess(args.orig_csv)
    print(f"    orig: {len(orig)} rows, {orig['uid'].nunique()} patients")

    print("\n>>> Building transitions ...")
    orig_trans = build_transitions(orig, "uid", ["datetime"])
    print(f"    orig transitions: {len(orig_trans)}")

    print("\n>>> Loading fixed train/test split ...")
    with open(args.train_uids) as f:
        train_uids = set(json.load(f))
    with open(args.test_uids) as f:
        test_uids = set(json.load(f))
    print(f"    train uids: {len(train_uids)}, test uids: {len(test_uids)}")

    train_set, val_set, test_set, test_pats = make_splits(
        orig_trans, train_uids, test_uids, seed=args.seed)
    print(f"    train transitions: {len(train_set)}, val transitions: {len(val_set)}")
    print(f"    test patients:     {len(test_pats)}, test transitions: {len(test_set)}")

    # ----- Step 1: Train DDQN on train_set -----
    state_dim = len(STATE_COLS)
    s_tr, _, _, _, _, _ = to_arrays(train_set)
    scaler = MinMaxScaler().fit(s_tr)

    print("\n>>> Step 1 — Train DDQN on real train_set only")
    t0 = time.time()
    qnet = train_ddqn(train_set, val_set, scaler, device,
                      n_iters=args.ddqn_iters,
                      eval_every=args.ddqn_eval_every,
                      batch=args.ddqn_batch,
                      lr=DDQN_LR, hidden=DDQN_HIDDEN,
                      use_amp=use_amp, verbose=True)
    print(f"    done in {time.time() - t0:.1f}s")
    torch.save(qnet.state_dict(), os.path.join(args.out_dir, "ddqn_original.pt"))

    # ----- Build policies: 1 learned + 3 fixed -----
    POLICY_NAMES = ["Original (train-only)", "No message", "Send a=1", "Send a=2"]
    K = len(POLICY_NAMES)

    def policy_actions_for_states(states_tensor):
        """Compute (K, N) action tensor: index 0 = learned, 1-3 = fixed."""
        with torch.no_grad():
            actions = torch.zeros(K, states_tensor.shape[0], dtype=torch.long, device=device)
            actions[0] = qnet(states_tensor).argmax(dim=1)
            actions[1] = 0    # No message
            actions[2] = 1    # Send a=1
            actions[3] = 2    # Send a=2
        return actions

    # ----- Step 2a: Point estimates on full test set -----
    print("\n>>> Step 2a — Point estimates on full test set (multi-policy FQE)")
    t0 = time.time()
    s_full, _, _, ns_full, _, _ = to_arrays(test_set)
    ns_full_t = torch.tensor(scaler.transform(ns_full), dtype=torch.float32, device=device)
    init_full = scaler.transform(get_initial_states(test_set))
    init_full_t = torch.tensor(init_full, dtype=torch.float32, device=device)

    pol_a_at_ns_full = policy_actions_for_states(ns_full_t)
    pol_a_at_init_full = policy_actions_for_states(init_full_t)

    qe = train_fqe_multi(pol_a_at_ns_full, test_set, scaler, state_dim, K, device,
                         n_iters=args.fqe_iters_point, batch=args.fqe_batch,
                         use_amp=use_amp)
    Vs = value_estimate_multi(qe, pol_a_at_init_full, init_full, device)
    point_estimates = {name: float(v) for name, v in zip(POLICY_NAMES, Vs)}
    print(f"    Point-estimate FQE done in {time.time()-t0:.1f}s")
    for name, v in point_estimates.items():
        print(f"      {name:25s}  V_hat = {v:.4f}")

    # ----- Step 2b: Bootstrap -----
    print(f"\n>>> Step 2b — Bootstrapping (B={args.bootstrap_B}) — multi-policy FQE")
    test_by_pid = defaultdict(list)
    for t in test_set:
        test_by_pid[t["patient_id"]].append(t)
    test_pids_list = list(test_by_pid.keys())

    rng = np.random.default_rng(args.seed)
    boot_values = {name: [] for name in POLICY_NAMES}
    t_start = time.time()

    for b in range(args.bootstrap_B):
        sampled_pids = rng.choice(test_pids_list, size=len(test_pids_list), replace=True)
        boot_trans = []
        for pid in sampled_pids:
            boot_trans.extend(test_by_pid[pid])
        if len(boot_trans) < 32:
            for name in POLICY_NAMES:
                boot_values[name].append(np.nan)
            continue

        s_b, _, _, ns_b, _, _ = to_arrays(boot_trans)
        ns_b_t = torch.tensor(scaler.transform(ns_b), dtype=torch.float32, device=device)
        init_b = scaler.transform(get_initial_states(boot_trans))
        init_b_t = torch.tensor(init_b, dtype=torch.float32, device=device)

        pol_a_at_ns = policy_actions_for_states(ns_b_t)
        pol_a_at_init = policy_actions_for_states(init_b_t)

        qe_b = train_fqe_multi(pol_a_at_ns, boot_trans, scaler, state_dim, K, device,
                               n_iters=args.fqe_iters_boot, batch=args.fqe_batch,
                               use_amp=use_amp)
        Vs_b = value_estimate_multi(qe_b, pol_a_at_init, init_b, device)
        for name, v in zip(POLICY_NAMES, Vs_b):
            boot_values[name].append(float(v))

        if (b + 1) % 5 == 0 or b == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (b + 1) * (args.bootstrap_B - b - 1)
            print(f"    boot {b+1}/{args.bootstrap_B}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    with open(os.path.join(args.out_dir, "boot_values.pkl"), "wb") as f:
        pickle.dump({"boot_values": boot_values,
                     "point_estimates": point_estimates,
                     "test_patients": list(test_pats)}, f)

    # ----- Summary CIs -----
    print("\n>>> Bootstrap summary")
    rows = []
    for name in POLICY_NAMES:
        Vhat = point_estimates[name]
        vals = np.array(boot_values[name], dtype=float)
        vals = vals[~np.isnan(vals)]
        median = float(np.median(vals)) if len(vals) else float("nan")
        if len(vals) >= 2:
            errors = vals - Vhat
            ci_lo = Vhat - np.quantile(errors, 0.975)
            ci_hi = Vhat - np.quantile(errors, 0.025)
            pct_lo = float(np.quantile(vals, 0.025))
            pct_hi = float(np.quantile(vals, 0.975))
        else:
            ci_lo = ci_hi = pct_lo = pct_hi = float("nan")
        rows.append({"policy": name, "point_estimate": Vhat,
                     "median": median, "ci_low": ci_lo, "ci_high": ci_hi,
                     "pct_ci_low": pct_lo, "pct_ci_high": pct_hi,
                     "n_boot_valid": int(len(vals))})
        print(f"    {name:25s}  Vhat={Vhat:7.4f}  median={median:7.4f}  "
              f"CI(thesis)=[{ci_lo:7.4f}, {ci_hi:7.4f}]  CI(pct)=[{pct_lo:7.4f}, {pct_hi:7.4f}]")
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "fqe_summary.csv"), index=False)

    # ----- Wilcoxon -----
    print("\n>>> Wilcoxon signed-rank tests (best vs others, one-sided 'greater')")
    medians = {n: np.nanmedian(boot_values[n]) for n in POLICY_NAMES}
    best = max(medians, key=medians.get)
    print(f"    Best policy: '{best}' (median={medians[best]:.4f})")

    wilc_rows = []
    for name in POLICY_NAMES:
        if name == best:
            continue
        a_arr = np.array(boot_values[best], dtype=float)
        b_arr = np.array(boot_values[name], dtype=float)
        m = ~(np.isnan(a_arr) | np.isnan(b_arr))
        diff = a_arr[m] - b_arr[m]
        try:
            W, p = sstats.wilcoxon(diff, alternative="greater")
        except ValueError:
            W, p = float("nan"), float("nan")
        wilc_rows.append({"best": best, "compared_to": name, "W": W, "p": p})
        print(f"    H1: '{best}' > '{name}'   W={W:.0f}   p={p:.4g}")
    pd.DataFrame(wilc_rows).to_csv(os.path.join(args.out_dir, "wilcoxon.csv"), index=False)

    # ----- Action distribution of learned policy on test set -----
    print("\n>>> Action distribution of learned policy on full test set")
    ts, _, _, _, _, _ = to_arrays(test_set)
    ts_t = torch.tensor(scaler.transform(ts), dtype=torch.float32, device=device)
    with torch.no_grad():
        ac = qnet(ts_t).argmax(dim=1).cpu().numpy()
    dist = {f"a={k}": float((ac == k).mean()) for k in range(NUM_ACTIONS)}
    print(f"    Original (train-only)  {dist}")
    with open(os.path.join(args.out_dir, "action_distributions.json"), "w") as f:
        json.dump({"original_train_only": dist}, f, indent=2)

    print(f"\n✅ DONE. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
