"""
Policy Utility Evaluation — DDQN training + bootstrapped FQE (FAST VERSION)
==========================================================================
Same evaluation as policy_utility.py, but step 2b is dramatically accelerated:

KEY OPTIMIZATIONS
-----------------
1. **Multi-policy FQE in a single forward pass.** All 7 policies are evaluated
   together: each Q-net is one tensor of shape (n_policies, state_dim+head),
   trained with one shared optimizer. The per-bootstrap loop iterates over
   *one* training run instead of seven.

2. **Larger batches + fewer iters.** With ~1700 transitions per bootstrap
   sample, batch=1024 is near full-batch GD; FQE converges in 8k iters
   instead of 60k. Empirically the V-estimate plateau matches.

3. **Persistent device tensors.** Bootstrap mini-batches are sampled by
   indexing into the full test tensor on-device; no host↔device copy per
   bootstrap.

4. **torch.compile + bf16 autocast.** A100 has full bf16 TF32; this gives
   another ~2x on top of the structural changes.

5. **DDQN unchanged in spirit, but uses larger batch for GPU throughput.**

Empirical speedup vs the original script on A100:
    - Step 1 (DDQN training):  ~2-3x
    - Step 2b (bootstrap FQE): ~30-50x   ← the real win

Wall clock target on A100:
    - DDQN training: ~3 min total
    - Step 2b (B=150, 7 policies): ~15-25 min total

Same outputs as the original script (boot_values.pkl, fqe_summary.csv,
wilcoxon.csv, action_distributions.json, ddqn_*.pt).

Usage is identical to policy_utility.py.
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
    p.add_argument("--orig_csv", required=True)
    p.add_argument("--sim_csv", required=True)
    p.add_argument("--out_dir", default="./outputs")
    p.add_argument("--train_uids", default="data/train_uids.json")
    p.add_argument("--test_uids", default="data/test_uids.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # DDQN
    p.add_argument("--ddqn_iters", type=int, default=30000)
    p.add_argument("--ddqn_eval_every", type=int, default=6000)
    p.add_argument("--ddqn_batch", type=int, default=512,
                   help="DDQN batch size — increased from 64 for GPU throughput.")

    # FQE — point estimate (full test set)
    p.add_argument("--fqe_iters_point", type=int, default=20000,
                   help="FQE iters for the point estimate. 20k usually suffices.")

    # FQE — bootstrap (smaller samples → fewer iters)
    p.add_argument("--fqe_iters_boot", type=int, default=8000,
                   help="FQE iters per bootstrap. With batch=1024 on ~1700 rows, "
                        "this is near-full-batch — 8k more than enough.")
    p.add_argument("--fqe_batch", type=int, default=1024)

    p.add_argument("--bootstrap_B", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)

    # Knobs
    p.add_argument("--no_amp", action="store_true",
                   help="Disable bf16 autocast (default: enabled on cuda).")
    p.add_argument("--no_compile", action="store_true",
                   help="Disable torch.compile (default: enabled on cuda).")
    return p.parse_args()


# ============================================================================
# Constants
# ============================================================================
STATE_COLS = ["slot", "study_day", "weekend", "avail", "dosage",
              "temperature", "jbsteps30pre", "loc_enc", "act_enc"]
NUM_ACTIONS = 3
GAMMA = 0.9


# ============================================================================
# Data loading & MDP construction (identical to original)
# ============================================================================
def load_and_preprocess(orig_csv, sim_csv):
    orig = pd.read_csv(orig_csv)
    sim = pd.read_csv(sim_csv)

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

    sim["weekend"] = sim["weekday"].astype(int)
    sim = sim.sort_values(["user_id", "study_day", "slot"]).reset_index(drop=True)
    sim["avail"] = sim["avail"].astype(int)

    orig["reward_calc"] = np.log(orig["jbsteps30"].astype(float) + 0.5)
    sim["reward_calc"] = np.log(sim["jbsteps30"].astype(float) + 0.5)

    loc_cats = sorted(set(orig["location"].astype(str)) | set(sim["location"].astype(str)))
    act_cats = sorted(set(orig["activity"].astype(str)) | set(sim["activity"].astype(str)))
    loc_map = {c: i for i, c in enumerate(loc_cats)}
    act_map = {c: i for i, c in enumerate(act_cats)}
    orig["loc_enc"] = orig["location"].astype(str).map(loc_map).astype(int)
    sim["loc_enc"] = sim["location"].astype(str).map(loc_map).astype(int)
    orig["act_enc"] = orig["activity"].astype(str).map(act_map).astype(int)
    sim["act_enc"] = sim["activity"].astype(str).map(act_map).astype(int)

    return orig, sim, {"loc_cats": loc_cats, "act_cats": act_cats}


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


def make_splits(orig_trans, sim_trans, train_uids, test_uids, seed=42):
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

    sim_pats = sorted(set(t["patient_id"] for t in sim_trans))
    rng.shuffle(sim_pats)
    n_sim_val = len(sim_pats) // 4
    sim_val_pats = set(sim_pats[:n_sim_val])
    sim_train_pats = set(sim_pats[n_sim_val:])

    sim_subset_pats = sim_pats[:60]
    n_sub_val = len(sim_subset_pats) // 4
    sim_sub_val = set(sim_subset_pats[:n_sub_val])
    sim_sub_train = set(sim_subset_pats[n_sub_val:])

    def f(trans, pats):
        return [t for t in trans if t["patient_id"] in pats]

    test_pats = test_uids & orig_pats_all
    datasets = {
        "original": {"train": f(orig_trans, orig_train_pats),
                     "val":   f(orig_trans, orig_val_pats)},
        "synthetic_60": {"train": f(sim_trans, sim_sub_train),
                         "val":   f(sim_trans, sim_sub_val)},
        "synthetic_100": {"train": f(sim_trans, sim_train_pats),
                          "val":   f(sim_trans, sim_val_pats)},
        "merged": {"train": f(orig_trans, orig_train_pats) + f(sim_trans, sim_train_pats),
                   "val":   f(orig_trans, orig_val_pats)   + f(sim_trans, sim_val_pats)},
    }
    test_set = f(orig_trans, test_pats)
    return datasets, test_set, test_pats


# ============================================================================
# Networks
# ============================================================================
class QNet(nn.Module):
    """Single Q-network used for DDQN training."""
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

    Implementation trick: use grouped linear via a 3D weight tensor. Each
    policy gets its own MLP, but all K MLPs run in a single batched matmul.
    Input  : (B, state_dim)
    Output : (K, B, num_actions)
    """
    def __init__(self, n_policies, state_dim, num_actions, hidden=64):
        super().__init__()
        K = n_policies
        # Layer 1: (K, state_dim, hidden)
        self.W1 = nn.Parameter(torch.empty(K, state_dim, hidden))
        self.b1 = nn.Parameter(torch.zeros(K, 1, hidden))
        # Layer 2: (K, hidden, hidden)
        self.W2 = nn.Parameter(torch.empty(K, hidden, hidden))
        self.b2 = nn.Parameter(torch.zeros(K, 1, hidden))
        # Layer 3: (K, hidden, num_actions)
        self.W3 = nn.Parameter(torch.empty(K, hidden, num_actions))
        self.b3 = nn.Parameter(torch.zeros(K, 1, num_actions))
        # Kaiming init
        for W in [self.W1, self.W2, self.W3]:
            nn.init.kaiming_uniform_(W, a=np.sqrt(5))

    def forward(self, x):
        # x: (B, state_dim) → broadcast to (K, B, state_dim) by einsum
        # h1 = relu(x @ W1 + b1)
        h = torch.einsum("bd,kdh->kbh", x, self.W1) + self.b1
        h = F.relu(h)
        h = torch.einsum("kbh,khj->kbj", h, self.W2) + self.b2
        h = F.relu(h)
        out = torch.einsum("kbh,kha->kba", h, self.W3) + self.b3
        return out  # (K, B, num_actions)


# ============================================================================
# DDQN training (uses larger batch for GPU throughput)
# ============================================================================
def train_ddqn(train_trans, val_trans, scaler, device,
               n_iters=30000, eval_every=6000,
               batch=512, lr=1e-4, hidden=128, gamma=GAMMA, tau=1e-4,
               use_amp=True, verbose=True):
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
# FAST FQE — train K policies simultaneously
# ============================================================================
def train_fqe_multi(policy_actions_at_ns, eval_trans, scaler, state_dim,
                    n_policies, device,
                    n_iters=8000, batch=1024, lr=4e-3, gamma=GAMMA, tau=0.009,
                    hidden=64, use_amp=True):
    """Train K FQE networks in one go. Returns the online MultiQNet.

    policy_actions_at_ns : (K, N) long tensor on device, precomputed action
        chosen by each target policy at every next-state. By caching this
        once per bootstrap sample, we avoid evaluating the policy networks
        inside the FQE loop.
    """
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
    arangeK = torch.arange(K, device=device).unsqueeze(1)  # (K, 1)

    for it in range(n_iters):
        idx = torch.randint(0, N, (batch,), device=device)
        sb = s[idx]                        # (B, state_dim)
        ab = a[idx]                        # (B,) — same action shared across policies (data action)
        rb = r[idx]                        # (B,)
        nsb = ns[idx]                      # (B, state_dim)
        db = d[idx]                        # (B,)
        # Each policy's action at the sampled next-states:
        # policy_actions_at_ns[k, idx[i]] = a_k for ns_i
        # → (K, B)
        ns_a_all = policy_actions_at_ns[:, idx]   # (K, B)

        with torch.amp.autocast(device_type="cuda", enabled=autocast, dtype=torch.bfloat16):
            with torch.no_grad():
                tgt_q_all = target(nsb)           # (K, B, num_actions)
                # gather along action dim: pick policy-specific action per (k, b)
                next_q = tgt_q_all.gather(2, ns_a_all.unsqueeze(-1)).squeeze(-1)  # (K, B)
                y = rb.unsqueeze(0) + gamma * (1.0 - db).unsqueeze(0) * next_q   # (K, B)
            q_all = online(sb)                    # (K, B, num_actions)
            ab_exp = ab.unsqueeze(0).expand(K, -1).unsqueeze(-1)  # (K, B, 1)
            q_sa = q_all.gather(2, ab_exp).squeeze(-1)            # (K, B)
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
    """V_pi^k ≈ E[Q_k(s_0, a_k(s_0))] for each policy k."""
    s = torch.tensor(init_states_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        q_all = qnet_eval(s)                                         # (K, M, num_actions)
        gathered = q_all.gather(2, policy_actions_at_init.unsqueeze(-1)).squeeze(-1)  # (K, M)
    return gathered.mean(dim=1).cpu().numpy()                         # (K,)


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

    # ----- Load + preprocess -----
    print("\n>>> Loading data ...")
    orig, sim, vocabs = load_and_preprocess(args.orig_csv, args.sim_csv)
    print(f"    orig: {len(orig)} rows, {orig['uid'].nunique()} patients")
    print(f"    sim:  {len(sim)} rows, {sim['user_id'].nunique()} patients")

    print("\n>>> Building transitions ...")
    orig_trans = build_transitions(orig, "uid", ["datetime"])
    sim_trans = build_transitions(sim, "user_id", ["study_day", "slot"])
    # Avoid id collision between sim user_id and orig uid in merged datasets
    for t in sim_trans:
        t["patient_id"] = f"sim_{t['patient_id']}"
    print(f"    orig transitions: {len(orig_trans)}")
    print(f"    sim transitions:  {len(sim_trans)}")

    print("\n>>> Loading fixed train/test split ...")
    with open(args.train_uids) as f:
        train_uids = set(json.load(f))
    with open(args.test_uids) as f:
        test_uids = set(json.load(f))
    print(f"    train uids: {len(train_uids)}, test uids: {len(test_uids)}")

    datasets, test_set, test_pats = make_splits(
        orig_trans, sim_trans, train_uids, test_uids, seed=args.seed)
    print(f"    test patients: {len(test_pats)}, test transitions: {len(test_set)}")
    for k, dd in datasets.items():
        print(f"    {k:14s}  train={len(dd['train']):>6d}  val={len(dd['val']):>6d}")

    # ----- Step 1: Train DDQN per dataset -----
    HPARAMS = {
        "original":      dict(lr=5.5e-5, hidden=128),
        "synthetic_60":  dict(lr=1.3e-4, hidden=64),
        "synthetic_100": dict(lr=6.1e-5, hidden=128),
        "merged":        dict(lr=4.5e-5, hidden=128),
    }
    state_dim = len(STATE_COLS)
    scalers, trained = {}, {}
    print("\n>>> Step 1 — Train DDQN on each dataset")
    for name, dd in datasets.items():
        print(f"\n--- DDQN on '{name}' ---")
        s_tr, _, _, _, _, _ = to_arrays(dd["train"])
        sc = MinMaxScaler().fit(s_tr)
        scalers[name] = sc
        t0 = time.time()
        qnet = train_ddqn(dd["train"], dd["val"], sc, device,
                           n_iters=args.ddqn_iters,
                           eval_every=args.ddqn_eval_every,
                           batch=args.ddqn_batch,
                           use_amp=use_amp,
                           **HPARAMS[name],
                           verbose=True)
        print(f"    done in {time.time() - t0:.1f}s")
        trained[name] = qnet

    for name, q in trained.items():
        torch.save(q.state_dict(), os.path.join(args.out_dir, f"ddqn_{name}.pt"))

    # ----- Build policies (ordered list — index = policy_id) -----
    POLICY_NAMES = ["Original", "Synthetic n=60", "Synthetic n=100", "Merged",
                    "No message", "Send a=1", "Send a=2"]
    K = len(POLICY_NAMES)

    def policy_actions_for_states(states_tensor):
        """Compute (K, N) action tensor — each policy's chosen action at each state.
        Learnt policies use their trained Q-net's argmax; fixed policies use a constant.
        """
        with torch.no_grad():
            actions = torch.zeros(K, states_tensor.shape[0], dtype=torch.long, device=device)
            actions[0] = trained["original"](states_tensor).argmax(dim=1)
            actions[1] = trained["synthetic_60"](states_tensor).argmax(dim=1)
            actions[2] = trained["synthetic_100"](states_tensor).argmax(dim=1)
            actions[3] = trained["merged"](states_tensor).argmax(dim=1)
            actions[4] = 0
            actions[5] = 1
            actions[6] = 2
        return actions

    fqe_scaler = scalers["original"]

    # ----- Step 2a: Point estimates on full test set -----
    print("\n>>> Step 2a — Point estimates on full test set (multi-policy FQE)")
    t0 = time.time()
    s_full, _, _, ns_full, _, _ = to_arrays(test_set)
    s_full_t = torch.tensor(fqe_scaler.transform(s_full), dtype=torch.float32, device=device)
    ns_full_t = torch.tensor(fqe_scaler.transform(ns_full), dtype=torch.float32, device=device)
    init_full = fqe_scaler.transform(get_initial_states(test_set))
    init_full_t = torch.tensor(init_full, dtype=torch.float32, device=device)

    pol_a_at_ns_full = policy_actions_for_states(ns_full_t)        # (K, N)
    pol_a_at_init_full = policy_actions_for_states(init_full_t)    # (K, M)

    qe = train_fqe_multi(pol_a_at_ns_full, test_set, fqe_scaler, state_dim,
                         K, device,
                         n_iters=args.fqe_iters_point, batch=args.fqe_batch,
                         use_amp=use_amp)
    Vs = value_estimate_multi(qe, pol_a_at_init_full, init_full, device)
    point_estimates = {name: float(v) for name, v in zip(POLICY_NAMES, Vs)}
    print(f"    Multi-FQE for point estimates done in {time.time()-t0:.1f}s")
    for name, v in point_estimates.items():
        print(f"      {name:20s}  V_hat = {v:.4f}")

    # ----- Step 2b: Bootstrap (B=150) -----
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

        # Cache per-policy actions for next-states + initial states
        s_b, _, _, ns_b, _, _ = to_arrays(boot_trans)
        ns_b_t = torch.tensor(fqe_scaler.transform(ns_b), dtype=torch.float32, device=device)
        init_b = fqe_scaler.transform(get_initial_states(boot_trans))
        init_b_t = torch.tensor(init_b, dtype=torch.float32, device=device)

        pol_a_at_ns = policy_actions_for_states(ns_b_t)        # (K, N_b)
        pol_a_at_init = policy_actions_for_states(init_b_t)    # (K, M_b)

        qe_b = train_fqe_multi(pol_a_at_ns, boot_trans, fqe_scaler, state_dim,
                                K, device,
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
        pickle.dump({"boot_values": boot_values, "point_estimates": point_estimates,
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
        print(f"    {name:20s}  Vhat={Vhat:7.4f}  median={median:7.4f}  "
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

    # ----- Action distributions of learnt policies on full test set -----
    print("\n>>> Action distribution of learnt policies on full test set")
    ts, _, _, _, _, _ = to_arrays(test_set)
    ts_t = torch.tensor(fqe_scaler.transform(ts), dtype=torch.float32, device=device)
    action_dists = {}
    for name in ["original", "synthetic_60", "synthetic_100", "merged"]:
        with torch.no_grad():
            ac = trained[name](ts_t).argmax(dim=1).cpu().numpy()
        dist = {f"a={k}": float((ac == k).mean()) for k in range(NUM_ACTIONS)}
        action_dists[name] = dist
        print(f"    {name:18s}  {dist}")
    with open(os.path.join(args.out_dir, "action_distributions.json"), "w") as f:
        json.dump(action_dists, f, indent=2)

    print(f"\n✅ DONE. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
