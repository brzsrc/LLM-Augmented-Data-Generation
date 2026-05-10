"""
test_full_pipeline_vs_real_parallel.py
======================================
test_full_pipeline_vs_real.py 的批量并行版本。

跟 serial 版本相比的差异:
  - serial: 30 个用户挨个跑, 每行 1 个 LLM 调用
  - parallel: 跨用户同步推进, 同一 step 所有 active 用户的 LLM 调用合成一个大 batch
              用 llm.batch_adjustment() 一次提交 → vLLM 上能拿到 5-15× 加速

并行设计 (与 simulate_parallel 一致, 适配真实数据):
  - 每个用户 N_u 个真实行, 按 (date, day_slot) 排序成 personal timeline
  - 全局推进 step k = 0, 1, 2, ..., max_N_u
  - 在 step k, active users = [u for u in users if k < N_u]:
      * 每个抽 n_runs 次 base
      * 这批 active_users × n_runs 个 prompt 一起 batch_adjustment
      * 把结果填回各用户的第 k 行
      * 用真实 jbsteps30 构造 observation, 加入各用户 memory stream

跟 serial 共享:
  - prompt 模板 / 输出列名 / 报告结构 / W 指标算法

进度日志 (--progress-every-step):
  每 N step 打一行: active 用户数、batch size、累计 LLM 调用、当前累计 MAE/bias、ETA

输入/输出与 serial 完全一致, 可以直接对照两个版本的 csv/png 看是否对齐。

用法:
    python test_full_pipeline_vs_real_parallel.py \\
        --cleaned ./data/cleaned_output.csv \\
        --train   ./data/train_uids.json \\
        --out     ./fullpipe_parallel \\
        --runs    10
        [--llm qwen --qwen-path ../models/Qwen3-8B-AWQ]
        [--with-reflection]
        [--max-users N]
        [--progress-every-step 25]
"""
import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'DejaVu Sans'


# =============================================================================
# stub data.cluster_config (跟 serial 一致)
# =============================================================================
def stub_cluster_config(extractor_dir: str):
    if 'data' in sys.modules:
        return
    try:
        sys.path.insert(0, extractor_dir)
        import cluster_config as cc
        mod = types.ModuleType('data.cluster_config')
        for k in ['CLUSTER_UIDS', 'CLUSTER_WEIGHTS', 'CLUSTER_NAMES',
                  'CLUSTER_LOCATION_DIST', 'CLUSTER_AVAIL_BY_SLOT']:
            setattr(mod, k, getattr(cc, k))
        data_mod = types.ModuleType('data'); data_mod.cluster_config = mod
        sys.modules['data'] = data_mod
        sys.modules['data.cluster_config'] = mod
        print('[stub] using real cluster_config.py')
    except Exception:
        mod = types.ModuleType('data.cluster_config')
        mod.CLUSTER_UIDS = {0: []}; mod.CLUSTER_WEIGHTS = {0: 1.0}
        mod.CLUSTER_NAMES = {0: 'all'}; mod.CLUSTER_LOCATION_DIST = {}
        mod.CLUSTER_AVAIL_BY_SLOT = {}
        data_mod = types.ModuleType('data'); data_mod.cluster_config = mod
        sys.modules['data'] = data_mod
        sys.modules['data.cluster_config'] = mod
        print('[stub] cluster_config stubbed empty')


def maybe_make_dummy_users_csv(cleaned_path: str, out_path: str):
    if os.path.exists(out_path):
        return out_path
    cleaned = pd.read_csv(cleaned_path)
    uids = sorted(cleaned['uid'].unique().tolist())
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        'user.index': uids,
        'selfeff.intake': rng.uniform(8, 22, len(uids)),
        'consc':          rng.uniform(15, 28, len(uids)),
        'age':            rng.randint(20, 60, len(uids)),
        'gender':         rng.choice(['M', 'F'], len(uids)),
    })
    df.to_csv(out_path, index=False)
    print(f'[dummy users] wrote {out_path}')
    return out_path


def prepare_real_rows(cleaned_path: str, train_uids: list,
                      avail_only: bool = True) -> pd.DataFrame:
    """跟 serial 版的 prepare_real_rows 完全一致。"""
    df = pd.read_csv(cleaned_path)
    df = df[df['uid'].isin(train_uids)].copy()
    if avail_only:
        df = df[df['avail'] == True].copy()
    df = df.dropna(subset=['jbsteps30', 'jbsteps30pre', 'location', 'activity'])

    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.dayofweek < 5
    df = df.sort_values(['uid', 'date', 'day_slot']).reset_index(drop=True)

    dosage_list = []
    for uid, ud in df.groupby('uid', sort=False):
        d = 0.0
        for s in ud['send'].values:
            d = 0.95 * d + (1.0 if int(s) > 0 else 0.0)
            dosage_list.append(d)
    df['dosage'] = dosage_list

    user_stats = []
    for uid, ud in df.groupby('uid'):
        m_all = ud['jbsteps30'].mean()
        m_send = ud[ud['send'] > 0]['jbsteps30'].mean()
        m_no   = ud[ud['send'] == 0]['jbsteps30'].mean()
        te = (m_send - m_no) if (not np.isnan(m_send) and not np.isnan(m_no)) else 0.0
        user_stats.append({'uid': uid, 'predicted_mean_steps': m_all,
                           'predicted_te': te})
    user_stats = pd.DataFrame(user_stats)
    df = df.merge(user_stats, on='uid', how='left')

    return df


def build_user_params(uid: int, real_mean: float, real_te: float,
                      users_df: pd.DataFrame) -> dict:
    """跟 serial 版一致。"""
    row = users_df[users_df['user.index'] == uid]
    if len(row) > 0:
        r = row.iloc[0]
        return {
            'age': r['age'], 'gender': r['gender'],
            'selfeff.intake': r['selfeff.intake'], 'consc': r['consc'],
            'predicted_mean_steps': real_mean, 'predicted_te': real_te,
        }
    return {
        'age': 35, 'gender': 'F', 'selfeff.intake': 14.5, 'consc': 22.0,
        'predicted_mean_steps': real_mean, 'predicted_te': real_te,
    }


def make_persona(params: dict) -> str:
    """跟 serial 版一致。"""
    return (f"Age {int(params['age'])}, {params['gender']}. "
            f"Self-efficacy {params['selfeff.intake']:.1f}/25, "
            f"conscientiousness {params['consc']:.1f}/30. "
            f"Baseline ~{int(params['predicted_mean_steps'])} steps/decision. "
            f"Initial treatment effect {params['predicted_te']:+.0f} steps.")


# =============================================================================
# 用户运行时状态: 一个用户一个 stream + cursor
# =============================================================================
class UserRuntime:
    def __init__(self, uid, params, persona, all_rows, MS_module):
        self.uid = uid
        self.params = params
        self.persona = persona
        # 重要: reset_index 让 .iloc[k] 工作; 排序确保 personal time order
        self.all_rows = all_rows.sort_values(['date', 'day_slot']).reset_index(drop=True)
        self.N = len(self.all_rows)

        self.stream = MS_module.MemoryStream()
        for line in persona.strip().split('. '):
            if line.strip():
                self.stream.add(MS_module.Memory(
                    "study_start", line.strip() + ".", "background", 7))

        # 输出缓冲: 一行一个 dict, 与 all_rows 同顺序; 未处理保持 None
        self.results = [None] * self.N


# =============================================================================
# 核心: 按 step k 推进, 每 step 内 batch active users × n_runs 个 prompt
# =============================================================================
def run_parallel_pipeline(real_df, ext, llm, n_runs, with_reflection,
                          DE, MS_module, prompts_module,
                          progress_every_step: int = 25,
                          verbose: bool = True):
    """跑完所有用户, 返回 runtimes 字典。"""
    Memory = MS_module.Memory
    create_observation = MS_module.create_observation
    PROMPT_ADJUSTMENT = prompts_module.PROMPT_ADJUSTMENT
    SYS_ADJUSTMENT = prompts_module.SYS_ADJUSTMENT

    if with_reflection:
        REFLECTION_Q_PROMPT = prompts_module.REFLECTION_Q_PROMPT
        REFLECTION_GEN_PROMPT = prompts_module.REFLECTION_GEN_PROMPT
        IMPORTANCE_SYSTEM = prompts_module.IMPORTANCE_SYSTEM
        IMPORTANCE_USER = prompts_module.IMPORTANCE_USER

    # ── Step 1: 给每个用户初始化 runtime ────────────────────────────
    runtimes = {}
    for uid, ud in real_df.groupby('uid'):
        params = {
            'predicted_mean_steps': float(ud['predicted_mean_steps'].iloc[0]),
            'predicted_te': float(ud['predicted_te'].iloc[0]),
            'age': int(ud['age'].iloc[0])
                if 'age' in ud.columns and not pd.isna(ud['age'].iloc[0]) else 35,
            'gender': ud['gender'].iloc[0]
                if 'gender' in ud.columns and isinstance(ud['gender'].iloc[0], str) else 'F',
            'selfeff.intake': float(ud['selfeff.intake'].iloc[0])
                if 'selfeff.intake' in ud.columns and not pd.isna(ud['selfeff.intake'].iloc[0]) else 14.5,
            'consc': float(ud['consc'].iloc[0])
                if 'consc' in ud.columns and not pd.isna(ud['consc'].iloc[0]) else 22.0,
        }
        persona = make_persona(params)
        runtimes[uid] = UserRuntime(uid, params, persona, ud, MS_module)

    sorted_uids = sorted(runtimes.keys())
    max_steps = max(rt.N for rt in runtimes.values())
    total_rows = sum(rt.N for rt in runtimes.values())
    print(f'[parallel] {len(runtimes)} users, max_steps={max_steps}, '
          f'total_rows={total_rows}')

    # ── Step 2: 按 k 推进 ───────────────────────────────────────────
    t0 = time.time()
    rows_done = 0
    # rolling 累计: 用于进度日志的 MAE/bias
    accum_abs_err = 0.0
    accum_signed_err = 0.0

    for k in range(max_steps):
        active_uids = [uid for uid in sorted_uids if k < runtimes[uid].N]
        if not active_uids:
            break

        t_step_start = time.time()
        llm_calls_at_step_start = llm.call_count

        # ── Phase A: 每个 active 用户抽 n_runs 次 base + 构造 prompts ─
        base_buffers = {}    # uid -> list of n_runs base values
        all_prompts = []     # 全部 prompt
        prompt_meta = []     # 同长 (uid, run_idx)
        active_ctx = {}      # uid -> (ctx, action, dosage, row, recent_obs)

        for uid in active_uids:
            rt = runtimes[uid]
            row = rt.all_rows.iloc[k]

            weather = row['weather'] \
                if 'weather' in rt.all_rows.columns and isinstance(row['weather'], str) \
                else 'Clear'
            temp = float(row['temperature']) if not pd.isna(row['temperature']) else 22.0
            ctx = {
                'slot': int(row['day_slot']),
                'location': row['location'],
                'weekday': bool(row['weekday']),
                'activity': row['activity'] if isinstance(row['activity'], str) else 'STILL',
                'weather': weather,
                'temperature': temp,
                'study_day': 1,
                'prior_30min_steps': int(row['jbsteps30pre']),
                'response': row['response']
                    if 'response' in rt.all_rows.columns and isinstance(row['response'], str)
                    else 'no_send',
            }
            action = int(row['send'])
            dosage = float(row['dosage'])

            bases = [DE.generate_base_steps(rt.params, ctx, action, dosage, ext)
                     for _ in range(n_runs)]
            base_buffers[uid] = bases

            obs_text = rt.stream.format_memories(rt.stream.get_recent(10))
            recent_obs = obs_text if obs_text else "No prior observations."
            active_ctx[uid] = (ctx, action, dosage, row, recent_obs)

            action_desc = {0: "No suggestion", 1: "Active walking suggestion",
                           2: "Sedentary stand-up suggestion"}.get(action, "No suggestion")
            weekday_desc = "weekday" if ctx['weekday'] else "weekend"

            for run_idx, base_val in enumerate(bases):
                if base_val > 0:
                    p = PROMPT_ADJUSTMENT.format(
                        persona=rt.persona,
                        study_day=int(row['day_slot']),
                        slot=int(row['day_slot']),
                        weekday_desc=weekday_desc,
                        location=ctx['location'], activity=ctx['activity'],
                        weather=ctx['weather'], temperature=ctx['temperature'],
                        prior_30min_steps=ctx['prior_30min_steps'],
                        action_desc=action_desc, base_steps=base_val,
                        recent_obs=recent_obs)
                    all_prompts.append({"system": SYS_ADJUSTMENT, "user": p})
                    prompt_meta.append((uid, run_idx))
                # else: base==0 → adj 直接 0, 不发 LLM

        # ── Phase B: 一次大 batch ─────────────────────────────────
        adj_results = llm.batch_adjustment(all_prompts) if all_prompts else []

        # ── Phase C: 把 adj 填回 + 写 results ─────────────────────
        adj_buffers = {uid: [0] * n_runs for uid in active_uids}
        for k_p, (uid, run_idx) in enumerate(prompt_meta):
            adj = max(-50, min(100, adj_results[k_p]))
            adj_buffers[uid][run_idx] = adj

        for uid in active_uids:
            rt = runtimes[uid]
            ctx, action, _, row, _ = active_ctx[uid]
            bases = np.array(base_buffers[uid])
            adjs  = np.array(adj_buffers[uid])
            steps_arr = np.maximum(0, (bases * (1 + adjs / 100.0)).astype(int))
            rt.results[k] = {
                'base_mean':  float(bases.mean()),
                'adj_mean':   float(adjs.mean()),
                'steps_mean': float(steps_arr.mean()),
                'steps_one':  int(steps_arr[0]),
            }
            # rolling MAE/bias
            err = float(steps_arr.mean()) - float(row['jbsteps30'])
            accum_abs_err += abs(err)
            accum_signed_err += err

        # ── Phase D: importance + 加入 memory stream ─────────────
        if with_reflection:
            imp_prompts = []
            imp_meta = []
            for uid in active_uids:
                rt = runtimes[uid]
                ctx, action, _, row, _ = active_ctx[uid]
                obs = create_observation(
                    day=1, slot=int(row['day_slot']),
                    ctx=ctx, action=action, steps=int(row['jbsteps30']))
                imp_prompts.append({
                    "system": IMPORTANCE_SYSTEM,
                    "user": IMPORTANCE_USER.format(observation=obs)
                })
                imp_meta.append((uid, obs, int(row['day_slot'])))
            imp_results = llm.batch_importance(imp_prompts) if imp_prompts else []
            for k_imp, (uid, obs, slot_n) in enumerate(imp_meta):
                try:
                    importance = max(1, min(10, int(imp_results[k_imp])))
                except Exception:
                    importance = 5
                runtimes[uid].stream.add(Memory(
                    f"step{k}_S{slot_n}", obs, "observation", importance))
        else:
            for uid in active_uids:
                rt = runtimes[uid]
                ctx, action, _, row, _ = active_ctx[uid]
                obs = create_observation(
                    day=1, slot=int(row['day_slot']),
                    ctx=ctx, action=action, steps=int(row['jbsteps30']))
                rt.stream.add(Memory(
                    f"step{k}_S{int(row['day_slot'])}", obs, "observation", 5))

        # ── Phase E: 反思 (可选) ─────────────────────────────────
        if with_reflection:
            reflect_uids = [uid for uid in active_uids
                            if runtimes[uid].stream.should_reflect()]
            if reflect_uids:
                q_prompts = []
                q_recent = {}
                for uid in reflect_uids:
                    recent = runtimes[uid].stream.get_recent(20)
                    recent_text = runtimes[uid].stream.format_memories(recent)
                    q_recent[uid] = recent_text
                    q_prompts.append({
                        "system": "You are a behavioral analysis system.",
                        "user": REFLECTION_Q_PROMPT.format(recent_memories=recent_text)
                    })
                q_responses = llm.batch_text(q_prompts)

                inf_prompts = []
                inf_meta = []
                for j, uid in enumerate(reflect_uids):
                    questions = [q.strip().lstrip('-*0123456789. )')
                                 for q in q_responses[j].split('\n')
                                 if q.strip()][:3]
                    for q in questions:
                        inf_prompts.append({
                            "system": "You are a behavioral analysis system. Write concise inferences.",
                            "user": REFLECTION_GEN_PROMPT.format(
                                question=q, relevant_memories=q_recent[uid])
                        })
                        inf_meta.append((uid, q))

                inf_responses = llm.batch_text(inf_prompts) if inf_prompts else []
                for k_inf, (uid, q) in enumerate(inf_meta):
                    runtimes[uid].stream.add(Memory(
                        f"step{k}_ref",
                        inf_responses[k_inf] if k_inf < len(inf_responses) else "",
                        "reflection", 8))
                for uid in reflect_uids:
                    runtimes[uid].stream.reset_reflection_counter()

        rows_done += len(active_uids)
        step_elapsed_ms = (time.time() - t_step_start) * 1000
        llm_calls_this_step = llm.call_count - llm_calls_at_step_start

        # ── 进度日志 ────────────────────────────────────────────
        if verbose and progress_every_step and (k + 1) % progress_every_step == 0:
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (max_steps - k - 1)
            mae_so_far  = accum_abs_err / rows_done
            bias_so_far = accum_signed_err / rows_done
            print(f'  [step {k+1}/{max_steps}] '
                  f'active={len(active_uids):>2} users, '
                  f'batch={len(all_prompts):>3} prompts '
                  f'({llm_calls_this_step} LLM calls in {step_elapsed_ms:.0f}ms), '
                  f'rows_done={rows_done}/{total_rows} '
                  f'(MAE={mae_so_far:.1f} bias={bias_so_far:+.1f}), '
                  f'LLM_total={llm.call_count}, '
                  f'elapsed={elapsed:.0f}s, ETA={eta:.0f}s')

    print(f'[parallel] done: {rows_done}/{total_rows} rows in '
          f'{time.time()-t0:.0f}s, total LLM calls={llm.call_count}')
    return runtimes


def assemble_output(real_df: pd.DataFrame, runtimes: dict) -> pd.DataFrame:
    """把 runtimes.results 拼回 real_df, 加 base_mean/adj_mean/steps_mean/steps_one。"""
    real_df = real_df.copy()
    real_df = real_df.sort_values(['uid', 'date', 'day_slot']).reset_index(drop=True)

    base_col = np.zeros(len(real_df), dtype=float)
    adj_col  = np.zeros(len(real_df), dtype=float)
    sm_col   = np.zeros(len(real_df), dtype=float)
    so_col   = np.zeros(len(real_df), dtype=int)

    cursor = 0
    for uid, ud in real_df.groupby('uid', sort=False):
        rt = runtimes[uid]
        if len(ud) != rt.N:
            print(f'[warn] uid={uid}: real_df slice n={len(ud)} != runtime.N={rt.N}')
        for k_local in range(len(ud)):
            r = rt.results[k_local] if k_local < rt.N and rt.results[k_local] is not None \
                else {'base_mean': 0.0, 'adj_mean': 0.0, 'steps_mean': 0.0, 'steps_one': 0}
            i = cursor + k_local
            base_col[i] = r['base_mean']
            adj_col[i]  = r['adj_mean']
            sm_col[i]   = r['steps_mean']
            so_col[i]   = r['steps_one']
        cursor += len(ud)

    real_df['base_mean']  = base_col
    real_df['adj_mean']   = adj_col
    real_df['steps_mean'] = sm_col
    real_df['steps_one']  = so_col
    real_df['err_mean']   = real_df['steps_mean'] - real_df['jbsteps30']
    real_df['abs_err']    = real_df['err_mean'].abs()
    return real_df


# =============================================================================
# 报告 + 绘图 (跟 serial 完全一致)
# =============================================================================
def report(out_df, save_dir: Path):
    real = out_df['jbsteps30']
    pm = out_df['steps_mean']
    one = out_df['steps_one']

    print('\n' + '=' * 72)
    print('总体: pipeline steps_mean (n_runs 次平均) vs real jbsteps30')
    print('=' * 72)
    print(f"  n_rows                       : {len(out_df)}")
    print(f"  real     jbsteps30      mean : {real.mean():>7.1f},  "
          f"median : {real.median():>5.0f},  std : {real.std():>6.1f}")
    print(f"  pipeline steps_mean     mean : {pm.mean():>7.1f},  "
          f"median : {pm.median():>5.0f},  std : {pm.std():>6.1f}")
    print()
    print(f"  MAE                          : {out_df['abs_err'].mean():>7.1f}")
    print(f"  bias  (mean signed err)      : {out_df['err_mean'].mean():>+7.1f}")
    print(f"  RMSE                         : "
          f"{np.sqrt((out_df['err_mean']**2).mean()):>7.1f}")
    print(f"  median |error|               : {out_df['abs_err'].median():>7.1f}")
    r2 = 1 - ((out_df['err_mean']**2).sum() / ((real-real.mean())**2).sum())
    print(f"  R² of steps_mean ~ jbsteps30 : {r2:.4f}")
    print(f"  Pearson  corr                : "
          f"{out_df[['steps_mean','jbsteps30']].corr().iloc[0,1]:+.3f}")
    print(f"  Spearman corr                : "
          f"{out_df[['steps_mean','jbsteps30']].corr(method='spearman').iloc[0,1]:+.3f}")
    print(f"  Wasserstein(real, steps_one) = "
          f"{wasserstein_distance(real, one):.1f}")
    print(f"  Wasserstein / mean(real)     = "
          f"{wasserstein_distance(real, one)/real.mean()*100:.1f}%")

    adj = out_df['adj_mean']
    print()
    print(f"  LLM adj_mean             mean: {adj.mean():>+7.1f},  "
          f"std: {adj.std():>6.1f},  range: [{adj.min():.0f}, {adj.max():.0f}]")
    base = out_df['base_mean']
    print(f"  base_mean (no LLM)       mean: {base.mean():>7.1f}")
    print(f"  steps_mean (with LLM)    mean: {pm.mean():>7.1f}  "
          f"(LLM 净影响: {pm.mean()-base.mean():+.1f} 步)")

    print(); print('=' * 72); print('按 send 分组'); print('=' * 72)
    print(f"  {'send':>5} | {'n':>5} | {'real':>7} | {'base':>7} | "
          f"{'steps':>7} | {'bias':>7} | {'MAE':>6}")
    print('  ' + '-' * 65)
    for s in sorted(out_df['send'].unique()):
        sub = out_df[out_df['send'] == s]
        print(f"  {int(s):>5} | {len(sub):>5} | {sub['jbsteps30'].mean():>7.1f} | "
              f"{sub['base_mean'].mean():>7.1f} | "
              f"{sub['steps_mean'].mean():>7.1f} | "
              f"{(sub['steps_mean']-sub['jbsteps30']).mean():>+7.1f} | "
              f"{sub['abs_err'].mean():>6.1f}")

    print(); print('=' * 72); print('按 day_slot 分组'); print('=' * 72)
    print(f"  {'slot':>5} | {'n':>5} | {'real':>7} | {'base':>7} | "
          f"{'steps':>7} | {'bias':>7} | {'MAE':>6}")
    print('  ' + '-' * 65)
    for s in sorted(out_df['day_slot'].unique()):
        sub = out_df[out_df['day_slot'] == s]
        print(f"  {int(s):>5} | {len(sub):>5} | {sub['jbsteps30'].mean():>7.1f} | "
              f"{sub['base_mean'].mean():>7.1f} | "
              f"{sub['steps_mean'].mean():>7.1f} | "
              f"{(sub['steps_mean']-sub['jbsteps30']).mean():>+7.1f} | "
              f"{sub['abs_err'].mean():>6.1f}")

    print(); print('=' * 72); print('按 location top-8 分组'); print('=' * 72)
    print(f"  {'location':<40} | {'n':>5} | {'real':>7} | "
          f"{'steps':>7} | {'bias':>7}")
    print('  ' + '-' * 80)
    top_locs = out_df['location'].value_counts().head(8).index.tolist()
    for loc in top_locs:
        sub = out_df[out_df['location'] == loc]
        print(f"  {loc[:40]:<40} | {len(sub):>5} | "
              f"{sub['jbsteps30'].mean():>7.1f} | "
              f"{sub['steps_mean'].mean():>7.1f} | "
              f"{(sub['steps_mean']-sub['jbsteps30']).mean():>+7.1f}")

    print(); print('=' * 72)
    print('桶级 Wasserstein: W(real_jbsteps30, steps_one)')
    print('=' * 72)
    rows = []
    for (slot, send, loc), grp in out_df.groupby(['day_slot', 'send', 'location']):
        if len(grp) >= 10:
            w = wasserstein_distance(grp['jbsteps30'].values,
                                     grp['steps_one'].values)
            rows.append({'slot': slot, 'send': send, 'loc': loc,
                         'n': len(grp), 'mean_real': grp['jbsteps30'].mean(),
                         'W': w})
    bw = pd.DataFrame(rows)
    if len(bw) > 0:
        avg_W = bw['W'].mean(); avg_mean = bw['mean_real'].mean()
        print(f"  桶定义: (slot, send, location), 门槛 n>=10")
        print(f"  桶数: {len(bw)},  平均 W: {avg_W:.1f}, "
              f"中位 W: {bw['W'].median():.1f},  最差 W: {bw['W'].max():.1f}")
        print(f"  桶级真实均值 (各桶 real 均值的均值): {avg_mean:.1f}")
        print(f"  W / 桶均值 = {avg_W / avg_mean * 100:.1f}%")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.scatter(real, pm, alpha=0.15, s=8, color='crimson', edgecolors='none')
    lim = min(2500, max(real.quantile(.99), pm.quantile(.99)) * 1.05)
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label='y = x')
    ax.set_xlabel('real jbsteps30'); ax.set_ylabel('pipeline steps_mean')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(f'(a) Pipeline (real prior + base + LLM adj) vs real\n'
                 f'MAE={out_df["abs_err"].mean():.0f}, '
                 f'bias={out_df["err_mean"].mean():+.0f}, R²={r2:.3f}')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.hist(out_df['err_mean'].clip(-1500, 1500), bins=80,
            color='salmon', edgecolor='black', linewidth=0.4)
    ax.axvline(0, color='black', lw=1)
    ax.axvline(out_df['err_mean'].mean(), color='red', lw=2,
               label=f"mean = {out_df['err_mean'].mean():+.0f}")
    ax.set_xlabel('error = steps_mean − real jbsteps30'); ax.set_ylabel('count')
    ax.set_title('(b) Signed error distribution (clipped to ±1500)')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    bins = np.linspace(0, 1500, 50)
    ax.hist(real, bins=bins, alpha=0.55, density=True,
            color='steelblue', edgecolor='black', linewidth=0.4,
            label=f'real jbsteps30 (mean={real.mean():.0f})')
    ax.hist(out_df['base_mean'], bins=bins, alpha=0.45, density=True,
            color='lightgreen', edgecolor='black', linewidth=0.4,
            label=f'base_mean (no LLM, mean={out_df["base_mean"].mean():.0f})')
    ax.hist(one, bins=bins, alpha=0.55, density=True,
            color='salmon', edgecolor='black', linewidth=0.4,
            label=f'pipeline steps_one (mean={one.mean():.0f})')
    ax.set_xlabel('steps'); ax.set_ylabel('density')
    ax.set_title(f'(c) Distribution overlay (truncated 1500)\n'
                 f'W(real, steps_one)={wasserstein_distance(real, one):.1f}')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    grp = out_df.groupby(['day_slot', 'send']).agg(
        real=('jbsteps30', 'mean'), base=('base_mean', 'mean'),
        steps=('steps_mean', 'mean'), n=('jbsteps30', 'size')
    ).reset_index()
    grp = grp[grp['n'] >= 10]
    x = np.arange(len(grp)); w = 0.27
    ax.bar(x - w, grp['real'], w, color='steelblue', edgecolor='black', label='real')
    ax.bar(x, grp['base'], w, color='lightgreen', edgecolor='black', label='base')
    ax.bar(x + w, grp['steps'], w, color='salmon', edgecolor='black', label='pipeline')
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{int(r.day_slot)}/k{int(r.send)}" for r in grp.itertuples()],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('mean steps')
    ax.set_title('(d) Per-(slot, send): real / base / pipeline')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Plan A pipeline (PARALLEL, real prior fed in)  vs  real jbsteps30 — train set',
                 fontsize=12, fontweight='bold', y=1.0)
    plt.tight_layout()
    fig_path = save_dir / 'full_pipeline_vs_real.png'
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'\n  图保存: {fig_path}')

    csv_path = save_dir / 'full_pipeline_vs_real_rows.csv'
    keep = ['uid', 'date', 'day_slot', 'location', 'activity', 'weekday',
            'send', 'dosage', 'jbsteps30pre', 'jbsteps30',
            'base_mean', 'adj_mean', 'steps_mean', 'steps_one',
            'err_mean', 'abs_err']
    out_df[keep].to_csv(csv_path, index=False)
    print(f'  逐行结果保存: {csv_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cleaned',   default='./data/cleaned_output.csv')
    p.add_argument('--train',     default='./data/train_uids.json')
    p.add_argument('--test_uids', default=None)
    p.add_argument('--users_csv', default=None)
    p.add_argument('--data_extractor_dir', default='.')
    p.add_argument('--out',       default='./full_pipeline_parallel_output')
    p.add_argument('--runs',      type=int, default=10,
                   help='每行抽样次数, 平均掉 base + LLM 的随机性')
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--with-reflection', action='store_true',
                   help='打开后做 importance scoring + reflection (慢)')
    p.add_argument('--max-users', type=int, default=None,
                   help='只跑前 N 个 train user (smoke test 用)')
    p.add_argument('--llm', choices=['simulated', 'qwen'], default='qwen')
    p.add_argument('--qwen-path', default='../models/Qwen3-8B-AWQ')
    p.add_argument('--progress-every-step', type=int, default=25,
                   help='每 N step 打一次进度 (active users / batch / ETA / rolling MAE). 0=关掉')
    args = p.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True, parents=True)

    with open(args.train) as f:
        train_uids = json.load(f)
    print(f'[load] {len(train_uids)} train users')
    if args.max_users:
        train_uids = train_uids[:args.max_users]
        print(f'[load] limited to first {len(train_uids)} users for smoke test')

    if args.test_uids:
        with open(args.test_uids) as f:
            test_uids = json.load(f)
    else:
        all_uids = set(pd.read_csv(args.cleaned)['uid'].unique().tolist())
        test_uids = sorted(list(all_uids - set(train_uids)))
        tmp_test = out_dir / 'test_uids_auto.json'
        with open(tmp_test, 'w') as f:
            json.dump(test_uids, f)
        args.test_uids = str(tmp_test)
        print(f'[auto] inferred {len(test_uids)} test uids')

    stub_cluster_config(args.data_extractor_dir)

    if args.users_csv is None:
        users_csv = str(out_dir / 'users_dummy.csv')
        maybe_make_dummy_users_csv(args.cleaned, users_csv)
    else:
        users_csv = args.users_csv
    users_df = pd.read_csv(users_csv)

    sys.path.insert(0, args.data_extractor_dir)
    import data_extractor as DE
    import memory_stream as MS
    import prompts as PR
    import llm as LM

    ext = DE.V1DataExtractor(users_csv, args.cleaned,
                             train_uids_path=args.train,
                             test_uids_path=args.test_uids)

    if args.llm == 'qwen':
        print(f'[llm] loading Qwen3BLLM from {args.qwen_path}')
        llm = LM.Qwen3BLLM(model_path=args.qwen_path)
    else:
        print('[llm] using SimulatedLLM (random)')
        llm = LM.SimulatedLLM()

    real_df = prepare_real_rows(args.cleaned, train_uids, avail_only=True)
    users_df_renamed = users_df.rename(columns={'user.index': 'uid'})
    real_df = real_df.merge(
        users_df_renamed[['uid', 'age', 'gender', 'selfeff.intake', 'consc']],
        on='uid', how='left')
    print(f'[real_rows] {len(real_df)} rows')

    t0 = time.time()
    runtimes = run_parallel_pipeline(
        real_df, ext, llm, n_runs=args.runs,
        with_reflection=args.with_reflection,
        DE=DE, MS_module=MS, prompts_module=PR,
        progress_every_step=args.progress_every_step, verbose=True)
    out_df = assemble_output(real_df, runtimes)
    print(f'\n[done] total rows={len(out_df)}, '
          f'total LLM calls={llm.call_count}, '
          f'time={time.time()-t0:.0f}s')

    report(out_df, out_dir)


if __name__ == '__main__':
    main()
