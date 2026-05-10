"""
test_full_pipeline_vs_real.py
=============================
端到端评估 Plan A 的合成步数 vs 真实 jbsteps30:
    base ← generate_base_steps(user_params, ctx, action, dosage, ext)
        ← 真实 jbsteps30pre 直接塞进 ctx['prior_30min_steps'] (摘掉 prior 那一环误差)
    adj  ← LLM(simplified Plan A prompt: persona / ctx / recent_obs / base / ...)
    steps = max(0, int(base × (1 + adj/100)))

对比目标: 真实 jbsteps30 (当前 30 分钟步数)。

跟之前两个测试的差别:
  - test_baseline_vs_real.py     : 只测 generate_base_steps, 没 LLM 调节
  - test_prior_vs_real.py        : 只测 prior 那一环, 不进 baseline 不进 LLM
  - 这个                         : 摘掉 prior 误差, 测 baseline + LLM 整条信号链

关键设计:
  对每个 train user 按 (date, day_slot) 排序逐行走, 维护:
    - memory_stream (用真实 ctx + 真实 jbsteps30 构造 observation, 不是合成步数!)
    - dosage (累计 0.95 衰减真实 send)
  对每行, 重复 N_RUNS 次 (base 和 adj 都是随机的) 取均值得 final_mean。
  
持久 memory stream 设计:
  默认情况下, 每个用户的 memory stream 整段对所有 N_RUNS 共享 (因为 memory
  本来就是用真实数据构造的, 跨 run 不变)。LLM 调节本身的随机性靠 N_RUNS 抽样
  捕捉, base 的随机性也靠 N_RUNS。

是否触发反思 (--with-reflection):
  默认 off, 只调 LLM 一次/行 (adj prompt), 快。
  on: 加上 importance scoring + reflection 触发, 完整复刻 simulate_user 的
  memory 行为, 但慢且对 SimulatedLLM 来说反思内容是噪声。

输入:
  - cleaned_output.csv (真实数据)
  - train_uids.json
  - data_extractor.py + cluster_config.py (Plan A 的版本)
  - prompts.py (Plan A 简化版)
  - llm.py
  - synthetic_user_generator.py (没在测试里直接用, 只是为了从中借 persona 构造)

输出:
  - full_pipeline_vs_real.png 四宫图
  - full_pipeline_vs_real_rows.csv 逐行结果
  - 控制台报告: 总体 / 按 slot / send / location 分组 + 桶级 W

用法:
    python test_full_pipeline_vs_real.py \
        --cleaned data/cleaned_output.csv \
        --train   data/train_uids.json \
        --out     ./full_pipeline_test_output \
        --runs    10
        [--with-reflection]
        [--max-users N]   # 只跑前 N 个用户便于快速 smoke test
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
# stub data.cluster_config (同前)
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


# =============================================================================
# 准备 train rows: 加 weekday, dosage, 用户级真实 mean / TE
# 跟 test_baseline_vs_real.py 一致, 但不再筛 avail (因为 LLM 也对 unavail 行不调用)
# =============================================================================
def prepare_real_rows(cleaned_path: str, train_uids: list,
                      avail_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(cleaned_path)
    df = df[df['uid'].isin(train_uids)].copy()
    if avail_only:
        df = df[df['avail'] == True].copy()
    df = df.dropna(subset=['jbsteps30', 'jbsteps30pre', 'location', 'activity'])

    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.dayofweek < 5

    # 严格按时间排序, 才能让累积 dosage 和 memory stream 有意义
    df = df.sort_values(['uid', 'date', 'day_slot']).reset_index(drop=True)

    # 累计 dosage (0.95 衰减), per-user 重置
    dosage_list = []
    for uid, ud in df.groupby('uid', sort=False):
        d = 0.0
        for s in ud['send'].values:
            d = 0.95 * d + (1.0 if int(s) > 0 else 0.0)
            dosage_list.append(d)
    df['dosage'] = dosage_list

    # 用户级真实 mean_steps / TE (用于 generate_base_steps 的 user_offset)
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
    """从 users.csv 拿 age/gender/selfeff/consc 构造 params。
    persona 字段一致, 但 mean_steps / TE 用真实值。"""
    row = users_df[users_df['user.index'] == uid]
    if len(row) > 0:
        r = row.iloc[0]
        params = {
            'age': r['age'], 'gender': r['gender'],
            'selfeff.intake': r['selfeff.intake'], 'consc': r['consc'],
            'predicted_mean_steps': real_mean,
            'predicted_te': real_te,
        }
    else:
        # Fallback if user not in users.csv
        params = {
            'age': 35, 'gender': 'F',
            'selfeff.intake': 14.5, 'consc': 22.0,
            'predicted_mean_steps': real_mean,
            'predicted_te': real_te,
        }
    return params


def make_persona(params: dict) -> str:
    return (f"Age {int(params['age'])}, {params['gender']}. "
            f"Self-efficacy {params['selfeff.intake']:.1f}/25, "
            f"conscientiousness {params['consc']:.1f}/30. "
            f"Baseline ~{int(params['predicted_mean_steps'])} steps/decision. "
            f"Initial treatment effect {params['predicted_te']:+.0f} steps.")


# =============================================================================
# 单用户走一遍: 维护 memory stream + dosage, 对每行返回 (base, adj, steps) 抽样
# =============================================================================
def evaluate_one_user(user_rows: pd.DataFrame, params: dict, ext, llm,
                      n_runs: int, with_reflection: bool,
                      DE, MS_module, prompts_module,
                      progress_every: int = 50, uid_label: str = ''):
    """
    返回 DataFrame: 与 user_rows 同长, 加列 base_mean / adj_mean / steps_mean / steps_one。

    progress_every: 每跑完这么多行, 打一行进度 (rolling MAE/bias/per-row latency)。
                    设为 0 关掉。
    uid_label: 进度行的前缀, 例如 'uid=12'.
    """
    Memory = MS_module.Memory
    MemoryStream = MS_module.MemoryStream
    create_observation = MS_module.create_observation
    PROMPT_ADJUSTMENT = prompts_module.PROMPT_ADJUSTMENT
    SYS_ADJUSTMENT = prompts_module.SYS_ADJUSTMENT

    persona = make_persona(params)
    stream = MemoryStream()
    # seed: persona 拆成几条 background memory (跟 simulate_user 一致)
    for line in persona.strip().split('. '):
        if line.strip():
            stream.add(Memory("study_start", line.strip() + ".", "background", 7))

    # 准备 reflection 相关 prompt (只在 with_reflection 时用到)
    if with_reflection:
        REFLECTION_Q_PROMPT = prompts_module.REFLECTION_Q_PROMPT
        REFLECTION_GEN_PROMPT = prompts_module.REFLECTION_GEN_PROMPT
        IMPORTANCE_SYSTEM = prompts_module.IMPORTANCE_SYSTEM
        IMPORTANCE_USER = prompts_module.IMPORTANCE_USER

    n = len(user_rows)
    base_mat  = np.zeros((n, n_runs), dtype=np.int32)
    adj_mat   = np.zeros((n, n_runs), dtype=np.int32)
    steps_mat = np.zeros((n, n_runs), dtype=np.int32)
    real_arr  = user_rows['jbsteps30'].values  # 用于 rolling MAE
    _t_user_start = time.time()
    _llm_calls_at_user_start = llm.call_count

    for i, row in enumerate(user_rows.itertuples(index=False)):
        ctx = {
            'slot':              int(row.day_slot),
            'location':          row.location,
            'weekday':           bool(row.weekday),
            'activity':          row.activity if isinstance(row.activity, str) else 'STILL',
            'weather':           row.weather if isinstance(row.weather, str) else 'Clear',
            'temperature':       float(row.temperature) if not np.isnan(row.temperature) else 22.0,
            'study_day':         1,
            'prior_30min_steps': int(row.jbsteps30pre),  # 关键: 真实 prior
            'response':          row.response if 'response' in row._fields else 'no_send',
        }
        action = int(row.send)
        dosage = float(row.dosage)
        action_desc = {0: "No suggestion", 1: "Active walking suggestion",
                       2: "Sedentary stand-up suggestion"}.get(action, "No suggestion")
        weekday_desc = "weekday" if ctx['weekday'] else "weekend"

        for r in range(n_runs):
            base = DE.generate_base_steps(params, ctx, action, dosage, ext)
            base_mat[i, r] = base
            if base > 0:
                obs_text = stream.format_memories(stream.get_recent(10))
                adj_prompt = PROMPT_ADJUSTMENT.format(
                    persona=persona,
                    study_day=int(row.day_slot),  # not actual study day, but row's slot ID — same as Plan A
                    slot=int(row.day_slot), weekday_desc=weekday_desc,
                    location=ctx['location'], activity=ctx['activity'],
                    weather=ctx['weather'], temperature=ctx['temperature'],
                    prior_30min_steps=ctx['prior_30min_steps'],
                    action_desc=action_desc, base_steps=base,
                    recent_obs=obs_text if obs_text else "No prior observations.")
                adj = llm.judge_adjustment(SYS_ADJUSTMENT, adj_prompt)
                adj = max(-50, min(100, adj))
            else:
                adj = 0
            adj_mat[i, r] = adj
            steps_mat[i, r] = max(0, int(base * (1 + adj / 100.0)))

        # 把这一行的真实 observation 加入 memory stream
        # 用真实 jbsteps30 (不是合成 steps_mat[i].mean()), 这样 memory 始终反映真实历史
        observation = create_observation(
            day=1, slot=int(row.day_slot),
            ctx=ctx, action=action, steps=int(row.jbsteps30))

        if with_reflection:
            imp_prompt = IMPORTANCE_USER.format(observation=observation)
            imp_str = llm.score_importance(IMPORTANCE_SYSTEM, imp_prompt)
            try:
                importance = max(1, min(10, int(imp_str)))
            except Exception:
                importance = 5
        else:
            importance = 5  # 如果不做反思, importance 是常数 (反正 should_reflect 不被检查)

        stream.add(Memory(f"D1S{int(row.day_slot)}_row{i}", observation,
                          "observation", importance))

        # 反思触发 (可选)
        if with_reflection and stream.should_reflect():
            # 简化的反思流程: 借用 synthetic_user_generator 的逻辑会引入文件解析等,
            # 这里手工直跑, 跟原版逻辑等价
            recent = stream.get_recent(20)
            recent_text = stream.format_memories(recent)
            q_prompt = REFLECTION_Q_PROMPT.format(recent_memories=recent_text)
            q_response = llm.generate_text("You are a behavioral analysis system.", q_prompt)
            # 简单解析 (按行切, 取前 3 个非空)
            questions = [q.strip().lstrip('-*0123456789. )')
                         for q in q_response.split('\n')
                         if q.strip()][:3]
            for q in questions:
                ref_prompt = REFLECTION_GEN_PROMPT.format(
                    question=q, relevant_memories=recent_text)
                ref_text = llm.generate_text(
                    "You are a behavioral analysis system. Write concise inferences.",
                    ref_prompt)
                stream.add(Memory(f"D1S{int(row.day_slot)}_ref{i}",
                                  ref_text, "reflection", 8))
            stream.reset_reflection_counter()

        # ── 行级进度 (rolling MAE / bias / 每行延迟) ─────────────────
        if progress_every and (i + 1) % progress_every == 0:
            # rolling: 用到目前为止已跑的行 [0..i]
            rolling_steps = steps_mat[:i+1].mean(axis=1)
            rolling_real  = real_arr[:i+1]
            rolling_err   = rolling_steps - rolling_real
            mae_so_far    = float(np.abs(rolling_err).mean())
            bias_so_far   = float(rolling_err.mean())
            elapsed_user  = time.time() - _t_user_start
            llm_this_user = llm.call_count - _llm_calls_at_user_start
            ms_per_row    = elapsed_user / (i + 1) * 1000
            eta_user      = elapsed_user / (i + 1) * (n - i - 1)
            print(f'    [{uid_label} row {i+1}/{n}] '
                  f'MAE={mae_so_far:.1f} bias={bias_so_far:+.1f} '
                  f'| {ms_per_row:.0f}ms/row, '
                  f'LLM(this user)={llm_this_user}, '
                  f'ETA(user)={eta_user:.0f}s')

    # 汇总
    out = user_rows.copy().reset_index(drop=True)
    out['base_mean']  = base_mat.mean(axis=1)
    out['adj_mean']   = adj_mat.mean(axis=1)
    out['steps_mean'] = steps_mat.mean(axis=1)
    out['steps_one']  = steps_mat[:, 0]
    out['err_mean']   = out['steps_mean'] - out['jbsteps30']
    out['abs_err']    = out['err_mean'].abs()
    return out


# =============================================================================
# 报告 + 绘图 (跟前几个测试结构一致)
# =============================================================================
def report(out_df, save_dir: Path):
    real = out_df['jbsteps30']
    pm   = out_df['steps_mean']
    one  = out_df['steps_one']

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

    # adj 自身的统计 (LLM 调节范围)
    adj = out_df['adj_mean']
    print()
    print(f"  LLM adj_mean             mean: {adj.mean():>+7.1f},  "
          f"std: {adj.std():>6.1f},  range: [{adj.min():.0f}, {adj.max():.0f}]")
    base = out_df['base_mean']
    print(f"  base_mean (no LLM)       mean: {base.mean():>7.1f}")
    print(f"  steps_mean (with LLM)    mean: {pm.mean():>7.1f}  "
          f"(LLM 净影响: {pm.mean()-base.mean():+.1f} 步)")

    # 按 send 分组
    print()
    print('=' * 72)
    print('按 send 分组')
    print('=' * 72)
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

    # 按 slot 分组
    print()
    print('=' * 72)
    print('按 day_slot 分组')
    print('=' * 72)
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

    # 按 location top-8
    print()
    print('=' * 72)
    print('按 location top-8 分组')
    print('=' * 72)
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

    # 桶级 W
    print()
    print('=' * 72)
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
        avg_W = bw['W'].mean()
        avg_mean = bw['mean_real'].mean()
        print(f"  桶定义: (slot, send, location), 门槛 n>=10")
        print(f"  桶数: {len(bw)},  平均 W: {avg_W:.1f}, "
              f"中位 W: {bw['W'].median():.1f},  最差 W: {bw['W'].max():.1f}")
        print(f"  桶级真实均值 (各桶 real 均值的均值): {avg_mean:.1f}")
        print(f"  W / 桶均值 = {avg_W / avg_mean * 100:.1f}%")

    # ─── 绘图 ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) scatter
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

    # (b) error histogram
    ax = axes[0, 1]
    ax.hist(out_df['err_mean'].clip(-1500, 1500), bins=80,
            color='salmon', edgecolor='black', linewidth=0.4)
    ax.axvline(0, color='black', lw=1)
    ax.axvline(out_df['err_mean'].mean(), color='red', lw=2,
               label=f"mean = {out_df['err_mean'].mean():+.0f}")
    ax.set_xlabel('error = steps_mean − real jbsteps30')
    ax.set_ylabel('count')
    ax.set_title('(b) Signed error distribution (clipped to ±1500)')
    ax.legend(); ax.grid(alpha=0.3)

    # (c) distribution overlay
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

    # (d) per-(slot, send) means: 4 series — real / base / steps
    ax = axes[1, 1]
    grp = out_df.groupby(['day_slot', 'send']).agg(
        real=('jbsteps30', 'mean'), base=('base_mean', 'mean'),
        steps=('steps_mean', 'mean'), n=('jbsteps30', 'size')
    ).reset_index()
    grp = grp[grp['n'] >= 10]
    x = np.arange(len(grp))
    w = 0.27
    ax.bar(x - w, grp['real'], w, color='steelblue', edgecolor='black', label='real')
    ax.bar(x, grp['base'], w, color='lightgreen', edgecolor='black', label='base')
    ax.bar(x + w, grp['steps'], w, color='salmon', edgecolor='black', label='pipeline')
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{int(r.day_slot)}/k{int(r.send)}" for r in grp.itertuples()],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('mean steps')
    ax.set_title('(d) Per-(slot, send): real / base / pipeline')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Plan A pipeline (real prior fed in)  vs  real jbsteps30 — train set',
                 fontsize=12, fontweight='bold', y=1.0)
    plt.tight_layout()
    fig_path = save_dir / 'full_pipeline_vs_real.png'
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'\n  图保存: {fig_path}')

    # 逐行 csv
    csv_path = save_dir / 'full_pipeline_vs_real_rows.csv'
    keep = ['uid', 'date', 'day_slot', 'location', 'activity', 'weekday',
            'send', 'dosage', 'jbsteps30pre', 'jbsteps30',
            'base_mean', 'adj_mean', 'steps_mean', 'steps_one',
            'err_mean', 'abs_err']
    out_df[keep].to_csv(csv_path, index=False)
    print(f'  逐行结果保存: {csv_path}')


# =============================================================================
# 主入口
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cleaned', default='./data/cleaned_output.csv')
    p.add_argument('--train',   default='./data/train_uids.json')
    p.add_argument('--test_uids', default=None)
    p.add_argument('--users_csv', default=None)
    p.add_argument('--data_extractor_dir', default='.')
    p.add_argument('--out',     default='./full_pipeline_test_output')
    p.add_argument('--runs',    type=int, default=10,
                   help='每行抽样次数, 用于平均掉 base + LLM 的随机性')
    p.add_argument('--seed',    type=int, default=42)
    p.add_argument('--with-reflection', action='store_true',
                   help='打开后会做 importance scoring + reflection (慢)')
    p.add_argument('--max-users', type=int, default=None,
                   help='只跑前 N 个 train user (smoke test 用)')
    p.add_argument('--llm', choices=['simulated', 'qwen'], default='qwen',
                   help='simulated: SimulatedLLM (随机), qwen: Qwen3BLLM (真模型)')
    p.add_argument('--qwen-path', default='../models/Qwen3-8B-AWQ')
    p.add_argument('--progress-every', type=int, default=50,
                   help='每 N 行打一次用户内进度 (rolling MAE/bias). 0=关掉')
    args = p.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True, parents=True)

    with open(args.train) as f:
        train_uids = json.load(f)
    print(f'[load] {len(train_uids)} train users')
    if args.max_users:
        train_uids = train_uids[:args.max_users]
        print(f'[load] limited to first {len(train_uids)} users for smoke test')

    # test_uids 推断
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

    # users.csv
    if args.users_csv is None:
        users_csv = str(out_dir / 'users_dummy.csv')
        maybe_make_dummy_users_csv(args.cleaned, users_csv)
    else:
        users_csv = args.users_csv
    users_df = pd.read_csv(users_csv)

    # imports
    sys.path.insert(0, args.data_extractor_dir)
    import data_extractor as DE
    import memory_stream as MS
    import prompts as PR
    import llm as LM

    # extractor
    ext = DE.V1DataExtractor(users_csv, args.cleaned,
                             train_uids_path=args.train,
                             test_uids_path=args.test_uids)

    # LLM
    if args.llm == 'qwen':
        print(f'[llm] loading Qwen3BLLM from {args.qwen_path}')
        llm = LM.Qwen3BLLM(model_path=args.qwen_path)
    else:
        print('[llm] using SimulatedLLM (random)')
        llm = LM.SimulatedLLM()

    # 准备数据
    real_df = prepare_real_rows(args.cleaned, train_uids, avail_only=True)
    print(f'[real_rows] {len(real_df)} rows after filtering '
          f'(avail=True, jbsteps30 not null)')

    # 逐用户走
    all_results = []
    t0 = time.time()
    for u_idx, uid in enumerate(train_uids):
        user_rows = real_df[real_df['uid'] == uid]
        if len(user_rows) == 0:
            continue
        params = build_user_params(
            uid=int(uid),
            real_mean=float(user_rows['predicted_mean_steps'].iloc[0]),
            real_te=float(user_rows['predicted_te'].iloc[0]),
            users_df=users_df,
        )
        out_u = evaluate_one_user(
            user_rows, params, ext, llm,
            n_runs=args.runs, with_reflection=args.with_reflection,
            DE=DE, MS_module=MS, prompts_module=PR,
            progress_every=args.progress_every,
            uid_label=f'uid={int(uid)}')
        all_results.append(out_u)

        elapsed = time.time() - t0
        avg_per_user = elapsed / (u_idx + 1)
        eta = avg_per_user * (len(train_uids) - u_idx - 1)
        print(f'  [{u_idx+1}/{len(train_uids)}] uid={int(uid)}: '
              f'n={len(user_rows)}, '
              f'MAE={out_u["abs_err"].mean():.1f}, '
              f'bias={out_u["err_mean"].mean():+.1f}, '
              f'LLM calls={llm.call_count}, '
              f'elapsed={elapsed:.0f}s, ETA={eta:.0f}s')

    out_df = pd.concat(all_results, ignore_index=True)
    print(f'\n[done] total rows={len(out_df)}, '
          f'total LLM calls={llm.call_count}, '
          f'time={time.time()-t0:.0f}s')

    report(out_df, out_dir)


if __name__ == '__main__':
    main()
