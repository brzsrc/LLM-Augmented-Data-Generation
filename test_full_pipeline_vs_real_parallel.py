"""
test_full_pipeline_vs_real_parallel.py
======================================
test_full_pipeline_vs_real.py 的批量并行版本。

跟 serial 版本相比的差异:
  - serial: 30 个用户挨个跑, 每行 1 个 LLM 调用
  - parallel: 跨用户同步推进, 同一 step 所有 active 用户的 LLM 调用合成一个大 batch
              用 llm.batch_steps() 一次提交 → vLLM 上能拿到 5-15× 加速

并行设计 (与 simulate_parallel 一致, 适配真实数据):
  - 每个用户 N_u 个真实行, 按 (date, day_slot) 排序成 personal timeline
  - 全局推进 step k = 0, 1, 2, ..., max_N_u
  - 在 step k, active users = [u for u in users if k < N_u]:
      * 每个抽 n_runs 次 base
      * 这批 active_users × n_runs 个 prompt 一起 batch_steps
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

from prompts import REFLECTION_Q_SYS, REFLECTION_GEN_SYS

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
    def __init__(self, uid, params, persona, all_rows, MS_module,
                 prompt_persona=None, logger=None):
        """
        persona: 简短的 5-line 描述, 用于 seed memory stream (保持兼容)
        prompt_persona: rich persona (来自 ext.get_rich_persona), 用于 LLM prompt;
                        若 None 则用 persona
        logger: 可选的 TraceLogger 实例 (per-user). 若给了, 会被 pipeline 用来
                落盘 observation / decision_point / reflection. user_init 和
                finalize 由调用方负责.
        """
        self.uid = uid
        self.params = params
        self.persona = persona
        self.prompt_persona = prompt_persona if prompt_persona is not None else persona
        self.logger = logger
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
                          verbose: bool = True,
                          logger_dir: 'Path' = None):
    """跑完所有用户, 返回 runtimes 字典。

    logger_dir: 若给了 Path, 每个用户会创建 TraceLogger 写到 {logger_dir}/user{uid}/.
                每个 runtime.logger 也会被赋上, 让 Phase D/E 钩子可以调用.
    """
    Memory = MS_module.Memory
    create_observation = MS_module.create_observation
    PROMPT_STEPS = prompts_module.PROMPT_STEPS
    SYS_STEPS = prompts_module.SYS_STEPS

    # logger 可选
    LoggerCls = None
    if logger_dir is not None:
        try:
            from trace_logger import TraceLogger
            LoggerCls = TraceLogger
        except ImportError as e:
            print(f'[logger] 不能 import trace_logger ({e}), 禁用')
            LoggerCls = None

    if with_reflection:
        REFLECTION_Q_PROMPT = prompts_module.REFLECTION_Q_PROMPT
        REFLECTION_GEN_PROMPT = prompts_module.REFLECTION_GEN_PROMPT
        IMPORTANCE_SYSTEM = prompts_module.IMPORTANCE_SYSTEM
        IMPORTANCE_USER = prompts_module.IMPORTANCE_USER
        # 完整复用 simulator 的反思解析逻辑 (处理 <think> 块/列表前缀/杂质行)
        from synthetic_user_generator import _parse_questions, _strip_think_tags

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
        # ── 方法 3: rich persona (从前 5 天 derive 出来的, 见 data_extractor) ──
        if hasattr(ext, 'user_persona_text') and int(uid) in ext.user_persona_text:
            prompt_persona = ext.user_persona_text[int(uid)]
        else:
            prompt_persona = persona  # fallback

        # per-user logger (可选)
        user_logger = None
        if LoggerCls is not None:
            user_dir = logger_dir / f'user{int(uid)}'
            user_logger = LoggerCls(str(user_dir),
                                    user_id=int(uid),
                                    user_params=params)
            user_logger.log_user_init(
                params=params,
                initial_state={'motivation': 0, 'habit': 0, 'receptivity': 0},
                persona=prompt_persona,
            )

        runtimes[uid] = UserRuntime(uid, params, persona, ud, MS_module,
                                    prompt_persona=prompt_persona,
                                    logger=user_logger)

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

        # ── Phase A: 给每个 active 用户构造 n_runs 个 prompt ─
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
                'study_day': int(row['study_day']),
                'prior_30min_steps': int(row['jbsteps30pre']),
                'response': row['response']
                    if 'response' in rt.all_rows.columns and isinstance(row['response'], str)
                    else 'no_send',
            }
            action = int(row['send'])
            dosage = float(row['dosage'])

            obs_text = rt.stream.format_memories(rt.stream.get_recent(10))
            recent_obs = obs_text if obs_text else "No prior observations."
            active_ctx[uid] = (ctx, action, dosage, row, recent_obs)

            action_desc = {0: "No suggestion", 1: "Active walking suggestion",
                           2: "Sedentary stand-up suggestion"}.get(action, "No suggestion")
            weekday_desc = "weekday" if ctx['weekday'] else "weekend"

            # reference points
            # if hasattr(ext, 'get_user_bin_mean'):
            #     u_bin, u_src = ext.get_user_bin_mean(
            #         int(uid), int(row['day_slot']), ctx['location'])
            #     pop_bin = ext.get_pop_bin_mean(int(row['day_slot']), ctx['location'])
            #     u_overall = ext.user_overall_mean.get(int(uid),
            #                                           float(ext.global_mean_steps))
            # else:
            #     u_bin, u_src = float(rt.params['predicted_mean_steps']), 'fallback'
            #     pop_bin = float(ext.global_mean_steps) if hasattr(ext, 'global_mean_steps') else 250.0
            #     u_overall = float(rt.params['predicted_mean_steps'])
            #
            # # slot-matched history (strict before row.date)
            # if hasattr(ext, 'get_user_slot_history'):
            #     hist = ext.get_user_slot_history(
            #         int(uid), int(row['day_slot']),
            #         before_date=row['date'], max_items=5)
            # else:
            #     hist = []
            # if hist:
            #     slot_history = "\n".join(
            #         f"- {d} (slot {int(row['day_slot'])}): {s} steps" for d, s in hist)
            # else:
            #     slot_history = "(no prior data for this slot before today)"

            # n_runs prompts for this row — all identical (memory stream is the same
            # at this step); n_runs only captures LLM stochasticity (temperature=0.3)
            for run_idx in range(n_runs):
                p = PROMPT_STEPS.format(
                    persona=rt.prompt_persona,
                    study_day=int(row['study_day']),
                    slot=int(row['day_slot']),
                    weekday_desc=weekday_desc,
                    location=ctx['location'], activity=ctx['activity'],
                    weather=ctx['weather'], temperature=ctx['temperature'],
                    prior_30min_steps=ctx['prior_30min_steps'],
                    action_desc=action_desc,
                    recent_obs=recent_obs)
                all_prompts.append({"system": SYS_STEPS, "user": p})
                prompt_meta.append((uid, run_idx))

        # ── Phase B: 一次大 batch ─────────────────────────────────
        steps_results = llm.batch_steps(all_prompts) if all_prompts else []

        # ── Phase C: 把 steps 填回 + 写 results ─────────────────────
        steps_buffers = {uid: [0] * n_runs for uid in active_uids}
        for k_p, (uid, run_idx) in enumerate(prompt_meta):
            s = max(0, int(steps_results[k_p]))
            steps_buffers[uid][run_idx] = s

        for uid in active_uids:
            rt = runtimes[uid]
            ctx, action, _, row, _ = active_ctx[uid]
            steps_arr = np.array(steps_buffers[uid])
            rt.results[k] = {
                'steps_mean': float(steps_arr.mean()),
                'steps_one':  int(steps_arr[0]),
                'steps_std':  float(steps_arr.std()),
            }
            err = float(steps_arr.mean()) - float(row['jbsteps30'])
            accum_abs_err += abs(err)
            accum_signed_err += err

        # ── Phase D: importance + 加入 observation memory  ─────────
        # 完整对应 synthetic_user_generator.simulate_parallel 的 Phase C+D.
        all_obs = []  # uid 顺序与 active_uids 对齐
        for uid in active_uids:
            rt = runtimes[uid]
            ctx, action, _, row, _ = active_ctx[uid]
            obs = create_observation(
                day=row['study_day'], slot=int(row['day_slot']),
                ctx=ctx, action=action, steps=int(row['jbsteps30']))
            all_obs.append(obs)

        if with_reflection:
            imp_prompts = [{"system": IMPORTANCE_SYSTEM,
                            "user": IMPORTANCE_USER.format(observation=obs)}
                           for obs in all_obs]
            imp_results = llm.batch_importance(imp_prompts) if imp_prompts else []
            all_imp = []
            for r in imp_results:
                try: all_imp.append(max(1, min(10, int(r))))
                except: all_imp.append(5)
        else:
            all_imp = [5] * len(active_uids)

        for idx_u, uid in enumerate(active_uids):
            rt = runtimes[uid]
            ctx, action, _, row, _ = active_ctx[uid]
            rt.stream.add(Memory(f"Day{row['study_day']}_S{int(row['day_slot'])}",
                                 all_obs[idx_u], "observation", all_imp[idx_u]))

            # ── Logger 钩子 1: log_observation + log_decision_point ──
            if rt.logger is not None:
                slot_n = int(row['day_slot'])
                rt.logger.log_observation(
                    day=row['study_day'], slot=slot_n,
                    obs_text=all_obs[idx_u],
                    importance=all_imp[idx_u],
                    importance_acc=rt.stream.importance_since_reflection,
                )
                # Plan B 已废 base/adj, 占位写 0
                ctx_for_log = dict(ctx)
                ctx_for_log.setdefault('avail', 1)
                ctx_for_log.setdefault('yesterday_steps', 0)
                rt.logger.log_decision_point(
                    day=row['study_day'], slot=slot_n, ctx=ctx_for_log,
                    action=action,
                    base_steps=0,
                    llm_adj_pct=0,
                    final_steps=int(rt.results[k]['steps_mean']),
                    state={'motivation': 0, 'habit': 0, 'receptivity': 0},
                    dosage=float(active_ctx[uid][2]),  # dosage from active_ctx tuple
                    prompt_text="",  # parallel 不留 prompt 文本副本, 太占空间
                    llm_raw_output=str(int(rt.results[k]['steps_mean'])),
                    importance=all_imp[idx_u],
                    importance_acc=rt.stream.importance_since_reflection,
                    reflection_triggered=False,  # 反思如果触发, log_reflection 另存
                )

        # ── Phase E: 反思 — 逐字搬 simulate_parallel 的逻辑 ─────────
        if with_reflection:
            reflect_uids = [uid for uid in active_uids
                            if runtimes[uid].stream.should_reflect()]
            if reflect_uids:
                # Step 1: 批量生成问题 — 跟 simulator 用 get_recent(15)
                # 同时缓存每个 uid 的 recent_text + q_prompt 给 logger 用
                q_prompts = []
                per_uid_q_prompt_text = {}   # uid -> q_prompt user content
                per_uid_recent_text = {}     # uid -> recent_text fed to LLM
                for uid in reflect_uids:
                    recent = runtimes[uid].stream.get_recent(25)
                    recent_text = runtimes[uid].stream.format_memories(recent)
                    q_user = REFLECTION_Q_PROMPT.format(recent_memories=recent_text)
                    per_uid_q_prompt_text[uid] = q_user
                    per_uid_recent_text[uid] = recent_text
                    q_prompts.append({
                        "system": REFLECTION_Q_SYS,
                        "user": q_user
                    })
                q_responses = llm.batch_text(q_prompts)

                # Step 2: 批量推断 — 用 _parse_questions, get_recent(15)
                # 同时按 uid 分组 questions, 之后好喂给 log_reflection
                inf_prompts = []
                inf_map = []  # (reflect_local_idx j, uid, question)
                per_uid_questions = {uid: [] for uid in reflect_uids}
                per_uid_inferences = {uid: [] for uid in reflect_uids}
                for j, uid in enumerate(reflect_uids):
                    recent_text = per_uid_recent_text[uid]
                    questions = _parse_questions(q_responses[j], max_n=3)
                    per_uid_questions[uid] = questions
                    for q in questions:
                        inf_prompts.append({
                            "system": REFLECTION_GEN_SYS,
                            "user": REFLECTION_GEN_PROMPT.format(
                                question=q, relevant_memories=recent_text)
                        })
                        inf_map.append((j, uid, q))

                inf_responses = llm.batch_text(inf_prompts) if inf_prompts else []
                # 写入记忆流 — 清洗 <think>...</think> 块
                for k_inf, (j, uid, q) in enumerate(inf_map):
                    resp = inf_responses[k_inf] if k_inf < len(inf_responses) else ""
                    resp = _strip_think_tags(resp)
                    per_uid_inferences[uid].append(resp)
                    _, _, _, row, _ = active_ctx[uid]
                    runtimes[uid].stream.add(Memory(
                        f"Day{row['study_day']}_S{int(row['day_slot'])}",
                        resp, "reflection", 8))

                for uid in reflect_uids:
                    runtimes[uid].stream.reset_reflection_counter()

                # ── Logger 钩子 3: log_reflection per uid ──────────
                # 对每个反思了的 uid, 写一条 log_reflection
                for uid in reflect_uids:
                    rt = runtimes[uid]
                    if rt.logger is None:
                        continue
                    state_zero = {'motivation': 0, 'habit': 0, 'receptivity': 0}
                    inferences = per_uid_inferences[uid]
                    questions = per_uid_questions[uid]
                    combined_text = "\n\n".join(inferences) if inferences else ""
                    rt.logger.log_reflection(
                        day=row['study_day'],
                        reflection_text=combined_text,
                        old_state=state_zero, new_state=state_zero,
                        raw_llm_state=(0, 0, 0),
                        constrained_state=state_zero,
                        reflect_prompt=per_uid_q_prompt_text[uid],
                        state_prompt="",  # Plan B 简化版无 state-update prompt
                        recent_obs_text=per_uid_recent_text[uid],
                        recent_ref_text="",
                        questions=questions,
                        inferences=inferences,
                    )

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
    """把 runtimes.results 拼回 real_df, 加 steps_mean/steps_one/steps_std。"""
    real_df = real_df.copy()
    real_df = real_df.sort_values(['uid', 'date', 'day_slot']).reset_index(drop=True)

    sm_col   = np.zeros(len(real_df), dtype=float)
    so_col   = np.zeros(len(real_df), dtype=int)
    sstd_col = np.zeros(len(real_df), dtype=float)

    cursor = 0
    for uid, ud in real_df.groupby('uid', sort=False):
        rt = runtimes[uid]
        if len(ud) != rt.N:
            print(f'[warn] uid={uid}: real_df slice n={len(ud)} != runtime.N={rt.N}')
        for k_local in range(len(ud)):
            r = rt.results[k_local] if k_local < rt.N and rt.results[k_local] is not None \
                else {'steps_mean': 0.0, 'steps_one': 0, 'steps_std': 0.0}
            i = cursor + k_local
            sm_col[i]   = r['steps_mean']
            so_col[i]   = r['steps_one']
            sstd_col[i] = r.get('steps_std', 0.0)
        cursor += len(ud)

    real_df['steps_mean'] = sm_col
    real_df['steps_one']  = so_col
    real_df['steps_std']  = sstd_col
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

    print()
    print(f"  LLM steps_mean over rows mean: {pm.mean():>7.1f},  "
          f"std (per-row var) mean: {out_df['steps_std'].mean():>6.1f}")
    print(f"  P(steps_mean == 0)         : {(pm == 0).mean()*100:>6.1f}%")
    print(f"  P(real == 0)               : {(real == 0).mean()*100:>6.1f}%")

    print(); print('=' * 72); print('按 send 分组'); print('=' * 72)
    print(f"  {'send':>5} | {'n':>5} | {'real':>7} | "
          f"{'steps':>7} | {'bias':>7} | {'MAE':>6}")
    print('  ' + '-' * 60)
    for s in sorted(out_df['send'].unique()):
        sub = out_df[out_df['send'] == s]
        print(f"  {int(s):>5} | {len(sub):>5} | {sub['jbsteps30'].mean():>7.1f} | "
              f"{sub['steps_mean'].mean():>7.1f} | "
              f"{(sub['steps_mean']-sub['jbsteps30']).mean():>+7.1f} | "
              f"{sub['abs_err'].mean():>6.1f}")

    print(); print('=' * 72); print('按 day_slot 分组'); print('=' * 72)
    print(f"  {'slot':>5} | {'n':>5} | {'real':>7} | "
          f"{'steps':>7} | {'bias':>7} | {'MAE':>6}")
    print('  ' + '-' * 60)
    for s in sorted(out_df['day_slot'].unique()):
        sub = out_df[out_df['day_slot'] == s]
        print(f"  {int(s):>5} | {len(sub):>5} | {sub['jbsteps30'].mean():>7.1f} | "
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
    ax.hist(one, bins=bins, alpha=0.55, density=True,
            color='salmon', edgecolor='black', linewidth=0.4,
            label=f'pipeline steps_one (mean={one.mean():.0f})')
    ax.hist(pm, bins=bins, alpha=0.35, density=True,
            color='gold', edgecolor='black', linewidth=0.4,
            label=f'pipeline steps_mean (mean={pm.mean():.0f})')
    ax.set_xlabel('steps'); ax.set_ylabel('density')
    ax.set_title(f'(c) Distribution overlay (truncated 1500)\n'
                 f'W(real, steps_one)={wasserstein_distance(real, one):.1f}')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    grp = out_df.groupby(['day_slot', 'send']).agg(
        real=('jbsteps30', 'mean'),
        steps=('steps_mean', 'mean'), n=('jbsteps30', 'size')
    ).reset_index()
    grp = grp[grp['n'] >= 10]
    x = np.arange(len(grp)); w = 0.4
    ax.bar(x - w/2, grp['real'], w, color='steelblue', edgecolor='black', label='real')
    ax.bar(x + w/2, grp['steps'], w, color='salmon', edgecolor='black', label='pipeline')
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{int(r.day_slot)}/k{int(r.send)}" for r in grp.itertuples()],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('mean steps')
    ax.set_title('(d) Per-(slot, send): real vs pipeline')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Plan B pipeline (PARALLEL, LLM predicts absolute steps)  vs  real jbsteps30 — train set',
                 fontsize=12, fontweight='bold', y=1.0)
    plt.tight_layout()
    fig_path = save_dir / 'full_pipeline_vs_real.png'
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'\n  图保存: {fig_path}')

    csv_path = save_dir / 'full_pipeline_vs_real_rows.csv'
    keep = ['uid', 'date', 'day_slot', 'location', 'activity', 'weekday',
            'send', 'dosage', 'jbsteps30pre', 'jbsteps30',
            'steps_mean', 'steps_one', 'steps_std',
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
    p.add_argument('--no-logger', action='store_true',
                   help='关闭 per-user trace logger (默认每用户写一个子目录的 trace 文件)')
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
    # 默认开启 logger; --no-logger 关
    logger_dir_arg = None if args.no_logger else out_dir
    if logger_dir_arg is not None:
        print(f'[logger] enabled, per-user traces -> {out_dir}/user<uid>/')

    runtimes = run_parallel_pipeline(
        real_df, ext, llm, n_runs=args.runs,
        with_reflection=args.with_reflection,
        DE=DE, MS_module=MS, prompts_module=PR,
        progress_every_step=args.progress_every_step, verbose=True,
        logger_dir=logger_dir_arg)
    out_df = assemble_output(real_df, runtimes)
    print(f'\n[done] total rows={len(out_df)}, '
          f'total LLM calls={llm.call_count}, '
          f'time={time.time()-t0:.0f}s')

    # finalize per-user logger 用 assemble_output 后的总结指标
    if logger_dir_arg is not None:
        for uid, rt in runtimes.items():
            if rt.logger is None:
                continue
            sub = out_df[out_df['uid'] == uid]
            if len(sub) == 0:
                rt.logger.finalize(
                    final_state={'motivation': 0, 'habit': 0, 'receptivity': 0},
                    summary_stats={'n_rows': 0})
                continue
            rt.logger.finalize(
                final_state={'motivation': 0, 'habit': 0, 'receptivity': 0},
                summary_stats={
                    'n_rows': int(len(sub)),
                    'mae': float(sub['abs_err'].mean()),
                    'bias': float(sub['err_mean'].mean()),
                    'mean_real': float(sub['jbsteps30'].mean()),
                    'mean_pred': float(sub['steps_mean'].mean()),
                })

    # 调试: 打印前 10 个原始 LLM 输出, 帮助诊断"全是 0"这类问题
    if hasattr(llm, 'debug_steps_outputs') and llm.debug_steps_outputs:
        print()
        print('=' * 72)
        print('LLM raw outputs (前 10 次调用, 用于 debug parsing):')
        print('=' * 72)
        for i, raw in enumerate(llm.debug_steps_outputs[:10]):
            print(f'  [{i}] {raw!r}')
        print('=' * 72)

    report(out_df, out_dir)


if __name__ == '__main__':
    main()
