import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import re

from memory_stream import create_observation, MemoryStream, Memory
from prompts import REFLECTION_Q_SYS, REFLECTION_GEN_SYS, PROMPT_STEPS, SYS_STEPS, IMPORTANCE_SYSTEM, IMPORTANCE_USER, \
    REFLECTION_Q_PROMPT, REFLECTION_GEN_PROMPT

matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'DejaVu Sans'


def _build_rich_persona(udf):
    """
    从all_df数据 derive 一段自然语言 persona, 用于 LLM prompt.
    """
    ref = udf

    # 基本面: 真实步数均值 (over all rows, including unavail)
    overall_mean = float(ref['jbsteps30pre'].mean()) if len(ref) > 0 else 0.0

    # 各 slot 的均值, 找最活跃 / 最不活跃
    slot_means = {}
    for s in [1, 2, 3, 4, 5]:
        sub = ref[ref['day_slot'] == s]
        if len(sub) > 0:
            slot_means[s] = float(sub['jbsteps30pre'].mean())
    if slot_means:
        most_active_slot = max(slot_means, key=slot_means.get)
        least_active_slot = min(slot_means, key=slot_means.get)
        slot_desc_map = {1: "early morning", 2: "morning", 3: "midday",
                         4: "afternoon", 5: "evening"}
        most_desc = slot_desc_map.get(most_active_slot, f"slot {most_active_slot}")
        least_desc = slot_desc_map.get(least_active_slot, f"slot {least_active_slot}")
        most_steps = int(slot_means[most_active_slot])
        least_steps = int(slot_means[least_active_slot])
    else:
        most_desc, least_desc, most_steps, least_steps = "midday", "early morning", 0, 0

    # 整体零率
    zero_rate = float((ref['jbsteps30pre'] == 0).mean()) if len(ref) > 0 else 0.0

    # location 分布
    loc_counts = ref['location'].value_counts(normalize=True)
    top3_locs = loc_counts.head(5)
    loc_strs = [f"{l} ({p * 100:.0f}%)" for l, p in top3_locs.items()]
    loc_desc = ", ".join(loc_strs) if loc_strs else "various"

    text = (
        f"Behavioral profile (user general walking behavior without intervention):\n"
        f"- Walks ~{int(overall_mean)} steps per 30-min slot on average; "
        f"{int(zero_rate * 100)}% of slots have zero steps.\n"
        f"- Most active in the {most_desc} (slot {most_active_slot if slot_means else '?'}, "
        f"~{most_steps} steps); least active in the {least_desc} "
        f"(slot {least_active_slot if slot_means else '?'}, ~{least_steps} steps).\n"
        f"- Top5 most frequent locations visited: {loc_desc}.\n"
    )
    return text


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks and any stray tags from LLM output."""
    # 移除完整的 <think>...</think> 块（包括跨行）
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 移除残留的孤立标签
    text = text.replace('<think>', '').replace('</think>', '')
    return text.strip()


def _parse_questions(raw: str, max_n: int = 3) -> List[str]:
    """
    Robustly parse a list of questions out of LLM output.
    Handles: leading </think> tags, blank lines, numbered list prefixes
    like '1.', '1)', '- ', '* '. Only keeps lines that look like real questions.
    """
    cleaned = _strip_think_tags(raw)
    questions = []
    for line in cleaned.split('\n'):
        line = line.strip()
        if not line:
            continue
        # 去掉行首的列表序号: "1.", "1)", "1:", "- ", "* ", "Q1:", etc.
        line = re.sub(r'^\s*(?:Q\s*\d+[.):]?|\d+[.):]|[-*•])\s*', '', line, flags=re.IGNORECASE).strip()
        if not line:
            continue
        # 只保留像问题的行：有一定长度，且形式上像一句话/问句
        # 排除明显不是问题的残留标签或元注释
        if len(line) < 15:
            continue
        if line.lower().startswith(('<think', '</think', 'note:', 'here are', 'list ')):
            continue
        questions.append(line)
        if len(questions) >= max_n:
            break
    return questions


def prepare_real_rows(cleaned_path: str, avail_only: bool = False) -> pd.DataFrame:
    df = pd.read_csv(cleaned_path)
    if avail_only:
        df = df[df['avail'] == True].copy()

    df = df.sort_values(['uid', 'study_day', 'day_slot']).reset_index(drop=True)

    dosage_list = []
    for uid, ud in df.groupby('uid', sort=False):
        d = 0.0
        for s in ud['send'].values:
            d = 0.95 * d + (1.0 if int(s) > 0 else 0.0)
            dosage_list.append(d)
    df['dosage'] = dosage_list

    return df


# =============================================================================
# 用户运行时状态: 一个用户一个 stream + cursor
# =============================================================================
class UserRuntime:
    def __init__(self, uid, all_rows, prompt_persona, logger=None):
        self.uid = uid
        self.prompt_persona = prompt_persona
        self.logger = logger
        # 重要: reset_index 让 .iloc[k] 工作; 排序确保 personal time order
        self.all_rows = all_rows.sort_values(['study_day', 'day_slot']).reset_index(drop=True)
        self.N = len(self.all_rows)

        self.stream = MemoryStream()
        for line in prompt_persona.strip().split('. '):
            if line.strip():
                self.stream.add(Memory(
                    "study_start", line.strip() + ".", "background", 7))

        # 输出缓冲: 一行一个 dict, 与 all_rows 同顺序; 未处理保持 None
        self.results = [None] * self.N


# =============================================================================
# 核心: 按 step k 推进, 每 step 内 batch active users × n_runs 个 prompt
# =============================================================================
def run_parallel_pipeline(real_df, llm, n_runs, with_reflection,
                          progress_every_step: int = 25,
                          verbose: bool = True,
                          logger_dir: 'Path' = None):
    """跑完所有用户, 返回 runtimes 字典。

    logger_dir: 若给了 Path, 每个用户会创建 TraceLogger 写到 {logger_dir}/user{uid}/.
                每个 runtime.logger 也会被赋上, 让 Phase D/E 钩子可以调用.
    """
    # logger 可选
    LoggerCls = None
    if logger_dir is not None:
        try:
            from trace_logger import TraceLogger
            LoggerCls = TraceLogger
        except ImportError as e:
            print(f'[logger] 不能 import trace_logger ({e}), 禁用')
            LoggerCls = None

    # ── Step 1: 给每个用户初始化 runtime ────────────────────────────
    runtimes = {}
    for uid, udf in real_df.groupby('uid'):
        # per-user logger (可选)
        user_logger = None
        prompt_persona = _build_rich_persona(udf)
        if LoggerCls is not None:
            user_dir = logger_dir / f'user{int(uid)}'
            user_logger = LoggerCls(str(user_dir),
                                    user_id=int(uid))
            user_logger.log_user_init(persona=prompt_persona)

        runtimes[uid] = UserRuntime(uid, udf, prompt_persona, logger=user_logger)

    sorted_uids = sorted(runtimes.keys())
    max_steps = max(rt.N for rt in runtimes.values())
    total_rows = sum(rt.N for rt in runtimes.values())
    print(f'[parallel] {len(runtimes)} users, max_steps={max_steps}, '
          f'total_rows={total_rows}')

    # ── Step 2: 按 k 推进 ───────────────────────────────────────────
    t0 = time.time()
    rows_done = 0

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

            ctx = {
                'slot': int(row['day_slot']),
                'location': row['location'],
                'weekday': bool(row['is_weekday']),
                'activity': row['activity'],
                'weather': row['weather'],
                'temperature': row['temperature'],
                'study_day': int(row['study_day']),
                'prior_30min_steps': int(row['jbsteps30pre']),
                'response': row['response']
                    if 'response' in rt.all_rows.columns and isinstance(row['response'], str)
                    else 'no_send',
            }
            action = int(row['send'])
            dosage = float(row['dosage'])

            obs_text = rt.stream.format_memories(rt.stream.get_recent(10, "observation"))
            ref_text = rt.stream.format_memories(rt.stream.get_recent(3, "reflection"))

            recent_obs = obs_text if obs_text else "No prior observations."
            recent_ref = ref_text if ref_text else "No prior reflections."

            active_ctx[uid] = (ctx, action, dosage, row, recent_obs)

            action_desc = {0: "No suggestion", 1: "Active walking suggestion",
                           2: "Sedentary stand-up suggestion"}.get(action, "No suggestion")
            weekday_desc = "weekday" if ctx['weekday'] else "weekend"


            # n_runs prompts for this row — all identical (memory stream is the same
            # at this step); n_runs only captures LLM stochasticity
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
                    recent_obs=recent_obs,
                    recent_ref=recent_ref,
                )
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

        # ── Phase D: importance + 加入 observation memory  ─────────
        # 完整对应 synthetic_user_generator.simulate_parallel 的 Phase C+D.
        all_obs = []  # uid 顺序与 active_uids 对齐
        for uid in active_uids:
            rt = runtimes[uid]
            ctx, action, _, row, _ = active_ctx[uid]
            obs = create_observation(
                day=row['study_day'], slot=int(row['day_slot']),
                ctx=ctx, action=action, steps=int(rt.results[k]['steps_mean']))
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

                ctx_for_log = dict(ctx)
                ctx_for_log.setdefault('avail', 1)
                ctx_for_log.setdefault('yesterday_steps', 0)
                rt.logger.log_decision_point(
                    day=row['study_day'], slot=slot_n, ctx=ctx_for_log,
                    action=action,
                    final_steps=int(rt.results[k]['steps_mean']),
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
                    _, _, _, row, _ = active_ctx[uid]
                    if rt.logger is None:
                        continue
                    inferences = per_uid_inferences[uid]
                    questions = per_uid_questions[uid]
                    combined_text = "\n\n".join(inferences) if inferences else ""
                    rt.logger.log_reflection(
                        day=row['study_day'],
                        reflection_text=combined_text,
                        reflect_prompt=per_uid_q_prompt_text[uid],
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
            print(f'  [step {k+1}/{max_steps}] '
                  f'active={len(active_uids):>2} users, '
                  f'batch={len(all_prompts):>3} prompts '
                  f'({llm_calls_this_step} LLM calls in {step_elapsed_ms:.0f}ms), '
                  f'rows_done={rows_done}/{total_rows} '
                  f'LLM_total={llm.call_count}, '
                  f'elapsed={elapsed:.0f}s, ETA={eta:.0f}s')

    print(f'[parallel] done: {rows_done}/{total_rows} rows in '
          f'{time.time()-t0:.0f}s, total LLM calls={llm.call_count}')
    return runtimes


def assemble_output(real_df: pd.DataFrame, runtimes: dict) -> pd.DataFrame:
    """把 runtimes.results 拼回 real_df, 加 steps_mean/steps_one/steps_std。"""
    real_df = real_df.copy()
    real_df = real_df.sort_values(['uid', 'study_day', 'day_slot']).reset_index(drop=True)

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
    real_df['jbsteps30'] = sm_col
    real_df['steps_one']  = so_col
    real_df['steps_std']  = sstd_col
    return real_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cleaned',   default='./sft/synthetic_trajectories.csv')
    p.add_argument('--out',       default='./synthetic_trajectories')
    p.add_argument('--runs',      type=int, default=5,
                   help='每行抽样次数, 平均掉 base + LLM 的随机性')
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--with-reflection', action='store_true',
                   help='打开后做 importance scoring + reflection (慢)')
    p.add_argument('--max_users', type=int, default=None,
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

    import llm as LM

    if args.llm == 'qwen':
        print(f'[llm] loading Qwen3BLLM from {args.qwen_path}')
        llm = LM.Qwen3BLLM(model_path=args.qwen_path)
    else:
        print('[llm] using SimulatedLLM (random)')
        llm = LM.SimulatedLLM()

    real_df = prepare_real_rows(args.cleaned, avail_only=False)
    print(f'[real_rows] {len(real_df)} rows')

    if args.max_users:
        real_df = real_df[real_df['uid'].isin(range(1, args.max_users+1))]

    t0 = time.time()
    # 默认开启 logger; --no-logger 关
    logger_dir_arg = None if args.no_logger else out_dir
    if logger_dir_arg is not None:
        print(f'[logger] enabled, per-user traces -> {out_dir}/user<uid>/')

    runtimes = run_parallel_pipeline(
        real_df, llm, n_runs=args.runs,
        with_reflection=args.with_reflection,
        progress_every_step=args.progress_every_step, verbose=True,
        logger_dir=logger_dir_arg)
    out_df = assemble_output(real_df, runtimes)
    print(f'\n[done] total rows={len(out_df)}, '
          f'total LLM calls={llm.call_count}, '
          f'time={time.time()-t0:.0f}s')

    csv_path = out_dir / 'full_pipeline_vs_real_rows.csv'
    keep = ['uid', 'day_slot', 'location', 'activity', 'is_weekday',
            'weather', 'temperature',
            'send', 'response', 'jbsteps30pre', 'jbsteps30'
            'steps_mean', 'steps_one', 'steps_std']
    out_df[keep].to_csv(csv_path, index=False)
    print(f'  逐行结果保存: {csv_path}')

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




if __name__ == '__main__':
    main()
