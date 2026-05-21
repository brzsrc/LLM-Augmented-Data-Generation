"""
Step 2 (PARALLEL): 跨用户同步推进 step k,vLLM 批量调用

用法:
  python step2_predict_rows_parallel.py                              # CSV 全部用户
  python step2_predict_rows_parallel.py --uids 1 11 37
  python step2_predict_rows_parallel.py --uids-file train_uids.json
  python step2_predict_rows_parallel.py --uids 1 11 37 --max-batch 8 # 限制单 batch 大小
  python step2_predict_rows_parallel.py --thinking                   # 独白用 thinking 模式(慢但质量高)

设计:
- 每个用户独立按 (date, day_slot) 顺序前进,所有用户在同一 k 对齐
- 单 step 内:active_users 同时构造 prompt -> 一次 batch_text + 一次 batch_steps
- vLLM 内部 continuous batching 处理并发
- max_batch 限制:当用户数很多时,单 batch 太大会爆显存,自动切片

可调参数(都有合理默认值):
- --max-batch:单次 batch 最大 prompt 数(默认无限制,vLLM 自己处理)
- --history-recent:近 K 行带完整独白(默认 20)
- --progress-every:每 N step 打一次进度(默认 10)
"""

import argparse
import os
import time

import pandas as pd

from llm import Qwen32BLLM   # ← 改成你实际的 import 路径
from common import (
    PROMPT_PATHS, add_uid_args, resolve_uids, load_users_for_prediction,
    UserState, build_monologue_prompt, build_post30_prompt,
    parse_monologue_output, make_result_record,
)


# ============================================
# 批切片:当 active_uids 太多时,分多个 batch 避免显存爆
# ============================================
def chunked(lst, n):
    """把 lst 切成长度 n 的块。n<=0 表示不切。"""
    if n <= 0 or len(lst) <= n:
        yield lst
        return
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ============================================
# 主流程:跨用户同步推进
# ============================================
def run_parallel(states: dict, llm: Qwen32BLLM,
                 mono_sys: str, mono_usr_tmpl: str,
                 post_sys: str, post_usr_tmpl: str,
                 thinking_mono: bool = False,
                 max_batch: int = 0,
                 progress_every: int = 10):
    sorted_uids = sorted(states.keys())
    max_steps = max(s.N for s in states.values())
    total_rows = sum(s.N for s in states.values())
    already_done = sum(len(s.prior_rows) for s in states.values())
    print(f"[parallel] {len(states)} users, max_steps={max_steps}, "
          f"total_rows={total_rows}, resumed={already_done}, "
          f"max_batch={'unlimited' if max_batch<=0 else max_batch}")

    t0 = time.time()
    rows_done = 0
    last_log = t0

    for k in range(max_steps):
        active_uids = [uid for uid in sorted_uids
                       if states[uid].cursor == k and states[uid].cursor < states[uid].N]
        if not active_uids:
            continue

        # ── 构造所有独白 prompt ──
        all_mono_prompts = []
        prompt_meta = []
        for uid in active_uids:
            s = states[uid]
            subs, user_prompt = build_monologue_prompt(s, mono_usr_tmpl)
            all_mono_prompts.append({"system": mono_sys, "user": user_prompt})
            prompt_meta.append((uid, subs))

        # ── 分块 batch 跑独白 ──
        all_raw_outputs = []
        for chunk in chunked(all_mono_prompts, max_batch):
            raw = llm.batch_text(chunk, thinking=thinking_mono)
            all_raw_outputs.extend(raw)

        # ── 解析独白 ──
        parsed_by_uid = {}
        for (uid, subs), raw in zip(prompt_meta, all_raw_outputs):
            parsed = parse_monologue_output(raw)
            parsed_by_uid[uid] = (subs, parsed, raw)

        # ── 构造 POST30 prompt + 分块 batch 跑数字 ──
        all_post_prompts = []
        post_meta = []
        for uid in active_uids:
            s = states[uid]
            subs, parsed, _ = parsed_by_uid[uid]
            user_prompt = build_post30_prompt(
                s, subs, parsed["MONOLOGUE"], parsed["REASONING"], post_usr_tmpl
            )
            all_post_prompts.append({"system": post_sys, "user": user_prompt})
            post_meta.append(uid)

        all_post30 = []
        for chunk in chunked(all_post_prompts, max_batch):
            res = llm.batch_steps(chunk)
            all_post30.extend(res)

        # ── 写结果 + 推进 cursor ──
        for uid, post30 in zip(post_meta, all_post30):
            s = states[uid]
            subs, parsed, raw = parsed_by_uid[uid]
            row = s.current_row()
            rec = make_result_record(uid, row, int(post30), parsed, raw)
            s.append_result(rec)
            rows_done += 1

        # ── 进度日志 ──
        if (k + 1) % progress_every == 0 or (time.time() - last_log) > 60:
            dt = time.time() - t0
            remaining = total_rows - already_done - rows_done
            eta = dt / max(1, rows_done) * remaining
            print(f"[step k={k+1:>3}/{max_steps}] active={len(active_uids):>2}  "
                  f"rows_done={rows_done}/{total_rows-already_done}  "
                  f"elapsed={dt:.0f}s  ETA={eta:.0f}s  "
                  f"llm_calls={llm.call_count}")
            last_log = time.time()

    print(f"[done] elapsed={time.time()-t0:.0f}s, rows={rows_done}, "
          f"llm_calls={llm.call_count}")


def catch_up_lagging_users(states: dict, llm: Qwen32BLLM,
                           mono_sys: str, mono_usr_tmpl: str,
                           post_sys: str, post_usr_tmpl: str,
                           thinking_mono: bool):
    """如果不同用户 cursor 不一致(中断 + 重跑或手动删 jsonl 导致),
    把落后用户串行追到最大 cursor,然后整体进入并行模式。"""
    max_c = max(s.cursor for s in states.values())
    min_c = min(s.cursor for s in states.values())
    if min_c >= max_c:
        return  # 已对齐

    print(f"[align] cursor 不一致: min={min_c} max={max_c}, 串行追平...")
    for uid, s in states.items():
        if s.cursor >= max_c:
            continue
        while s.cursor < max_c:
            subs, mono_user = build_monologue_prompt(s, mono_usr_tmpl)
            raw = llm.generate_text(system=mono_sys, user=mono_user, thinking=thinking_mono)
            parsed = parse_monologue_output(raw)
            post_user = build_post30_prompt(
                s, subs, parsed["MONOLOGUE"], parsed["REASONING"], post_usr_tmpl
            )
            post30 = llm.judge_steps(system=post_sys, user=post_user)
            row = s.current_row()
            rec = make_result_record(uid, row, int(post30), parsed, raw)
            s.append_result(rec)
            print(f"  catch-up User {uid}: cursor={s.cursor}/{max_c}")
    print(f"[align] 全部对齐到 cursor={max_c}, 进入并行模式")


def main():
    parser = argparse.ArgumentParser()
    add_uid_args(parser)
    parser.add_argument("--thinking", action="store_true", default=False,
                        help="独白生成用 thinking 模式(慢 3-5x,质量更高)")
    parser.add_argument("--max_batch", type=int, default=0,
                        help="单次 batch 最大 prompt 数,0=无限制(默认)")
    parser.add_argument("--progress-every", type=int, default=10,
                        help="每 N step 打一次进度")
    parser.add_argument("--mono-max-tokens", type=int, default=600,
                        help="独白阶段 no_think 最大输出 token,默认 600")
    args = parser.parse_args()

    uids = resolve_uids(args.uids, args.uids_file, args.input_csv)
    loaded = load_users_for_prediction(uids, args.input_csv, args.persona_dir,
                                       skip_missing_persona=True)
    if not loaded:
        print("[err] 没有可处理的用户(检查画像是否生成)")
        return

    # 加载 prompts
    mono_sys = open(PROMPT_PATHS["mono_sys"], encoding="utf-8").read()
    mono_usr_tmpl = open(PROMPT_PATHS["mono_usr"], encoding="utf-8").read()
    post_sys = open(PROMPT_PATHS["post_sys"], encoding="utf-8").read()
    post_usr_tmpl = open(PROMPT_PATHS["post_usr"], encoding="utf-8").read()

    # 加载 LLM
    print(f"[load] Qwen3-32B...")
    llm = Qwen32BLLM()
    llm.text_params_no_think.max_tokens = args.mono_max_tokens

    # 创建 UserState
    os.makedirs(args.monologue_dir, exist_ok=True)
    states = {}
    for uid, sub_df, persona in loaded:
        jsonl = os.path.join(args.monologue_dir, f"user_{uid}.jsonl")
        states[uid] = UserState(uid, sub_df, persona, jsonl)
        print(f"[init] User {uid}: N={states[uid].N}, "
              f"resumed cursor={states[uid].cursor}")

    # 对齐 + 并行跑
    try:
        # 先打开所有句柄
        for s in states.values():
            s.open_out()
        catch_up_lagging_users(states, llm, mono_sys, mono_usr_tmpl,
                               post_sys, post_usr_tmpl, args.thinking)
        run_parallel(states, llm, mono_sys, mono_usr_tmpl, post_sys, post_usr_tmpl,
                     thinking_mono=args.thinking, max_batch=args.max_batch,
                     progress_every=args.progress_every)
    finally:
        for s in states.values():
            s.close_out()


if __name__ == "__main__":
    main()
