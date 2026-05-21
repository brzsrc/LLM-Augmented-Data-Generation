"""
Step 2 (SERIAL): 逐行揣测 — 每行 2 次单调用(独白 + POST30)

用法同 step2_predict_rows_parallel.py,但内部不批量。
适合调试或用户数特别少时。

  python step2_predict_rows.py --uids 1 11 37
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


def predict_user(state: UserState, llm: Qwen32BLLM,
                 mono_sys, mono_usr_tmpl, post_sys, post_usr_tmpl,
                 thinking: bool = False):
    print(f"[run ] User {state.uid}: 起始 cursor={state.cursor}/{state.N}")
    t0 = time.time()
    while state.cursor < state.N:
        subs, mono_user = build_monologue_prompt(state, mono_usr_tmpl)
        raw = llm.generate_text(system=mono_sys, user=mono_user, thinking=thinking)
        parsed = parse_monologue_output(raw)
        post_user = build_post30_prompt(state, subs,
                                        parsed["MONOLOGUE"], parsed["REASONING"],
                                        post_usr_tmpl)
        post30 = llm.judge_steps(system=post_sys, user=post_user)
        row = state.current_row()
        rec = make_result_record(state.uid, row, int(post30), parsed, raw)
        state.append_result(rec)

        if state.cursor % 10 == 0:
            dt = time.time() - t0
            eta = dt / max(1, state.cursor) * (state.N - state.cursor)
            print(f"       d{rec['study_day']:>2} s{rec['slot']} "
                  f"pre={rec['pre30']:>5.0f} -> post={rec['post30']:>4}  "
                  f"({state.cursor}/{state.N}, ETA {eta:.0f}s)")


def main():
    parser = argparse.ArgumentParser()
    add_uid_args(parser)
    parser.add_argument("--qwen-path", default="../models/Qwen3-32B-AWQ")
    parser.add_argument("--thinking", action="store_true", default=False)
    parser.add_argument("--mono-max-tokens", type=int, default=600)
    args = parser.parse_args()

    uids = resolve_uids(args.uids, args.uids_file, args.input_csv)
    loaded = load_users_for_prediction(uids, args.input_csv, args.persona_dir,
                                       skip_missing_persona=True)
    if not loaded:
        print("[err] 没有可处理的用户")
        return

    mono_sys = open(PROMPT_PATHS["mono_sys"], encoding="utf-8").read()
    mono_usr_tmpl = open(PROMPT_PATHS["mono_usr"], encoding="utf-8").read()
    post_sys = open(PROMPT_PATHS["post_sys"], encoding="utf-8").read()
    post_usr_tmpl = open(PROMPT_PATHS["post_usr"], encoding="utf-8").read()

    print(f"[load] Qwen3-32B from {args.qwen_path}")
    llm = Qwen32BLLM(model_path=args.qwen_path)
    llm.text_params_no_think.max_tokens = args.mono_max_tokens

    os.makedirs(args.monologue_dir, exist_ok=True)
    for uid, sub_df, persona in loaded:
        jsonl = os.path.join(args.monologue_dir, f"user_{uid}.jsonl")
        state = UserState(uid, sub_df, persona, jsonl)
        state.open_out()
        try:
            predict_user(state, llm, mono_sys, mono_usr_tmpl,
                         post_sys, post_usr_tmpl, args.thinking)
        finally:
            state.close_out()


if __name__ == "__main__":
    main()
