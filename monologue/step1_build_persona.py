"""
Step 1: 为指定 uids 生成画像
用法:
  python step1_build_persona.py                          # 默认跑 CSV 全部用户
  python step1_build_persona.py --uids 1 11 37          # 跑指定用户
  python step1_build_persona.py --uids-file train_uids.json
  python step1_build_persona.py --uids 1 11 37 --force  # 强制重生成已有画像
"""

import argparse
import os

import pandas as pd

from llm import Qwen32BLLM   # ← 改成你实际的 import 路径
from common import (
    PROMPT_PATHS, add_uid_args, resolve_uids, build_rows_table,
)


def main():
    parser = argparse.ArgumentParser()
    add_uid_args(parser)
    parser.add_argument("--force", action="store_true",
                        help="强制重生成已有的画像文件")
    parser.add_argument("--thinking", action="store_true", default=True,
                        help="画像生成使用 thinking 模式(默认开)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="画像输出最大 token 数")
    args = parser.parse_args()

    uids = resolve_uids(args.uids, args.uids_file, args.input_csv)
    os.makedirs(args.persona_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    available_uids = set(df["uid"].unique())

    to_run = []
    skipped = 0
    for uid in uids:
        if uid not in available_uids:
            print(f"[skip] User {uid}: CSV 中无数据")
            skipped += 1
            continue
        out_path = os.path.join(args.persona_dir, f"user_{uid}.md")
        if os.path.exists(out_path) and not args.force:
            skipped += 1
            continue
        to_run.append(uid)

    print(f"[plan] 总 uid {len(uids)}, 已存在跳过 {skipped}, 本次将生成 {len(to_run)}")
    if not to_run:
        print("[done] 无事可做")
        return

    print(f"[load] Qwen3-32B...")
    llm = Qwen32BLLM()
    llm.text_params_thinking.max_tokens = args.max_tokens

    sys_tmpl = open(PROMPT_PATHS["step1_sys"], encoding="utf-8").read()
    usr_tmpl = open(PROMPT_PATHS["step1_usr"], encoding="utf-8").read()

    for i, uid in enumerate(to_run, 1):
        sub = df[df["uid"] == uid].copy()
        rows_table = build_rows_table(sub)
        usr = usr_tmpl\
            .replace("{UID}", str(uid))\
            .replace("{N_ROWS}", str(len(sub)))\
            .replace("{ROWS_TABLE}", rows_table)

        print(f"[{i}/{len(to_run)}] User {uid}: {len(sub)} 行, prompt {len(usr)} chars")
        persona = llm.generate_text(system=sys_tmpl, user=usr, thinking=args.thinking)

        out_path = os.path.join(args.persona_dir, f"user_{uid}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(persona)
        print(f"          -> {out_path}")


if __name__ == "__main__":
    main()
