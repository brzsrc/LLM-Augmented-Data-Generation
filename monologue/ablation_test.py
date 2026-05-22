"""
Ablation 实验:验证 MONOLOGUE / REASONING 对 POST30 预测的真实影响

设计:对每一行,跑 4 种条件,看 Qwen 给出的 POST30 数字怎么变:
  A. 全量:画像原则 + MONOLOGUE + REASONING (= 你现在的生产配置)
  B. 去 MONOLOGUE:只保留 画像原则 + REASONING
  C. 去 REASONING:只保留 画像原则 + MONOLOGUE
  D. 都去:只保留 画像原则(基线)

如果:
- A ≈ D → 独白和 reasoning 都没用,POST30 只看画像 + 当前数据
- A ≈ B,C 偏离 → REASONING 决定数字,MONOLOGUE 是装饰
- A ≈ C,B 偏离 → MONOLOGUE 决定数字,REASONING 是装饰
- A ≈ B ≈ C ≠ D → 两个都贡献信息,且高度互补
- A ≠ B ≠ C ≠ D → 都有独立贡献

输出:CSV 记录每行 4 种条件下的 POST30,以及行间差异统计
"""

import argparse
import json
import os
import time

import pandas as pd

from llm import Qwen32BLLM
from common import (
    PROMPT_PATHS, add_uid_args, resolve_uids, load_users_for_prediction,
    UserState, build_monologue_prompt, build_post30_prompt,
    parse_monologue_output,
)


# ============================================
# 变体 POST30 prompt 构造器
# ============================================
def build_post30_variant(state, subs, mono_text, reasoning, post_usr_tmpl, variant: str):
    """根据 variant 决定要不要包含 mono / reasoning。
    variant: 'A' = 全量, 'B' = no_mono, 'C' = no_reasoning, 'D' = neither
    """
    if variant == "A":
        mono, reason = mono_text, reasoning
    elif variant == "B":
        mono = "(本次实验隐藏了心理独白部分)"
        reason = reasoning
    elif variant == "C":
        mono = mono_text
        reason = "(本次实验隐藏了 reasoning 部分)"
    elif variant == "D":
        mono = "(本次实验隐藏了心理独白和 reasoning,只基于画像原则和数据预测)"
        reason = ""
    else:
        raise ValueError(variant)

    user_prompt = post_usr_tmpl\
        .replace("{PERSONA_PRINCIPLES}", state.principles)\
        .replace("{MONOLOGUE}", mono)\
        .replace("{REASONING}", reason)
    for k, v in subs.items():
        user_prompt = user_prompt.replace(k, v)
    return user_prompt


def main():
    parser = argparse.ArgumentParser()
    add_uid_args(parser)
    parser.add_argument("--n-rows", type=int, default=30,
                        help="每个用户测多少行(默认 30,够看统计意义)")
    parser.add_argument("--n-repeats", type=int, default=3,
                        help="每个 (row, variant) 跑几次取均值(默认 3,看数字稳定性)")
    parser.add_argument("--output", default="outputs/ablation_results.csv")
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

    print("[load] Qwen3-32B...")
    llm = Qwen32BLLM()
    llm.text_params_no_think.max_tokens = args.mono_max_tokens

    results = []
    t0 = time.time()

    for uid, sub_df, persona in loaded:
        jsonl = f"/tmp/ablation_user_{uid}.jsonl"
        if os.path.exists(jsonl):
            os.remove(jsonl)

        state = UserState(uid, sub_df, persona, jsonl)
        state.open_out()

        n_to_test = min(args.n_rows, state.N)
        print(f"\n[user {uid}] 测试 {n_to_test} 行 × 4 条件 × {args.n_repeats} 次 = "
              f"{n_to_test * 4 * args.n_repeats} 次调用")

        for i in range(n_to_test):
            row = state.current_row()

            # 第一步:生成 MONOLOGUE / REASONING(只生成 1 次,所有 variant 共用)
            subs, mono_user = build_monologue_prompt(state, mono_usr_tmpl)
            raw = llm.generate_text(system=mono_sys, user=mono_user, thinking=False)
            parsed = parse_monologue_output(raw)

            # 第二步:对 4 个 variant,各跑 n_repeats 次 POST30
            row_result = {
                "uid": uid,
                "study_day": int(row["study_day"]),
                "slot": int(row["day_slot"]),
                "location": str(row["location"]),
                "send": int(row["send"]),
                "response": str(row["response"]),
                "pre30": float(row["jbsteps30pre"]),
                "mono": parsed["MONOLOGUE"][:80],
                "reasoning": parsed["REASONING"][:80],
            }

            for variant in ["A", "B", "C", "D"]:
                preds = []
                for _ in range(args.n_repeats):
                    post_user = build_post30_variant(
                        state, subs, parsed["MONOLOGUE"], parsed["REASONING"],
                        post_usr_tmpl, variant
                    )
                    p = llm.judge_steps(system=post_sys, user=post_user)
                    preds.append(int(p))
                row_result[f"post30_{variant}"] = sum(preds) / len(preds)
                row_result[f"post30_{variant}_runs"] = preds

            results.append(row_result)
            elapsed = time.time() - t0
            done = len(results)
            print(f"  [{i+1}/{n_to_test}] d{row_result['study_day']} s{row_result['slot']} "
                  f"loc={row_result['location'][:8]:<8} pre={row_result['pre30']:>5.0f}  "
                  f"A={row_result['post30_A']:>5.0f} B={row_result['post30_B']:>5.0f} "
                  f"C={row_result['post30_C']:>5.0f} D={row_result['post30_D']:>5.0f}  "
                  f"({done} rows in {elapsed:.0f}s)")

            # 推进 state(用 variant A 作为正式预测,写入 jsonl)
            from common import make_result_record
            rec = make_result_record(uid, row, int(row_result["post30_A"]), parsed, raw,
                                     persona=state.persona, principles=state.principles)
            state.append_result(rec)

        state.close_out()

    # ============================================
    # 写 CSV + 统计分析
    # ============================================
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n[done] 已写入 {args.output} ({len(df)} 行)")

    # ── 统计 ──
    print("\n" + "=" * 70)
    print("Ablation 结论")
    print("=" * 70)
    for v in ["A", "B", "C", "D"]:
        m = df[f"post30_{v}"].mean()
        s = df[f"post30_{v}"].std()
        print(f"  {v}: mean={m:>6.1f}  std={s:>6.1f}")

    print()
    print("── A vs B (去 MONOLOGUE 的影响) ──")
    diff_AB = (df["post30_A"] - df["post30_B"]).abs()
    print(f"  |A - B| mean={diff_AB.mean():.1f}  median={diff_AB.median():.1f}  "
          f"max={diff_AB.max():.0f}")
    print(f"  A == B 比例: {(df['post30_A'] == df['post30_B']).mean()*100:.0f}%")
    print(f"  解读: 数字越大,MONOLOGUE 对预测影响越显著")

    print()
    print("── A vs C (去 REASONING 的影响) ──")
    diff_AC = (df["post30_A"] - df["post30_C"]).abs()
    print(f"  |A - C| mean={diff_AC.mean():.1f}  median={diff_AC.median():.1f}  "
          f"max={diff_AC.max():.0f}")
    print(f"  A == C 比例: {(df['post30_A'] == df['post30_C']).mean()*100:.0f}%")

    print()
    print("── A vs D (画像原则单独的力量) ──")
    diff_AD = (df["post30_A"] - df["post30_D"]).abs()
    print(f"  |A - D| mean={diff_AD.mean():.1f}  median={diff_AD.median():.1f}  "
          f"max={diff_AD.max():.0f}")
    print(f"  A == D 比例: {(df['post30_A'] == df['post30_D']).mean()*100:.0f}%")

    print()
    print("── B vs D (REASONING 单独的贡献) ──")
    diff_BD = (df["post30_B"] - df["post30_D"]).abs()
    print(f"  |B - D| mean={diff_BD.mean():.1f}  median={diff_BD.median():.1f}")

    print()
    print("── C vs D (MONOLOGUE 单独的贡献) ──")
    diff_CD = (df["post30_C"] - df["post30_D"]).abs()
    print(f"  |C - D| mean={diff_CD.mean():.1f}  median={diff_CD.median():.1f}")

    print()
    print("── 解读指南 ──")
    print("  如果 A ≈ B → MONOLOGUE 没用,REASONING 撑起一切")
    print("  如果 A ≈ C → REASONING 没用,MONOLOGUE 撑起一切")
    print("  如果 A ≠ B 且 A ≠ C → 两者各有独立贡献")
    print("  如果 A ≈ D → 画像原则就够了,MONOLOGUE/REASONING 都是装饰")
    print("  如果 4 个都差不多 → POST30 主要看画像原则 + 当前数据,中间过程被忽略")


if __name__ == "__main__":
    main()
