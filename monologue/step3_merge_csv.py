"""
Step 3: 把 jsonl 预测合并到原 CSV,新增 post30_pred 等列

用法:
  python step3_merge_csv.py                          # 合并 jsonl 目录里所有用户
  python step3_merge_csv.py --uids 1 11 37
  python step3_merge_csv.py --uids-file train_uids.json
  python step3_merge_csv.py --output outputs/my_preds.csv
"""

import argparse
import glob
import json
import os
import re

import pandas as pd

from common import add_uid_args, resolve_uids, DEFAULT_PREDICTIONS_CSV


def discover_uids_from_jsonl(monologue_dir: str) -> list:
    """从 jsonl 文件名解析出已有结果的 uid。"""
    uids = []
    for path in glob.glob(os.path.join(monologue_dir, "user_*.jsonl")):
        m = re.search(r"user_(\d+)\.jsonl$", path)
        if m:
            uids.append(int(m.group(1)))
    return sorted(uids)


def main():
    parser = argparse.ArgumentParser()
    add_uid_args(parser)
    parser.add_argument("--output", default=DEFAULT_PREDICTIONS_CSV,
                        help="输出 CSV 路径")
    parser.add_argument("--all-from-jsonl", action="store_true",
                        help="自动从 jsonl 目录发现所有 uid(忽略 --uids)")
    args = parser.parse_args()

    # 决定要合并哪些 uid
    if args.all_from_jsonl:
        uids = discover_uids_from_jsonl(args.monologue_dir)
        print(f"[discover] 从 {args.monologue_dir} 发现 {len(uids)} 个 uid: {uids[:10]}...")
    else:
        uids = resolve_uids(args.uids, args.uids_file, args.input_csv)

    df = pd.read_csv(args.input_csv)

    # 读所有预测
    preds = {}
    loaded_uids = []
    missing_uids = []
    for uid in uids:
        path = os.path.join(args.monologue_dir, f"user_{uid}.jsonl")
        if not os.path.exists(path):
            missing_uids.append(uid)
            continue
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                preds[(r["uid"], r["study_day"], r["slot"])] = r
                n += 1
        loaded_uids.append((uid, n))

    if missing_uids:
        print(f"[warn] 缺 jsonl: {missing_uids[:10]}{'...' if len(missing_uids)>10 else ''}")
    print(f"[load] 加载 {len(loaded_uids)} 个用户的预测,共 {len(preds)} 条")

    # 字段查找
    def lookup(row, field):
        rec = preds.get((row["uid"], row["study_day"], row["day_slot"]))
        return rec[field] if rec else None

    df["post30_pred"] = df.apply(lambda r: lookup(r, "post30"), axis=1)
    df["monologue"] = df.apply(lambda r: lookup(r, "monologue"), axis=1)
    df["reasoning"] = df.apply(lambda r: lookup(r, "reasoning"), axis=1)
    df["summary"] = df.apply(lambda r: lookup(r, "summary"), axis=1)

    target_uids = set(uid for uid, _ in loaded_uids)
    out = df[df["uid"].isin(target_uids)].copy()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)

    n_total = len(out)
    n_pred = out["post30_pred"].notna().sum()
    print(f"[done] {n_pred}/{n_total} 行有预测 -> {args.output}")

    # 如果输入 CSV 含 jbsteps30(真实 post30),给个准确度
    if "jbsteps30" in out.columns:
        ok = out.dropna(subset=["post30_pred", "jbsteps30"])
        if len(ok) > 0:
            err = (ok["post30_pred"] - ok["jbsteps30"]).abs()
            print(f"      MAE: {err.mean():.1f}, Median AE: {err.median():.1f}, n={len(ok)}")


if __name__ == "__main__":
    main()
