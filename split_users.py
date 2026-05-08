"""
split_users.py — 一次性生成 train/test split（按 uid 分）

跑一次后产出 data/train_uids.json 和 data/test_uids.json，
后续所有脚本（cluster_analysis.py, data_extractor.py 等）
都从这两个文件读 split，确保整个流水线一致。

用法:
    python data/split_users.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

CLEANED = 'data/cleaned_output.csv'
OUT_DIR = Path('data')
N_TEST = 7
SEED = 42


def main():
    df = pd.read_csv(CLEANED)
    all_uids = sorted(int(u) for u in df['uid'].unique())
    print(f"Total users: {len(all_uids)}")

    rng = np.random.default_rng(SEED)
    shuffled = all_uids.copy()
    rng.shuffle(shuffled)

    test_uids = sorted(int(x) for x in shuffled[:N_TEST])
    train_uids = sorted(int(x) for x in shuffled[N_TEST:])

    assert len(set(train_uids) & set(test_uids)) == 0, "split overlap!"
    assert len(train_uids) + len(test_uids) == len(all_uids), "split size mismatch!"

    print(f"Train ({len(train_uids)}): {train_uids}")
    print(f"Test  ({len(test_uids)}): {test_uids}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / 'train_uids.json', 'w') as f:
        json.dump(train_uids, f)
    with open(OUT_DIR / 'test_uids.json', 'w') as f:
        json.dump(test_uids, f)
    print(f"\nSaved: {OUT_DIR}/train_uids.json, {OUT_DIR}/test_uids.json")


if __name__ == '__main__':
    main()
