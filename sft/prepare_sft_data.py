"""
prepare_sft_data.py
====================
把 cleaned_output.csv 转成 LLaMA-Factory 的 SFT 数据格式。
参考 Geo-Llama 的天级轨迹序列化 + visit-wise permutation。

输出：
  data/heartsteps_train.json  — LLaMA-Factory alpaca 格式
  data/heartsteps_eval.json   — 验证集（可选）

用法：
  python prepare_sft_data.py \
      --input cleaned_output.csv \
      --train-uids train_uids.json \
      --output-dir data \
      --n-permutations 10 \
      --min-slots 3
"""

import argparse
import json
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

from data.cluster_config import CLUSTER_UIDS, CLUSTER_NAMES


# ================================================================
# 1. 序列化：把一天的决策点转成 Geo-Llama 风格的文本
# ================================================================

def serialize_visit(row):
    """把一个决策点（visit）序列化为文本。

    格式参考 Geo-Llama: "arrival time is X, location is Y, duration is Z"
    适配 HeartSteps: 把 (slot, avail, steps_pre, send, response, activity,
                       location, weather, temperature) 拼成一条 visit。
    """
    avail = "yes" if row['avail'] else "no"
    is_rand = "yes" if row['is_randomized'] else "no"

    # send 语义化（0=没发, 1=非活动消息, 2=活动消息）
    send_map = {0: "no_send", 1: "non_activity", 2: "activity"}
    send_str = send_map.get(int(row['send']), str(row['send']))

    return (f"day_slot is {int(row['day_slot'])}, "
            # f"avail is {avail}, "
            f"in_trial is {is_rand}, "
            f"weather is {row['weather']}, "
            f"temperature is {row['temperature']}, "
            f"location is {row['location']}, "  # ← 提前
            f"activity is {row['activity']}, "  # ← 提前
            f"steps_pre is {row['jbsteps30pre_bucket']}"  # ← 现在能看到所有上下文
            f"send is {send_str}, "
            f"response is {row['response']}, "
    )


def serialize_day_trajectory(day_rows, permute=False):
    """把一天的所有决策点序列化为一条轨迹文本。

    参考 Geo-Llama 的 visit-wise permutation：
    训练时随机打乱 visit 顺序，生成时按 slot 排序恢复。
    """
    visits = []
    for _, row in day_rows.iterrows():
        visits.append(serialize_visit(row))

    if permute:
        random.shuffle(visits)

    return " ## ".join(visits)


# ================================================================
# 2. 构建 LLaMA-Factory 数据集
# ================================================================

SYSTEM_PROMPT = (
    "You are a behavioral trajectory generator for an mHealth walking study. "
    "Given a participant's activity profile(activity level: low/mid/high, "
    "zero-step tendency: rare/common/frequent) and day context, generate a realistic "
    "daily trajectory of 3-5 decision points. Each decision point includes: "
    "day_slot (1-5), trial eligibility, weather, temperature, "
    "location, activity state, prior 30-min step level, sent notification type, and "
    "user response. Separate decision points with ' ## '. "
    "Output ONLY the trajectory, no explanation."
)



def compute_user_profiles(df):
    """计算每个 uid 的步数画像"""
    profiles = {}
    for uid in df['uid'].unique():
        user_df = df[df['uid'] == uid]
        mean = user_df['jbsteps30pre'].mean()
        zero_rate = (user_df['jbsteps30pre'] == 0).mean()

        # 离散化为标签
        if mean < 150:
            activity_level = "low"
        elif mean < 350:
            activity_level = "mid"
        else:
            activity_level = "high"

        if zero_rate < 0.20:
            zero_tendency = "rare"
        elif zero_rate < 0.40:
            zero_tendency = "common"
        else:
            zero_tendency = "frequent"

        profiles[uid] = {
            'activity_level': activity_level,
            'zero_tendency': zero_tendency,
        }
    return profiles

def build_instruction(user_profile, is_weekday, n_slots):
    day_type = "weekday" if is_weekday else "weekend"
    act = user_profile['activity_level']
    zero = user_profile['zero_tendency']

    return (f"Generate a {day_type} trajectory with {n_slots} decision points "
            f"for a participant with {act} activity level "
            f"and {zero} zero-step periods.")


def build_sft_examples(df, n_permutations=10, min_slots=3):
    """把整个数据集转成 LLaMA-Factory 的 alpaca 格式。

    每条天级轨迹生成 n_permutations 个不同排列的训练样本。
    """
    examples = []
    profiles = compute_user_profiles(df)

    # 按 (uid, date) 分组
    for (uid, date), day_group in df.groupby(['uid', 'date']):
        day_group = day_group.sort_values('day_slot')

        if len(day_group) < min_slots:
            continue

        # is_weekday：取这一天的第一个值（同一天所有行应该一致）
        is_weekday = bool(day_group['is_weekday'].iloc[0])
        n_slots = len(day_group)

        instruction = build_instruction(profiles[uid], is_weekday, n_slots)

        # 原始顺序（1 份）
        original_text = serialize_day_trajectory(day_group, permute=False)
        examples.append({
            "instruction": instruction,
            "input": "",
            "output": original_text,
            "system": SYSTEM_PROMPT,
        })

        # 随机排列（n_permutations - 1 份）
        for _ in range(n_permutations - 1):
            permuted_text = serialize_day_trajectory(day_group, permute=True)
            examples.append({
                "instruction": instruction,
                "input": "",
                "output": permuted_text,
                "system": SYSTEM_PROMPT,
            })

    return examples



# ================================================================
# 4. 主函数
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='../data/cleaned_output.csv')
    parser.add_argument('--train-uids', default='../data/train_uids.json',
                        help='JSON file with list of train uid integers')
    parser.add_argument('--output-dir', default='data')
    parser.add_argument('--n-permutations', type=int, default=10,
                        help='每条轨迹生成多少个排列变体')
    parser.add_argument('--min-slots', type=int, default=3,
                        help='最少 slot 数（少于此数的天被丢弃）')
    parser.add_argument('--eval-ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # 读取数据
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows, {df['uid'].nunique()} users")

    # 筛选 train users
    if args.train_uids:
        with open(args.train_uids) as f:
            train_uids = json.load(f)
        df = df[df['uid'].isin(train_uids)]
        print(f"Filtered to {len(df)} rows, {df['uid'].nunique()} train users")

    # ================================================================
    # 基于训练数据计算 jbsteps30pre 的分位数切分点
    # ================================================================
    nonzero = df.loc[df['jbsteps30pre'] > 0, 'jbsteps30pre']
    edges_nz = nonzero.quantile(np.linspace(0, 1, 10)).values
    edges_nz = np.round(edges_nz).astype(int).tolist()

    # 最终边界 [-1, 0, ..., +inf]
    BIN_EDGES = [-1, 0] + edges_nz[1:-1] + [float('inf')]
    BIN_LABELS = [
        'steps_zero',  # = 0
        'steps_q1',    # (0, 35]
        'steps_q2',    # (35, 69]
        'steps_q3',    # (69, 111]
        'steps_q4',    # (111, 164]
        'steps_q5',    # (164, 233]
        'steps_q6',    # (233, 321]
        'steps_q7',    # (321, 475]
        'steps_q8',    # (475, 830]
        'steps_high',  # > 830
    ]
    print(f"\nSteps bin edges: {BIN_EDGES}")

    # pd.cut 把数值分到对应桶
    df['jbsteps30pre_bucket'] = pd.cut(
        df['jbsteps30pre'],
        bins=BIN_EDGES,
        labels=BIN_LABELS,
        include_lowest=False,  # 左边界 -1，0 落在 (-1, 0] 即 steps_zero
    )
    # 强转为 str（pd.cut 返回 Categorical，序列化时会报错）
    df['jbsteps30pre_bucket'] = df['jbsteps30pre_bucket'].astype(str)

    # 验证无未分桶
    assert df['jbsteps30pre_bucket'].notna().all(), "有未分桶的步数值"

    # 保存切分点供 inference 复用
    bins_path = os.path.join(args.output_dir, 'steps_bins.json')
    with open(bins_path, 'w') as f:
        json.dump({
            'edges': [float(e) if e != float('inf') else 'inf' for e in BIN_EDGES],
            'labels': BIN_LABELS,
        }, f, indent=2)
    print(f"Saved bin edges to {bins_path}")

    all_uids = df['uid'].unique()
    np.random.shuffle(all_uids)
    n_eval_uids = max(1, int(len(all_uids) * args.eval_ratio))
    eval_uids = set(all_uids[:n_eval_uids])
    # 训练前就过滤
    train_df = df[~df['uid'].isin(eval_uids)]
    eval_df = df[df['uid'].isin(eval_uids)]

    # 构建 SFT 数据
    train_examples = build_sft_examples(
        train_df,
        n_permutations=args.n_permutations,
        min_slots=args.min_slots
    )
    n_unique_days = len(train_examples) // args.n_permutations
    print(f"\nGenerated {len(train_examples)} training examples "
          f"({n_unique_days} unique days × {args.n_permutations} permutations)")

    eval_examples = build_sft_examples(
        eval_df,
        n_permutations=args.n_permutations,
        min_slots=args.min_slots
    )
    n_unique_days = len(eval_examples) // args.n_permutations
    print(f"\nGenerated {len(eval_examples)} eval examples "
          f"({n_unique_days} unique days × {args.n_permutations} permutations)")

    # 保存
    train_path = os.path.join(args.output_dir, 'heartsteps_train.json')
    eval_path = os.path.join(args.output_dir, 'heartsteps_eval.json')

    with open(train_path, 'w') as f:
        json.dump(train_examples, f, indent=2, ensure_ascii=False)
    with open(eval_path, 'w') as f:
        json.dump(eval_examples, f, indent=2, ensure_ascii=False)

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_examples)} examples)")
    print(f"  Eval:  {eval_path} ({len(eval_examples)} examples)")

    # 打印样本
    print(f"\n{'='*70}")
    print("Sample training example:")
    print(f"{'='*70}")
    sample = train_examples[0]
    print(f"System:      {sample['system'][:80]}...")
    print(f"Instruction: {sample['instruction']}")
    print(f"Output:      {sample['output']}")

    # 统计
    output_lengths = [len(ex['output'].split()) for ex in train_examples]
    print(f"\nOutput word count: mean={np.mean(output_lengths):.0f}, "
          f"max={max(output_lengths)}, min={min(output_lengths)}")


if __name__ == '__main__':
    main()
