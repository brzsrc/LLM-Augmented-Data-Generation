"""
generate_trajectories.py
=========================
用微调后的 Qwen3-0.6B 生成合成天级轨迹。

支持两种模式：
  1. 无约束生成：给 cluster + weekday 条件，模型自由生成
  2. 有约束生成：给定部分 visit 作为 prompt，模型补全

输出：
  synthetic_trajectories.csv  — 和 cleaned_output.csv 同格式

用法：
  # 用合并后的模型
  python generate_trajectories.py \
      --model saves/heartsteps_qwen3_merged \
      --n-users 30 \
      --n-days 43 \
      --output synthetic_trajectories.csv

  # 或用 LoRA adapter（未合并）
  python generate_trajectories.py \
      --model Qwen/Qwen3-0.6B-Base \
      --adapter saves/heartsteps_qwen3_lora \
      --n-users 30 \
      --n-days 43
"""

import argparse
import json
import re
import os
import random
import numpy as np
import pandas as pd
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ================================================================
# 1. 模型加载
# ================================================================

def load_model(model_path, adapter_path=None):
    """加载模型（合并版或 LoRA 版）"""
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )

    if adapter_path:
        print(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    print(f"Model loaded. Device: {next(model.parameters()).device}")
    return model, tokenizer


# ================================================================
# 2. 生成 Prompt
# ================================================================

SYSTEM_PROMPT = (
    "You are a behavioral trajectory generator for an mHealth walking study. "
    "Given a participant's cluster type and day context, generate a realistic "
    "daily trajectory of 3-5 decision points. Each decision point includes: "
    "slot (1-5), location, activity, availability, prior step level, and "
    "notification response. Separate visits with ' ## '. "
    "Output ONLY the trajectory, no explanation."
)

CLUSTER_NAMES = {
    0: "Food_Centric_Worker",
    1: "Home_Work_Commuter",
    2: "Retail_Medical_Worker",
    3: "Mixed_Pattern",
    4: "Student_Outdoor",
}

CLUSTER_WEIGHTS = {0: 1, 1: 18, 2: 9, 3: 2, 4: 7}


def build_prompt(cluster_name, is_weekday, n_slots, tokenizer):
    """构建完整的生成 prompt。"""
    day_type = "weekday" if is_weekday else "weekend"
    instruction = (
        f"Generate a {day_type} trajectory with {n_slots} decision points "
        f"for a participant of type: {cluster_name}."
    )

    # 用 default template 格式（和训练时一致）
    # LLaMA-Factory 的 default template 格式：
    # Human: {system}\n{instruction}\nAssistant:
    prompt = f"Human: {SYSTEM_PROMPT}\n{instruction}\nAssistant: "
    return prompt


# ================================================================
# 3. 生成 + 解析
# ================================================================

def generate_batch(model, tokenizer, prompts, temperature=1.0, max_new_tokens=256):
    """批量生成。"""
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=512
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 只解码新生成的 tokens
    generated = outputs[:, inputs['input_ids'].shape[1]:]
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return texts


# Visit 解析正则
VISIT_PATTERN = re.compile(
    r'slot is (\d+), '
    r'location is (\w+), '
    r'activity is (\w+), '
    r'avail is (\w+), '
    r'steps_pre is (\w+), '
    r'response is (\w+)'
)

# steps_pre 桶 → 数值映射
STEPS_BIN_TO_VALUE = {
    "zero": 0,
    "very_low": 25,
    "low": 100,
    "medium": 225,
    "high": 450,
    "very_high": 800,
}

# 简化 location → 原始 location 映射
LOCATION_REVERSE = {
    'Auto_Transport': 'Auto & Transport - Outdoor Low-Activity',
    'Clothing_Store': 'Clothing & Fashion Store',
    'Electronics': 'Electronics Store',
    'Food': 'Food & Dining',
    'Grocery': 'Grocery & Convenience Store',
    'Healthcare': 'Healthcare & Personal Care - Indoor Low-Activity',
    'Home': 'Home',
    'Furniture_Store': 'Home & Furniture Store',
    'Store': 'Just Store',
    'Leisure': 'Leisure & Entertainment- Indoor Low-Activity',
    'Other_Store': 'Other Specialty Store',
    'Park': 'Parks & Recreation - Outdoor High-Activity',
    'Sports': 'Sports & Fitness - Indoor High-Activity',
    'School': 'University/School',
    'Unknown': 'Unknown',
    'Work': 'Work',
}


def parse_trajectory(text):
    """解析生成的轨迹文本为结构化数据。"""
    visits = []
    for segment in text.split(" ## "):
        segment = segment.strip()
        match = VISIT_PATTERN.match(segment)
        if not match:
            continue

        slot = int(match.group(1))
        location = match.group(2)
        activity = match.group(3)
        avail = match.group(4) == "yes"
        steps_bin = match.group(5)
        response = match.group(6)

        # 验证
        if slot < 1 or slot > 5:
            continue

        visits.append({
            'day_slot': slot,
            'location': LOCATION_REVERSE.get(location, location),
            'activity': activity,
            'avail': avail,
            'jbsteps30pre': STEPS_BIN_TO_VALUE.get(steps_bin, 0),
            'response': response,
        })

    if len(visits) < 3:
        return None

    # 按 slot 排序（恢复时间顺序，Geo-Llama 的 temporal reorder）
    visits.sort(key=lambda v: v['day_slot'])

    # 去重（同一个 slot 只保留第一个）
    seen_slots = set()
    unique_visits = []
    for v in visits:
        if v['day_slot'] not in seen_slots:
            seen_slots.add(v['day_slot'])
            unique_visits.append(v)

    return unique_visits if len(unique_visits) >= 3 else None


# ================================================================
# 4. 分位数校准
# ================================================================

def calibrate_steps_pre(gen_data, real_data):
    """按 slot 分层的分位数校准 jbsteps30pre。"""
    calibrated = gen_data.copy()

    for slot in range(1, 6):
        gen_mask = calibrated['day_slot'] == slot
        real_slot = real_data[real_data['day_slot'] == slot]['jbsteps30pre'].values

        if len(real_slot) == 0 or gen_mask.sum() == 0:
            continue

        gen_vals = calibrated.loc[gen_mask, 'jbsteps30pre'].values
        # 排名 → 映射到真实分布
        ranks = pd.Series(gen_vals).rank(pct=True).values
        ranks = np.clip(ranks, 0.001, 0.999)
        calibrated.loc[gen_mask, 'jbsteps30pre'] = np.quantile(real_slot, ranks).astype(int)

    return calibrated


# ================================================================
# 5. 主生成循环
# ================================================================

def generate_synthetic_dataset(model, tokenizer, n_users=30, n_days=43,
                                temperature=1.0, batch_size=16,
                                real_data=None):
    """生成完整的合成数据集。"""

    # 为每个虚拟用户分配 cluster
    cluster_ids = list(CLUSTER_WEIGHTS.keys())
    cluster_probs = np.array(list(CLUSTER_WEIGHTS.values()), dtype=float)
    cluster_probs /= cluster_probs.sum()
    user_clusters = np.random.choice(cluster_ids, n_users, p=cluster_probs)

    all_rows = []
    total_generated = 0
    total_failed = 0

    for user_id in range(1, n_users + 1):
        cluster_id = user_clusters[user_id - 1]
        cluster_name = CLUSTER_NAMES[cluster_id]

        for day in range(1, n_days + 1):
            is_weekday = (day - 1) % 7 < 5

            # 随机 slot 数（匹配真实分布：69% 有 5 个 slot）
            n_slots = random.choices([3, 4, 5], weights=[0.1, 0.2, 0.7])[0]

            prompt = build_prompt(cluster_name, is_weekday, n_slots, tokenizer)

            # 生成（单条，可以改成批量提速）
            text = generate_batch(model, tokenizer, [prompt],
                                  temperature=temperature, max_new_tokens=256)[0]

            # 解析
            visits = parse_trajectory(text)

            if visits is None:
                total_failed += 1
                # 重试一次
                text = generate_batch(model, tokenizer, [prompt],
                                      temperature=max(0.7, temperature - 0.2),
                                      max_new_tokens=256)[0]
                visits = parse_trajectory(text)

            if visits is None:
                total_failed += 1
                continue

            for v in visits:
                v['user_id'] = user_id
                v['study_day'] = day
                v['cluster'] = cluster_id
                v['cluster_name'] = cluster_name
                v['weekday'] = is_weekday
                all_rows.append(v)

            total_generated += 1

        if user_id % 5 == 0:
            print(f"  User {user_id}/{n_users}: "
                  f"{total_generated} days generated, {total_failed} failed")

    df = pd.DataFrame(all_rows)
    print(f"\nGeneration complete: {len(df)} rows, "
          f"{total_generated} days, {total_failed} failed ({total_failed/(total_generated+total_failed)*100:.1f}%)")

    # 分位数校准 jbsteps30pre
    if real_data is not None:
        print("Calibrating jbsteps30pre...")
        df = calibrate_steps_pre(df, real_data)

    return df


# ================================================================
# 6. 主函数
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='saves/heartsteps_qwen3_merged')
    parser.add_argument('--adapter', default=None,
                        help='LoRA adapter path (if not merged)')
    parser.add_argument('--n-users', type=int, default=30)
    parser.add_argument('--n-days', type=int, default=43)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', default='synthetic_trajectories.csv')
    parser.add_argument('--real-data', default=None,
                        help='真实数据路径（用于分位数校准）')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 加载模型
    model, tokenizer = load_model(args.model, args.adapter)

    # 加载真实数据（用于校准）
    real_data = None
    if args.real_data:
        real_data = pd.read_csv(args.real_data)

    # 生成
    df = generate_synthetic_dataset(
        model, tokenizer,
        n_users=args.n_users,
        n_days=args.n_days,
        temperature=args.temperature,
        batch_size=args.batch_size,
        real_data=real_data,
    )

    # 保存
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

    # 打印统计
    print(f"\n{'='*50}")
    print("Generated data statistics:")
    print(f"{'='*50}")
    print(f"Rows: {len(df)}")
    print(f"Users: {df['user_id'].nunique()}")
    print(f"Location distribution:\n{df['location'].value_counts(normalize=True).head(5)}")
    print(f"Activity distribution:\n{df['activity'].value_counts(normalize=True)}")
    print(f"Avail rate: {df['avail'].mean():.3f}")
    print(f"Steps_pre zero rate: {(df['jbsteps30pre']==0).mean():.3f}")


if __name__ == '__main__':
    main()
