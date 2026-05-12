"""
generate_trajectories.py
=========================
用微调后的 Qwen3-0.6B 生成合成天级轨迹。

输出：
  synthetic_trajectories.csv  — 和 cleaned_output.csv 同格式

用法：
  # 用合并后的模型
  python generate_trajectories.py \
      --model saves/heartsteps_qwen3_merged \
      --output synthetic_trajectories.csv

  # 或用 LoRA adapter（未合并）
  python generate_trajectories.py \
      --model ../../models/Qwen3-0.6B-Base \
      --adapter saves/heartsteps_qwen3_lora/checkpoint-800 \
      --output synthetic_trajectories.csv

注意:
  脚本中的 SYSTEM_PROMPT、序列化字段顺序、steps 桶名 必须和 prepare_sft_data.py 完全一致。
  当前训练数据里 steps_pre 和 send 之间缺逗号（粘连）, 本脚本的解析正则已适配这一点。
"""

import argparse
import json
import re
import os
import random
import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ================================================================
# 必须和 prepare_sft_data.py 中的 SYSTEM_PROMPT 完全一致
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
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    if adapter_path:
        print(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    print(f"Model loaded. Device: {next(model.parameters()).device}")
    return model, tokenizer


# ================================================================
# 2. 构建 Prompt（必须和 build_instruction 一致）
# ================================================================

def build_instruction(activity_level, zero_tendency, is_weekday, n_slots):
    """构建 instruction，必须和 prepare_sft_data.py 的 build_instruction 完全一致。"""
    day_type = "weekday" if is_weekday else "weekend"
    return (f"Generate a {day_type} trajectory with {n_slots} decision points "
            f"for a participant with {activity_level} activity level "
            f"and {zero_tendency} zero-step periods.")


def build_prompt(activity_level, zero_tendency, is_weekday, n_slots, tokenizer):
    """用 tokenizer 的 chat template 构建 prompt（自动套上 Qwen3 的 ChatML 格式）。"""
    instruction = build_instruction(activity_level, zero_tendency, is_weekday, n_slots)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    # tokenize=False 拿到字符串；add_generation_prompt=True 加上 "<|im_start|>assistant\n"
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


# ================================================================
# 3. 生成 + 解析
# ================================================================

def generate_batch(model, tokenizer, prompts, temperature=1.0, max_new_tokens=512):
    """批量生成。"""
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 只解码新生成的 tokens
    generated = outputs[:, inputs['input_ids'].shape[1]:]
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return texts


# ================================================================
# Visit 解析正则
# ================================================================
# 训练数据的实际格式（注意 steps_pre 和 send 之间没有逗号，是粘连的）：
# "day_slot is 2, in_trial is yes, weather is Clear, temperature is temp_warm,
#  location is Work, activity is STILL, steps_pre is steps_q5send is non_activity,
#  response is good, "
#
# 用宽松匹配（允许字段间空白稍变），并适配 "{bucket}send" 粘连。
VISIT_PATTERN = re.compile(
    r'day_slot is (\d+),\s*'
    r'in_trial is (yes|no),\s*'
    r'weather is (\w+),\s*'
    r'temperature is (\w+),\s*'
    r'location is (\w+),\s*'
    r'activity is (\w+),\s*'
    r'steps_pre is (\w+?)'      # ← 非贪婪，因为后面紧跟 "send"
    r'send is (\w+),\s*'
    r'response is (\w+)',
    re.IGNORECASE,
)


# 10 个步数桶（必须和 prepare_sft_data.py 完全一致）
# 这些是 instruction 学到的 token，反映 prepare 计算的分位数桶
STEPS_BUCKETS = [
    'steps_zero',
    'steps_q1', 'steps_q2', 'steps_q3', 'steps_q4',
    'steps_q5', 'steps_q6', 'steps_q7', 'steps_q8',
    'steps_high',
]


def parse_trajectory(text):
    """解析生成的轨迹文本为结构化数据。"""
    visits = []
    for segment in text.split(" ## "):
        segment = segment.strip()
        match = VISIT_PATTERN.search(segment)
        if not match:
            continue

        slot = int(match.group(1))
        in_trial = match.group(2) == "yes"
        weather = match.group(3)
        temperature = match.group(4)
        location = match.group(5)
        activity = match.group(6)
        steps_bucket = match.group(7)
        send = match.group(8)
        response = match.group(9)

        # 验证
        if slot < 1 or slot > 5:
            continue
        if steps_bucket not in STEPS_BUCKETS:
            # 模型可能生成无效桶名，跳过这一 visit
            continue

        visits.append({
            'day_slot': slot,
            'is_randomized': in_trial,
            'weather': weather,
            'temperature': temperature,
            'location': location,
            'activity': activity,
            'jbsteps30pre_bucket': steps_bucket,
            'send': send,
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
# 4. 步数桶 → 实际步数值（用真实数据的分位数采样）
# ================================================================

def build_bucket_to_value_sampler(real_data, bin_edges):
    """
    构造从桶名 → 实际步数值的采样器。

    bin_edges 是 prepare_sft_data.py 用过的切分点（保存在 steps_bins.json）。
    对每个桶，从真实数据中**该桶内的所有值**中均匀采样（保留分布）。
    """
    bucket_to_values = {}

    real_steps = real_data['jbsteps30pre'].values

    # 各桶对应的边界（左开右闭，除 zero 是 ==0）
    for i, label in enumerate(STEPS_BUCKETS):
        if label == 'steps_zero':
            vals = real_steps[real_steps == 0]
        elif label == 'steps_high':
            vals = real_steps[real_steps > bin_edges[-2]]
        else:
            lo, hi = bin_edges[i], bin_edges[i + 1]
            vals = real_steps[(real_steps > lo) & (real_steps <= hi)]

        if len(vals) == 0:
            # 兜底：用桶的中点
            if label == 'steps_zero':
                vals = np.array([0])
            elif label == 'steps_high':
                vals = np.array([bin_edges[-2] + 100])
            else:
                vals = np.array([(bin_edges[i] + bin_edges[i + 1]) / 2])

        bucket_to_values[label] = vals

    return bucket_to_values


def bucket_to_value(bucket, sampler):
    """从桶里随机采样一个真实步数值。"""
    vals = sampler.get(bucket)
    if vals is None or len(vals) == 0:
        return 0
    return int(np.random.choice(vals))


# ================================================================
# 5. 主生成循环
# ================================================================

# 用户画像组合（用于条件生成）
PROFILE_COMBOS = [
    ('low', 'rare'), ('low', 'common'), ('low', 'frequent'),
    ('mid', 'rare'), ('mid', 'common'), ('mid', 'frequent'),
    ('high', 'rare'), ('high', 'common'), ('high', 'frequent'),
]


def generate_synthetic_dataset(model, tokenizer, n_users=30, n_days=43,
                                temperature=1.0, real_data=None, bin_edges=None,
                                profile_strategy='sample_from_real'):
    """生成完整的合成数据集。

    profile_strategy:
      'sample_from_real': 从真实数据的用户画像分布中采样（推荐）
      'uniform': 9 种画像组合均匀采样
    """
    all_rows = []
    total_generated = 0
    total_failed = 0

    # 准备桶 → 步数采样器
    sampler = None
    if real_data is not None and bin_edges is not None:
        sampler = build_bucket_to_value_sampler(real_data, bin_edges)

    # 准备用户画像分布（如果用 sample_from_real 策略）
    real_profiles = None
    if profile_strategy == 'sample_from_real' and real_data is not None:
        real_profiles = compute_user_profiles_from_real(real_data)

    for user_id in range(1, n_users + 1):
        # 给每个合成用户分配一个画像
        if profile_strategy == 'sample_from_real' and real_profiles:
            profile = random.choice(real_profiles)
        else:
            profile = random.choice(PROFILE_COMBOS)
        activity_level, zero_tendency = profile

        for day in range(1, n_days + 1):
            is_weekday = (day - 1) % 7 < 5

            # 随机 slot 数（匹配真实分布）
            n_slots = random.choices([3, 4, 5], weights=[0.1, 0.2, 0.7])[0]

            prompt = build_prompt(activity_level, zero_tendency, is_weekday,
                                  n_slots, tokenizer)

            # 生成
            text = generate_batch(model, tokenizer, [prompt],
                                  temperature=temperature, max_new_tokens=512)[0]

            visits = parse_trajectory(text)

            # 失败重试一次（用更低温度）
            if visits is None:
                text = generate_batch(model, tokenizer, [prompt],
                                      temperature=max(0.7, temperature - 0.2),
                                      max_new_tokens=512)[0]
                visits = parse_trajectory(text)

            if visits is None:
                total_failed += 1
                continue

            for v in visits:
                v['user_id'] = user_id
                v['study_day'] = day
                v['is_weekday'] = is_weekday
                v['activity_level'] = activity_level
                v['zero_tendency'] = zero_tendency

                # 桶 → 实际步数值
                if sampler is not None:
                    v['jbsteps30pre'] = bucket_to_value(
                        v['jbsteps30pre_bucket'], sampler
                    )

                all_rows.append(v)

            total_generated += 1

        if user_id % 5 == 0:
            print(f"  User {user_id}/{n_users}: "
                  f"{total_generated} days OK, {total_failed} failed")

    df = pd.DataFrame(all_rows)
    fail_rate = total_failed / max(1, total_generated + total_failed)
    print(f"\nGeneration complete: {len(df)} rows, "
          f"{total_generated} days OK, {total_failed} failed ({fail_rate*100:.1f}%)")

    return df


def compute_user_profiles_from_real(df):
    """从真实数据中提取每个 uid 的画像，用于采样合成用户分布。"""
    profiles = []
    for uid in df['uid'].unique():
        user_df = df[df['uid'] == uid]
        mean = user_df['jbsteps30pre'].mean()
        zero_rate = (user_df['jbsteps30pre'] == 0).mean()

        if mean < 150:
            act = "low"
        elif mean < 350:
            act = "mid"
        else:
            act = "high"

        if zero_rate < 0.20:
            zt = "rare"
        elif zero_rate < 0.40:
            zt = "common"
        else:
            zt = "frequent"

        profiles.append((act, zt))
    return profiles


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
    parser.add_argument('--output', default='synthetic_trajectories.csv')
    parser.add_argument('--real-data', default='../data/cleaned_output.csv',
                        help='真实数据路径（用于桶→步数采样和画像分布）')
    parser.add_argument('--bins-file', default='/data/steps_bins.json',
                        help='prepare_sft_data.py 保存的桶切分点')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 加载模型
    model, tokenizer = load_model(args.model, args.adapter)

    # 加载真实数据 + 桶切分点
    real_data = None
    bin_edges = None
    if args.real_data and os.path.exists(args.real_data):
        real_data = pd.read_csv(args.real_data)
        print(f"Loaded real data: {len(real_data)} rows")

    if args.bins_file and os.path.exists(args.bins_file):
        with open(args.bins_file) as f:
            bins_info = json.load(f)
        bin_edges = [float(e) if e != 'inf' else float('inf')
                     for e in bins_info['edges']]
        print(f"Loaded bin edges: {bin_edges}")

    # 生成
    df = generate_synthetic_dataset(
        model, tokenizer,
        n_users=args.n_users,
        n_days=args.n_days,
        temperature=args.temperature,
        real_data=real_data,
        bin_edges=bin_edges,
    )

    if len(df) == 0:
        print("⚠️ 生成数据为空！检查 SYSTEM_PROMPT / 正则 / 模型路径是否正确")
        return

    # 保存
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

    # 打印统计
    print(f"\n{'='*50}")
    print("Generated data statistics:")
    print(f"{'='*50}")
    print(f"Rows: {len(df)}")
    print(f"Users: {df['user_id'].nunique()}")
    print(f"\nLocation distribution (top 5):")
    print(df['location'].value_counts(normalize=True).head(5))
    print(f"\nActivity distribution:")
    print(df['activity'].value_counts(normalize=True))
    print(f"\nsteps_pre_bucket distribution:")
    print(df['jbsteps30pre_bucket'].value_counts(normalize=True).sort_index())
    if 'jbsteps30pre' in df.columns:
        print(f"\njbsteps30pre stats:")
        print(df['jbsteps30pre'].describe())
        print(f"Zero rate: {(df['jbsteps30pre']==0).mean():.3f}")


if __name__ == '__main__':
    main()
