"""
generate_trajectories_vllm.py
==============================
vLLM 极速版 —— 用 PagedAttention + continuous batching。

相比 HF 单条版速度提升 10-30 倍。
1290 条 prompt 预计 1-2 分钟跑完。

依赖:
  pip install vllm

注意:
  - vLLM 加载 LoRA 需要：
    * 启动时 enable_lora=True
    * 用 SamplingParams.lora_request 或在 generate 时指定 LoRARequest
  - 推荐先 merge LoRA 到 base model，加载 merged 模型更省事
  - 你已经有 saves/heartsteps_qwen3_merged，直接用它

用法:
  python generate_trajectories_vllm.py \
      --model ../LlamaFactory/saves/heartsteps_qwen3_merged \
      --n-users 30 --n-days 43 \
      --output synthetic.csv
"""

import argparse
import json
import re
import os
import random
import time
import numpy as np
import pandas as pd

from vllm import LLM, SamplingParams

from sys_prompt import SYSTEM_PROMPT


def build_prompt(activity_level, zero_tendency, is_weekday, n_slots):
    day_type = "weekday" if is_weekday else "weekend"
    instruction = (f"Generate a {day_type} trajectory with {n_slots} decision points "
                   f"for a participant with {activity_level} activity level "
                   f"and {zero_tendency} zero-step periods.")
    return f"Human: {SYSTEM_PROMPT}\n{instruction}\nAssistant: "


# ================================================================
# 解析
# ================================================================

VISIT_PATTERN = re.compile(
    r'day_slot is (\d+)\s*,?\s*'
    r'is_weekday is (yes|no),\s*'
    r'in_trial is (yes|no),\s*'
    r'weather is (\w+),\s*'
    r'temperature is (\w+),\s*'
    r'location is (\w+),\s*'
    r'activity is (\w+),\s*'
    r'steps_pre is (steps_\w+?),\s*'
    r'send is (\w+),\s*'
    r'response is (\w+)',
    re.IGNORECASE,
)

STEPS_BUCKETS = [
    'steps_zero', 'steps_q1', 'steps_q2', 'steps_q3', 'steps_q4',
    'steps_q5', 'steps_q6', 'steps_q7', 'steps_q8', 'steps_high',
]



def parse_trajectory(text, debug=False):
    visits = []
    raw_matches = list(VISIT_PATTERN.finditer(text))

    if debug and not raw_matches:
        print(f"  [debug] 正则未匹配任何 visit")
        print(f"  [debug] text 前 200 字: {text[:200]!r}")


    for m in raw_matches:
        slot = int(m.group(1))
        if slot < 1 or slot > 5:
            continue
        if m.group(8) not in STEPS_BUCKETS:
            continue
        visits.append({
            'day_slot': slot,
            # 'is_weekday_str': m.group(2).lower() == 'yes',
            'is_randomized': m.group(3).lower() == 'yes',
            'weather': m.group(4),
            'temperature': m.group(5),
            'location': m.group(6),
            'activity': m.group(7),
            'jbsteps30pre_bucket': m.group(8),
            'send': m.group(9),
            'response': m.group(10),
        })
    if debug:
        print(f"  [debug] 匹配 {len(raw_matches)} 个，通过 {len(visits)} 个")

    if len(visits) < 3:
        return None
    visits.sort(key=lambda v: v['day_slot'])
    seen = set()
    unique = []
    for v in visits:
        if v['day_slot'] not in seen:
            seen.add(v['day_slot'])
            unique.append(v)
    return unique if len(unique) >= 3 else None


def build_bucket_to_value_sampler(real_data, bin_edges):
    sampler = {}
    real_steps = real_data['jbsteps30pre'].values
    for i, label in enumerate(STEPS_BUCKETS):
        if label == 'steps_zero':
            vals = real_steps[real_steps == 0]
        elif label == 'steps_high':
            vals = real_steps[real_steps > bin_edges[-2]]
        else:
            lo, hi = bin_edges[i], bin_edges[i + 1]
            vals = real_steps[(real_steps > lo) & (real_steps <= hi)]
        if len(vals) == 0:
            vals = np.array([0 if label == 'steps_zero' else 100])
        sampler[label] = vals
    return sampler


def bucket_to_value(bucket, sampler):
    vals = sampler.get(bucket)
    return int(np.random.choice(vals)) if vals is not None and len(vals) > 0 else 0


def compute_user_profiles_from_real(df):
    profiles = []
    for uid in df['uid'].unique():
        sub = df[df['uid'] == uid]
        mean = sub['jbsteps30pre'].mean()
        zero_rate = (sub['jbsteps30pre'] == 0).mean()
        act = "low" if mean < 150 else "mid" if mean < 350 else "high"
        zt = "rare" if zero_rate < 0.20 else "common" if zero_rate < 0.40 else "frequent"
        profiles.append((act, zt))
    return profiles


# ================================================================
# 主函数
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='../LlamaFactory/saves/heartsteps_qwen3_merged',
                        help='vLLM 推荐用 merged 模型（不带 adapter）')
    parser.add_argument('--n-users', type=int, default=30)
    parser.add_argument('--n-days', type=int, default=43)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--output', default='synthetic_trajectories.csv')
    parser.add_argument('--real-data', default='../data/cleaned_output.csv')
    parser.add_argument('--bins-file', default='./data/steps_bins.json')
    parser.add_argument('--gpu-mem-frac', type=float, default=0.9,
                        help='vLLM 占用显存比例（0-1）')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ================ 加载 vLLM ================
    print(f"Loading vLLM model from {args.model}...")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_frac,
        max_model_len=2048,
        seed=args.seed,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # ================ 加载真实数据 ================
    real_data = pd.read_csv(args.real_data) if os.path.exists(args.real_data) else None
    bin_edges = None
    if args.bins_file and os.path.exists(args.bins_file):
        with open(args.bins_file) as f:
            bin_edges = [float(e) if e != 'inf' else float('inf')
                         for e in json.load(f)['edges']]
        print(f"Loaded bin edges: {bin_edges}")

    sampler = build_bucket_to_value_sampler(real_data, bin_edges) if (real_data is not None and bin_edges) else None
    real_profiles = compute_user_profiles_from_real(real_data) if real_data is not None else None

    # ================ 准备所有任务 ================
    tasks = []
    for user_id in range(1, args.n_users + 1):
        profile = random.choice(real_profiles) if real_profiles else ('mid', 'common')
        for day in range(1, args.n_days + 1):
            is_weekday = (day - 1) % 7 < 5
            n_slots = random.choices([3, 4, 5], weights=[0.1, 0.2, 0.7])[0]
            tasks.append({
                'user_id': user_id,
                'study_day': day,
                'is_weekday': is_weekday,
                'n_slots': n_slots,
                'activity_level': profile[0],
                'zero_tendency': profile[1],
                'prompt': build_prompt(profile[0], profile[1], is_weekday, n_slots),
            })

    print(f"\n准备了 {len(tasks)} 个任务，开始生成...")

    # ================ 一次性扔给 vLLM ================
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    t1 = time.time()
    outputs = llm.generate([t['prompt'] for t in tasks], sampling)
    t_gen = time.time() - t1
    print(f"vLLM 生成完成，耗时 {t_gen:.1f}s ({len(tasks)/t_gen:.1f} tasks/s)")

    # ================ 解析 ================
    all_rows = []
    failed_tasks = []
    for task, out in zip(tasks, outputs):
        text = out.outputs[0].text
        visits = parse_trajectory(text)
        if visits is None:
            failed_tasks.append(task)
            continue
        for v in visits:
            v.update({
                'user_id': task['user_id'],
                'study_day': task['study_day'],
                'is_weekday': task['is_weekday'],
                'activity_level': task['activity_level'],
                'zero_tendency': task['zero_tendency'],
            })
            if sampler:
                v['jbsteps30pre'] = bucket_to_value(v['jbsteps30pre_bucket'], sampler)
            all_rows.append(v)

    # ================ 多轮重试（温度递降） ================
    RETRY_TEMPS = [0.8, 0.7, 0.5]
    for retry_round, retry_temp in enumerate(RETRY_TEMPS, 1):
        if not failed_tasks:
            break
        print(f"\n重试 {retry_round}/{len(RETRY_TEMPS)}: "
              f"{len(failed_tasks)} 个任务, temperature={retry_temp}")

        retry_sampling = SamplingParams(
            temperature=retry_temp,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        outputs = llm.generate([t['prompt'] for t in failed_tasks], retry_sampling)

        still_failed = []
        for task, out in zip(failed_tasks, outputs):
            visits = parse_trajectory(out.outputs[0].text)
            if visits is None:
                still_failed.append(task)
                continue
            for v in visits:
                v.update({
                    'user_id': task['user_id'],
                    'study_day': task['study_day'],
                    'is_weekday': task['is_weekday'],
                    'activity_level': task['activity_level'],
                    'zero_tendency': task['zero_tendency'],
                })
                if sampler:
                    v['jbsteps30pre'] = bucket_to_value(v['jbsteps30pre_bucket'], sampler)
                all_rows.append(v)

        recovered = len(failed_tasks) - len(still_failed)
        print(f"  恢复 {recovered} 个, 剩余 {len(still_failed)} 个")
        failed_tasks = still_failed

    if failed_tasks:
        print(f"\n⚠️ 最终 {len(failed_tasks)} 个任务无法解析（已丢弃）")

    # # ================ 失败重试（更低温度） ================
    # if failed_tasks:
    #     print(f"\n重试 {len(failed_tasks)} 个失败任务...")
    #     retry_sampling = SamplingParams(
    #         temperature=max(0.7, args.temperature - 0.2),
    #         top_p=args.top_p,
    #         max_tokens=args.max_tokens,
    #     )
    #     outputs = llm.generate([t['prompt'] for t in failed_tasks], retry_sampling)
    #     for task, out in zip(failed_tasks, outputs):
    #         visits = parse_trajectory(out.outputs[0].text)
    #         if visits is None:
    #             continue
    #         for v in visits:
    #             v.update({
    #                 'user_id': task['user_id'],
    #                 'study_day': task['study_day'],
    #                 'is_weekday': task['is_weekday'],
    #                 'activity_level': task['activity_level'],
    #                 'zero_tendency': task['zero_tendency'],
    #             })
    #             if sampler:
    #                 v['jbsteps30pre'] = bucket_to_value(v['jbsteps30pre_bucket'], sampler)
    #             all_rows.append(v)

    df = pd.DataFrame(all_rows)
    ok_days = df.groupby(['user_id', 'study_day']).ngroups if len(df) > 0 else 0
    fail_days = len(tasks) - ok_days
    print(f"\n=== Done in {time.time()-t0:.1f}s total ===")
    print(f"OK days: {ok_days}/{len(tasks)}, Failed: {fail_days} ({fail_days/len(tasks)*100:.1f}%)")

    if len(df) == 0:
        print("⚠️ 生成数据为空！")
        return

    df['send'] = df['send'].map({'no_send': 0, 'non_activity': 2, 'activity': 1})
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")

    print(f"\nlocation: {df['location'].value_counts(normalize=True).head(5).to_dict()}")
    print(f"steps_bucket: {dict(df['jbsteps30pre_bucket'].value_counts(normalize=True).sort_index())}")
    if 'jbsteps30pre' in df.columns:
        print(f"jbsteps30pre: mean={df['jbsteps30pre'].mean():.0f}, "
              f"zero_rate={(df['jbsteps30pre']==0).mean():.3f}")


if __name__ == '__main__':
    main()
