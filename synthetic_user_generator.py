"""
synthetic_user_generator_v3_cleaned.py
======================================
基于 cleaned_output.csv 重写的合成用户生成器。

与 v3 原版相比的所有修改：
  【修改1】输入数据：suggestions.csv + users.csv → cleaned_output.csv + users.csv
  【修改2】列名映射：user.index→uid, sugg.select.slot→day_slot, 
           dec.temperature→temperature, recognized.activity→activity,
           dec.location.category→location, dec.weather.condition→weather
  【修改3】send 编码：True/False → 0/1/2 (0=no_send, 1=active, 2=sedentary)
  【修改4】location 分类：3类(home/work/other) → 16类(直接用cleaned的分类)
  【修改5】avail 率：硬编码(70%/83%) → 从数据按slot提取实际值
  【修改6】步数统计：global_mean=276 → 从cleaned计算=246.5
  【修改7】零值率：0.27 → 从cleaned计算=0.371
  【修改8】activity 名称：WALKING → ON_FOOT
  【修改9】发送概率：固定π_b=0.3 → 基于is_randomized的三值概率
  【修改10】TE 计算：send==True vs False → send>0 vs send==0
  【修改11】response 列：新增 good/bad/no_response/snoozed 信息用于观察记录
  【修改12】dosage 计算：适配 send=1 和 send=2 都算发送
  【修改13】create_observation：区分 active suggestion 和 sedentary suggestion
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
from collections import Counter
from pathlib import Path
import argparse
import warnings

from data_extractor import V1DataExtractor, generate_base_steps, generate_context, generate_baseline_vectors
from llm import Qwen3BLLM, SimulatedLLM
from memory_stream import Memory, create_observation, MemoryStream
from prompts import REFLECTION_GEN_PROMPT, MOTIVATION_PROMPT, HABIT_PROMPT, RECEPTIVITY_PROMPT, SYSTEM_SCORE, \
    REFLECTION_Q_PROMPT, IMPORTANCE_SYSTEM, IMPORTANCE_USER, PROMPT_ADJUSTMENT, SYS_ADJUSTMENT

warnings.filterwarnings('ignore')



# ================================================================
# 第六部分：单用户模拟（逐步串行，结构不变）
# ================================================================

def simulate_user(
    user_id: int, user_params: dict, ext: V1DataExtractor, llm,
    n_days: int = 42, slots_per_day: int = 5, n_samples: int = 3,
    output_dir: str = "./output", verbose: bool = True,
) -> pd.DataFrame:
    from trace_logger import TraceLogger
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Simulating Synthetic User {user_id} ({n_days} days × {slots_per_day} slots)")
    print(f"{'='*60}")
    
    stream = MemoryStream()
    persona = (f"Age {int(user_params['age'])}, {user_params['gender']}. "
               f"Self-efficacy {user_params['selfeff.intake']:.1f}/25, "
               f"conscientiousness {user_params['consc']:.1f}/30. "
               f"Baseline ~{int(user_params['predicted_mean_steps'])} steps/decision. "
               f"Initial treatment effect {user_params['predicted_te']:+.0f} steps.")
    for line in persona.strip().split('. '):
        if line.strip():
            stream.add(Memory("study_start", line.strip() + ".", "background", 7))
    
    logger = TraceLogger(output_dir, user_id=user_id, user_params=user_params)
    logger.log_user_init(user_params,
        {'motivation': 3, 'habit': 1, 'receptivity': 4}, persona)
    
    trajectory = []
    dosage = 0.0
    motivation_raw, habit_raw, receptivity_raw = [], [], []
    start_time = time.time()
    T = n_days * slots_per_day
    step_idx = 0
    
    for day in range(1, n_days + 1):
        for slot in range(1, slots_per_day + 1):
            step_idx += 1
            ctx = generate_context(ext, user_params, day, slot, trajectory)
            action = ctx.pop('action', 0) if 'action' in ctx else (
                # action 已在 generate_context 中通过 send_probs 决定
                0  # fallback
            )
            # 实际上 action 需要从 ctx 外部获取——修正：
            # generate_context 返回的 ctx 不包含 action，action 需要单独处理
            # 但在上面的 generate_context 中我们已经计算了 action 但没返回
            # 这里重新用 ctx 的信息计算
            
            # 重新计算 action（和 generate_context 中一致的逻辑）
            if ctx['avail'] and np.random.random() < ext.randomization_rate:
                send_vals = list(ext.send_probs.keys())
                send_ps = list(ext.send_probs.values())
                action = int(np.random.choice(send_vals, p=send_ps))
            else:
                action = 0
            if not ctx['avail']:
                action = 0
            
            # 生成步数
            if ctx['avail']:
                base = generate_base_steps(user_params, ctx, action, dosage, ext)
                obs_text = stream.format_memories(stream.get_recent(5, "observation"))
                # 【修改3】action_desc 区分 0/1/2
                action_desc = {0: "No", 1: "Active walking suggestion", 2: "Sedentary stand-up suggestion"}.get(action, "No")
                adj_prompt = PROMPT_ADJUSTMENT.format(
                    persona=persona[:200],
                    motivation=motivation_raw[-1] if motivation_raw else 3,
                    habit=habit_raw[-1] if habit_raw else 1,
                    receptivity=receptivity_raw[-1] if receptivity_raw else 4,
                    study_day=day, slot=slot, location=ctx['location'],
                    action_desc=action_desc, base_steps=base,
                    recent_obs=obs_text[:400] if obs_text else "No prior observations.")
                adj = llm.judge_adjustment(SYS_ADJUSTMENT, adj_prompt)
                adj = max(-50, min(100, adj))
                steps = max(0, int(base * (1 + adj / 100)))
            else:
                base, adj, steps, adj_prompt = 0, 0, 0, ""
            
            reward = np.log(steps + 0.5)
            # 【修改12】dosage: send=1 和 send=2 都累加
            dosage = 0.95 * dosage + (1 if action > 0 else 0)
            
            observation = create_observation(day, slot, ctx, action, steps)
            
            imp_prompt = IMPORTANCE_USER.format(observation=observation)
            imp_str = llm.score_importance(IMPORTANCE_SYSTEM, imp_prompt)
            try: importance = max(1, min(10, int(imp_str)))
            except: importance = 5
            
            stream.add(Memory(f"Day{day}_Slot{slot}", observation, "observation", importance))
            logger.log_observation(day, slot, observation, importance,
                                   stream.importance_since_reflection)
            
            # 反思触发（不变）
            if stream.should_reflect():
                recent = stream.get_recent(15)
                recent_text = stream.format_memories(recent)
                q_prompt_text = REFLECTION_Q_PROMPT.format(recent_memories=recent_text)
                q_response = llm.generate_text("You are a behavioral analysis system.", q_prompt_text)
                questions = [q.strip() for q in q_response.strip().split('\n') if q.strip()][:3]
                reflection_details = []
                for q in questions:
                    if len(q) < 5: continue
                    ref_gen_prompt = REFLECTION_GEN_PROMPT.format(question=q, relevant_memories=recent_text)
                    ref_text = llm.generate_text("You are a behavioral analysis system. Write concise inferences.", ref_gen_prompt)
                    stream.add(Memory(f"Day{day}_Slot{slot}", ref_text.strip(), "reflection", 8))
                    reflection_details.append({"question": q, "prompt": ref_gen_prompt[:500], "response": ref_text.strip()})
                stream.reset_reflection_counter()
                
                cur_state = {
                    'motivation': motivation_raw[-1] if motivation_raw else 3,
                    'habit': habit_raw[-1] if habit_raw else 1,
                    'receptivity': receptivity_raw[-1] if receptivity_raw else 4,
                }
                reflection_text = "\n".join(d['response'] for d in reflection_details)
                logger.log_reflection(
                    day=day,
                    reflection_text=reflection_text,
                    old_state=cur_state,
                    new_state=cur_state,  # 反思不直接改状态，状态在评分时更新
                    raw_llm_state=(cur_state['motivation'], cur_state['habit'], cur_state['receptivity']),
                    constrained_state=cur_state,
                    reflect_prompt=q_prompt_text,
                    recent_obs_text=recent_text[:600],
                    questions=[d['question'] for d in reflection_details],
                    inferences=[d['response'] for d in reflection_details],
                )
                if verbose: print(f"  [D{day}S{slot}] Reflection triggered ({len(questions)} questions)")
            
            # 评分（不变）
            recent_obs = stream.get_recent(10, "observation")
            recent_ref = stream.get_recent(3, "reflection")
            recent_obs_text = stream.format_memories(recent_obs)
            recent_ref_text = stream.format_memories(recent_ref) if recent_ref else "No reflections yet."
            same_slot_data = [t for t in trajectory if t['slot'] == slot][-7:]
            slot_pattern = ", ".join(f"D{t['study_day']}={t['jbsteps30']}" for t in same_slot_data) or "No data"
            if len(same_slot_data) >= 3:
                ss = [t['jbsteps30'] for t in same_slot_data]
                cv = np.std(ss) / (np.mean(ss) + 1)
                consistency_desc = f"CV={cv:.2f}"
            else:
                consistency_desc = "Too few data points"
            
            action_desc = {0: "No", 1: "Active suggestion", 2: "Sedentary suggestion"}.get(action, "No")
            score_prompts = []
            m_user = MOTIVATION_PROMPT.format(seed_memory=persona[:400], recent_reflections=recent_ref_text,
                recent_observations=recent_obs_text, study_day=day, slot=slot,
                pre30_steps=ctx['prior_30min_steps'], action_desc=action_desc, post_steps=steps)
            for _ in range(n_samples): score_prompts.append({"system": SYSTEM_SCORE, "user": m_user})
            h_user = HABIT_PROMPT.format(seed_memory=persona[:400], recent_observations=recent_obs_text,
                slot_pattern=slot_pattern, consistency_desc=consistency_desc, study_day=day)
            for _ in range(n_samples): score_prompts.append({"system": SYSTEM_SCORE, "user": h_user})
            r_user = RECEPTIVITY_PROMPT.format(seed_memory=persona[:400], recent_reflections=recent_ref_text,
                recent_observations=recent_obs_text, study_day=day, slot=slot, dosage=dosage)
            for _ in range(n_samples): score_prompts.append({"system": SYSTEM_SCORE, "user": r_user})
            
            all_scores = llm.batch_score(score_prompts)
            def parse_scores(raw):
                parsed = [max(1, min(5, int(s))) if s.isdigit() else 3 for s in raw]
                return Counter(parsed).most_common(1)[0][0]
            m_score = parse_scores(all_scores[0:n_samples])
            h_score = parse_scores(all_scores[n_samples:2*n_samples])
            r_score = parse_scores(all_scores[2*n_samples:3*n_samples])
            motivation_raw.append(m_score); habit_raw.append(h_score); receptivity_raw.append(r_score)
            
            logger.log_decision_point(
                day=day, slot=slot, ctx=ctx, action=action,
                base_steps=base, llm_adj_pct=adj, final_steps=steps,
                state={'motivation': m_score, 'habit': h_score, 'receptivity': r_score},
                dosage=dosage,
                prompt_text=adj_prompt if ctx['avail'] else "",
                llm_raw_output=str(adj),
                importance=importance,
                importance_acc=stream.importance_since_reflection,
                reflection_triggered=False,
            )
            
            # 【修改3】send 列记录 0/1/2
            trajectory.append(dict(
                user_id=user_id, study_day=day, slot=slot, avail=ctx['avail'],
                send=action, jbsteps30=steps, base_steps=base, llm_adj_pct=adj,
                jbsteps30pre=ctx['prior_30min_steps'], location=ctx['location'],
                temperature=ctx['temperature'], weather=ctx['weather'],
                dosage=round(dosage, 3), reward=round(reward, 4),
                motivation_raw=m_score, habit_raw=h_score, receptivity_raw=r_score,
                weekday=ctx['weekday'], activity=ctx['activity'],
                response=ctx.get('response', 'no_send'),
            ))
            
            if step_idx % 35 == 0 and verbose:
                elapsed = time.time() - start_time
                print(f"  [{step_idx}/{T}] M={m_score} H={h_score} R={r_score}, LLM calls={llm.call_count}, {elapsed:.0f}s")
    
    # 时序平滑
    def smooth(scores, alpha=0.7):
        result = [float(scores[0])]
        for i in range(1, len(scores)):
            result.append(alpha * scores[i] + (1 - alpha) * result[-1])
        return [max(1, min(5, round(s))) for s in result]
    
    df = pd.DataFrame(trajectory)
    df['motivation'] = smooth(motivation_raw)
    df['habit'] = smooth(habit_raw)
    df['receptivity'] = smooth(receptivity_raw)
    
    logger.finalize(
        final_state={'motivation': motivation_raw[-1] if motivation_raw else 3,
                     'habit': habit_raw[-1] if habit_raw else 1,
                     'receptivity': receptivity_raw[-1] if receptivity_raw else 4},
        summary_stats={'total_steps': len(df), 'llm_calls': llm.call_count,
                       'elapsed_seconds': round(time.time() - start_time, 1)}
    )
    memory_log = [m.to_dict() for m in stream.memories]
    with open(os.path.join(output_dir, f"user{user_id}_memory_log.json"), 'w', encoding='utf-8') as f:
        json.dump(memory_log, f, ensure_ascii=False, indent=2)
    
    total_time = time.time() - start_time
    print(f"\nUser {user_id} done: {len(df)} points, {llm.call_count} LLM calls, {total_time:.0f}s")
    return df


def main():
    users_csv = 'data/users.csv'
    cleaned_csv = 'data/cleaned_output.csv'

    ext = V1DataExtractor(users_csv, cleaned_csv)

    # llm = Qwen3BLLM()
    llm = SimulatedLLM()
    
    baseline_df = generate_baseline_vectors(ext, 10)
    
    params = baseline_df.iloc[0].to_dict()
    user_output_dir = os.path.join('data/outputs', "user1")
    df = simulate_user(user_id=1, user_params=params, ext=ext, llm=llm,
                       n_days=43, slots_per_day=5, n_samples=3,
                       output_dir=user_output_dir, verbose=True)
    df.to_csv(os.path.join(user_output_dir, "user1_trajectory.csv"), index=False)
    
    print(f"\nTotal LLM calls: {llm.call_count}")


if __name__ == "__main__":
    main()
