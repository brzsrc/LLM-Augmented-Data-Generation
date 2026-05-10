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


import re

def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks and any stray tags from LLM output."""
    # 移除完整的 <think>...</think> 块（包括跨行）
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 移除残留的孤立标签
    text = text.replace('<think>', '').replace('</think>', '')
    return text.strip()


def _parse_questions(raw: str, max_n: int = 3) -> List[str]:
    """
    Robustly parse a list of questions out of LLM output.
    Handles: leading </think> tags, blank lines, numbered list prefixes
    like '1.', '1)', '- ', '* '. Only keeps lines that look like real questions.
    """
    cleaned = _strip_think_tags(raw)
    questions = []
    for line in cleaned.split('\n'):
        line = line.strip()
        if not line:
            continue
        # 去掉行首的列表序号: "1.", "1)", "1:", "- ", "* ", "Q1:", etc.
        line = re.sub(r'^\s*(?:Q\s*\d+[.):]?|\d+[.):]|[-*•])\s*', '', line, flags=re.IGNORECASE).strip()
        if not line:
            continue
        # 只保留像问题的行：有一定长度，且形式上像一句话/问句
        # 排除明显不是问题的残留标签或元注释
        if len(line) < 15:
            continue
        if line.lower().startswith(('<think', '</think', 'note:', 'here are', 'list ')):
            continue
        questions.append(line)
        if len(questions) >= max_n:
            break
    return questions


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
            action = ctx['action']
            
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
                questions = _parse_questions(q_response, max_n=3)
                reflection_details = []
                for q in questions:
                    ref_gen_prompt = REFLECTION_GEN_PROMPT.format(question=q, relevant_memories=recent_text)
                    ref_text = llm.generate_text("You are a behavioral analysis system. Write concise inferences.", ref_gen_prompt)
                    ref_text_clean = _strip_think_tags(ref_text)
                    stream.add(Memory(f"Day{day}_Slot{slot}", ref_text_clean, "reflection", 8))
                    reflection_details.append({"question": q, "prompt": ref_gen_prompt, "response": ref_text_clean})
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


# ================================================================
# 第八部分：跨用户并行批量推理
# ================================================================

class UserState:
    """一个用户的全部运行时状态"""
    def __init__(self, uid: int, params: dict):
        self.uid = uid
        self.params = params
        self.stream = MemoryStream()
        self.trajectory = []
        self.dosage = 0.0
        self.motivation_raw = []
        self.habit_raw = []
        self.receptivity_raw = []
        
        # 初始心理状态
        selfeff_z = (params.get('selfeff.intake', 14.5) - 14.5) / 3.3
        self.last_m = int(np.clip(3 + selfeff_z, 2, 5))
        self.last_h = 1
        self.last_r = 4
        
        self.persona = (f"Age {int(params['age'])}, {params['gender']}. "
                        f"Self-efficacy {params['selfeff.intake']:.1f}/25, "
                        f"conscientiousness {params['consc']:.1f}/30. "
                        f"Baseline ~{int(params['predicted_mean_steps'])} steps/decision. "
                        f"Initial treatment effect {params['predicted_te']:+.0f} steps.")
        
        for line in self.persona.strip().split('. '):
            if line.strip():
                self.stream.add(Memory("study_start", line.strip() + ".", "background", 7))


def simulate_parallel(
    baseline_df: pd.DataFrame,
    ext: V1DataExtractor,
    llm,
    n_days: int = 42,
    n_samples: int = 3,
    output_dir: str = "./output",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    N 个用户同步推进，每一步把所有用户的 LLM 调用合成一个大 batch。
    
    每个 (day, slot) 分 6 个 Phase:
      A: 程序化生成上下文 + 基础步数（无 LLM）
      B: 批量步数调节 → N 个 prompt 一次提交
      C: 批量重要性评分 → N 个 prompt 一次提交
      D: 写入记忆流 + 收集需要反思的用户
      E: 批量反思（问题 + 推断）
      F: 批量评分 → N × 3 × n_samples 个 prompt 一次提交
    """
    os.makedirs(output_dir, exist_ok=True)
    N = len(baseline_df)
    
    print(f"\n{'='*60}")
    print(f"Parallel simulation: {N} users × {n_days} days × 5 slots")
    print(f"Estimated batch sizes: adj={N}, imp={N}, score={N*3*n_samples}")
    print(f"{'='*60}")
    
    # 初始化所有用户
    users = []
    for i, row in baseline_df.iterrows():
        users.append(UserState(uid=i+1, params=row.to_dict()))
    
    start_time = time.time()
    
    for day in range(1, n_days + 1):
        for slot in range(1, 6):
            
            # ── Phase A: 程序化（无 LLM） ──
            all_ctx = []
            all_action = []
            all_base = []
            
            for u in users:
                ctx = generate_context(ext, u.params, day, slot, u.trajectory)
                action = ctx['action']
                
                base = generate_base_steps(u.params, ctx, action, u.dosage, ext) if ctx['avail'] else 0
                all_ctx.append(ctx)
                all_action.append(action)
                all_base.append(base)
            
            # ── Phase B: 批量步数调节 ──
            adj_prompts = []
            adj_user_idx = []
            
            for i, u in enumerate(users):
                if all_ctx[i]['avail'] and all_base[i] > 0:
                    obs_text = u.stream.format_memories(u.stream.get_recent(5, "observation"))
                    action_desc = {0: "No", 1: "Active walking suggestion", 2: "Sedentary stand-up suggestion"}.get(all_action[i], "No")
                    p = PROMPT_ADJUSTMENT.format(
                        persona=u.persona[:200],
                        motivation=u.last_m, habit=u.last_h, receptivity=u.last_r,
                        study_day=day, slot=slot, location=all_ctx[i]['location'],
                        action_desc=action_desc, base_steps=all_base[i],
                        recent_obs=obs_text[:400] if obs_text else "No prior observations.")
                    adj_prompts.append({"system": SYS_ADJUSTMENT, "user": p})
                    adj_user_idx.append(i)
            
            adj_results = llm.batch_adjustment(adj_prompts) if adj_prompts else []
            
            # 计算最终步数
            all_steps = [0] * N
            all_adj = [0] * N
            for j, i in enumerate(adj_user_idx):
                adj = max(-50, min(100, adj_results[j]))
                all_adj[i] = adj
                all_steps[i] = max(0, int(all_base[i] * (1 + adj / 100)))
            
            # ── Phase C: 批量重要性评分 ──
            all_obs = []
            for i, u in enumerate(users):
                obs = create_observation(day, slot, all_ctx[i], all_action[i], all_steps[i])
                all_obs.append(obs)
            
            imp_prompts = [{"system": IMPORTANCE_SYSTEM,
                            "user": IMPORTANCE_USER.format(observation=obs)} for obs in all_obs]
            imp_results = llm.batch_importance(imp_prompts)
            
            all_imp = []
            for r in imp_results:
                try: all_imp.append(max(1, min(10, int(r))))
                except: all_imp.append(5)
            
            # ── Phase D: 写入记忆流 + 收集反思用户 ──
            reflect_users = []
            
            for i, u in enumerate(users):
                u.stream.add(Memory(f"Day{day}_S{slot}", all_obs[i], "observation", all_imp[i]))
                u.dosage = 0.95 * u.dosage + (1 if all_action[i] > 0 else 0)
                
                if u.stream.should_reflect():
                    reflect_users.append(i)
            
            # ── Phase E: 批量反思 ──
            if reflect_users:
                # Step 1: 批量生成问题
                q_prompts = []
                for i in reflect_users:
                    recent = users[i].stream.get_recent(15)
                    recent_text = users[i].stream.format_memories(recent)
                    q_prompts.append({
                        "system": "You are a behavioral analysis system.",
                        "user": REFLECTION_Q_PROMPT.format(recent_memories=recent_text)
                    })
                q_responses = llm.batch_text(q_prompts)
                
                # Step 2: 批量推断
                inf_prompts = []
                inf_map = []  # (reflect_local_idx, user_idx, question)
                
                for j, i in enumerate(reflect_users):
                    recent_text = users[i].stream.format_memories(users[i].stream.get_recent(15))
                    questions = _parse_questions(q_responses[j], max_n=3)
                    for q in questions:
                        inf_prompts.append({
                            "system": "You are a behavioral analysis system. Write concise inferences.",
                            "user": REFLECTION_GEN_PROMPT.format(question=q, relevant_memories=recent_text)
                        })
                        inf_map.append((j, i, q))
                
                inf_responses = llm.batch_text(inf_prompts) if inf_prompts else []
                
                # 写入记忆流
                for k, (j, i, q) in enumerate(inf_map):
                    resp = inf_responses[k] if k < len(inf_responses) else ""
                    # 清洗 think 标签（包括完整 <think>...</think> 块）
                    resp = _strip_think_tags(resp)
                    users[i].stream.add(Memory(f"Day{day}_S{slot}", resp, "reflection", 8))
                
                for i in reflect_users:
                    users[i].stream.reset_reflection_counter()
            
            # ── Phase F: 批量评分 ──
            score_prompts = []
            
            for i, u in enumerate(users):
                recent_obs = u.stream.get_recent(10, "observation")
                recent_ref = u.stream.get_recent(3, "reflection")
                obs_text = u.stream.format_memories(recent_obs)
                ref_text = u.stream.format_memories(recent_ref) if recent_ref else "No reflections yet."
                
                same_slot_data = [t for t in u.trajectory if t['slot'] == slot][-7:]
                slot_pattern = ", ".join(f"D{t['study_day']}={t['jbsteps30']}" for t in same_slot_data) or "No data"
                if len(same_slot_data) >= 3:
                    ss = [t['jbsteps30'] for t in same_slot_data]
                    cv_val = np.std(ss) / (np.mean(ss) + 1)
                    con = f"CV={cv_val:.2f}"
                else:
                    con = "Too few data points"
                
                action_desc = {0: "No", 1: "Active suggestion", 2: "Sedentary suggestion"}.get(all_action[i], "No")
                
                m_user = MOTIVATION_PROMPT.format(
                    seed_memory=u.persona[:400], recent_reflections=ref_text,
                    recent_observations=obs_text, study_day=day, slot=slot,
                    pre30_steps=all_ctx[i]['prior_30min_steps'],
                    action_desc=action_desc, post_steps=all_steps[i])
                h_user = HABIT_PROMPT.format(
                    seed_memory=u.persona[:400], recent_observations=obs_text,
                    slot_pattern=slot_pattern, consistency_desc=con, study_day=day)
                r_user = RECEPTIVITY_PROMPT.format(
                    seed_memory=u.persona[:400], recent_reflections=ref_text,
                    recent_observations=obs_text, study_day=day, slot=slot, dosage=u.dosage)
                
                for _ in range(n_samples):
                    score_prompts.append({"system": SYSTEM_SCORE, "user": m_user})
                for _ in range(n_samples):
                    score_prompts.append({"system": SYSTEM_SCORE, "user": h_user})
                for _ in range(n_samples):
                    score_prompts.append({"system": SYSTEM_SCORE, "user": r_user})
            
            all_scores = llm.batch_score(score_prompts)
            
            # 解析评分
            def parse_scores(raw):
                parsed = [max(1, min(5, int(s))) if s.isdigit() else 3 for s in raw]
                return Counter(parsed).most_common(1)[0][0]
            
            cursor = 0
            for i, u in enumerate(users):
                m_s = all_scores[cursor:cursor+n_samples]; cursor += n_samples
                h_s = all_scores[cursor:cursor+n_samples]; cursor += n_samples
                r_s = all_scores[cursor:cursor+n_samples]; cursor += n_samples
                
                m, h, r = parse_scores(m_s), parse_scores(h_s), parse_scores(r_s)
                u.last_m, u.last_h, u.last_r = m, h, r
                u.motivation_raw.append(m)
                u.habit_raw.append(h)
                u.receptivity_raw.append(r)
                
                reward = np.log(all_steps[i] + 0.5)
                u.trajectory.append(dict(
                    user_id=u.uid, study_day=day, slot=slot, avail=all_ctx[i]['avail'],
                    send=all_action[i], jbsteps30=all_steps[i], base_steps=all_base[i],
                    llm_adj_pct=all_adj[i], jbsteps30pre=all_ctx[i]['prior_30min_steps'],
                    location=all_ctx[i]['location'], temperature=all_ctx[i]['temperature'],
                    dosage=round(u.dosage, 3), reward=round(reward, 4),
                    motivation_raw=m, habit_raw=h, receptivity_raw=r,
                    weekday=all_ctx[i]['weekday'], activity=all_ctx[i]['activity'],
                    response=all_ctx[i].get('response', 'no_send'),
                ))
        
        # 每周打印进度
        if verbose and day % 7 == 0:
            elapsed = time.time() - start_time
            avg_steps = np.mean([t['jbsteps30'] for u in users for t in u.trajectory[-35:]])
            print(f"  Week {day//7}: avg_steps={avg_steps:.0f}, "
                  f"reflections_this_slot={len(reflect_users)}, "
                  f"LLM calls={llm.call_count}, {elapsed:.0f}s")
    
    # ── 后处理：平滑 + 保存 ──
    def smooth(scores, alpha=0.7):
        result = [float(scores[0])]
        for i in range(1, len(scores)):
            result.append(alpha * scores[i] + (1 - alpha) * result[-1])
        return [max(1, min(5, round(s))) for s in result]
    
    all_dfs = []
    for u in users:
        df = pd.DataFrame(u.trajectory)
        df['motivation'] = smooth(u.motivation_raw)
        df['habit'] = smooth(u.habit_raw)
        df['receptivity'] = smooth(u.receptivity_raw)
        all_dfs.append(df)
        
        # # 保存单用户
        # udir = os.path.join(output_dir, f"user{u.uid}")
        # os.makedirs(udir, exist_ok=True)
        # df.to_csv(os.path.join(udir, f"user{u.uid}_trajectory.csv"), index=False)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(os.path.join(output_dir, "all_trajectories.csv"), index=False)
    
    total = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Parallel simulation complete: {N} users, {n_days} days")
    print(f"Total time: {total:.0f}s ({total/60:.1f} min)")
    print(f"Total LLM calls: {llm.call_count}")
    print(f"Output: {output_dir}/all_trajectories.csv")
    print(f"{'='*60}")
    
    return combined


def main():
    users_csv = 'data/users.csv'
    cleaned_csv = 'data/cleaned_output.csv'

    ext = V1DataExtractor(users_csv, cleaned_csv)

    # llm = Qwen3BLLM()
    llm = SimulatedLLM()
    
    n_users = 100
    n_days = 42
    n_samples = 3
    
    baseline_df = generate_baseline_vectors(ext, n_users)
    
    if n_users == 1:
        # 单用户：逐步串行，有完整 trace logger
        params = baseline_df.iloc[0].to_dict()
        user_output_dir = os.path.join('data/outputs', "user1")
        df = simulate_user(user_id=1, user_params=params, ext=ext, llm=llm,
                           n_days=n_days, slots_per_day=5, n_samples=n_samples,
                           output_dir=user_output_dir, verbose=True)
        df.to_csv(os.path.join(user_output_dir, "user1_trajectory.csv"), index=False)
    else:
        # 多用户：跨用户并行批量推理
        df = simulate_parallel(
            baseline_df=baseline_df,
            ext=ext, llm=llm,
            n_days=n_days, n_samples=n_samples,
            output_dir='data/outputs', verbose=True)
    
    print(f"\nTotal LLM calls: {llm.call_count}")


if __name__ == "__main__":
    main()
