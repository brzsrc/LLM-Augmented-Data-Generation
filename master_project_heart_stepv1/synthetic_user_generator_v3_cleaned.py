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
warnings.filterwarnings('ignore')


# ================================================================
# 第一部分：从 cleaned_output.csv + users.csv 提取参数
# ================================================================

class V1DataExtractor:
    """
    【修改1】输入从 suggestions.csv 改为 cleaned_output.csv
    【修改2】所有列名适配 cleaned 格式
    """
    def __init__(self, users_path: str, cleaned_path: str):
        self.users = pd.read_csv(users_path)
        self.sugg = pd.read_csv(cleaned_path)  # 【修改1】
        self._compute()
    
    def _compute(self):
        s, u = self.sugg, self.users
        
        # Cholesky 基线参数（从 users.csv，不变）
        self.baseline_cols = ['selfeff.intake', 'consc', 'age']
        data = u[self.baseline_cols].dropna()
        self.baseline_mu = data.mean().values
        self.baseline_sigma = data.cov().values
        eig = np.min(np.linalg.eigvalsh(self.baseline_sigma))
        if eig < 1e-6:
            self.baseline_sigma += (abs(eig) + 1e-5) * np.eye(len(self.baseline_cols))
        
        # 回归模型
        from sklearn.linear_model import LinearRegression
        # 【修改2】user.index → uid
        user_stats = s.groupby('uid').agg(
            mean_steps=('jbsteps30', 'mean'),
            std_steps=('jbsteps30', 'std')
        ).reset_index()
        user_stats['cv'] = user_stats['std_steps'] / (user_stats['mean_steps'] + 1)
        
        # 【修改5】avail 率按 slot 从数据提取
        self.avail_by_slot = {}
        for slot in [1, 2, 3, 4, 5]:
            self.avail_by_slot[slot] = s[s['day_slot'] == slot]['avail'].mean()
        
        # 【修改10】TE 计算：send>0 vs send==0
        avail = s[s['avail'] == True]
        te_list = []
        for uid in avail['uid'].unique():
            ud = avail[avail['uid'] == uid]
            sent_mean = ud[ud['send'] > 0]['jbsteps30'].mean()    # 【修改3】send>0
            nosent_mean = ud[ud['send'] == 0]['jbsteps30'].mean()
            te = sent_mean - nosent_mean if not np.isnan(sent_mean - nosent_mean) else 0
            te_list.append({'uid': uid, 'te': te})
        te_df = pd.DataFrame(te_list)
        
        # 合并 users 和 stats（uid↔user.index 映射）
        user_stats_renamed = user_stats.rename(columns={'uid': 'user.index'})
        te_df_renamed = te_df.rename(columns={'uid': 'user.index'})
        merged = u.merge(user_stats_renamed, on='user.index').merge(te_df_renamed, on='user.index')
        valid = merged.dropna(subset=self.baseline_cols + ['mean_steps'])
        X = valid[self.baseline_cols].values
        self.model_steps = LinearRegression().fit(X, valid['mean_steps'])
        self.resid_std_steps = np.std(valid['mean_steps'] - self.model_steps.predict(X))
        self.model_te = LinearRegression().fit(X, valid['te'])
        self.resid_std_te = np.std(valid['te'] - self.model_te.predict(X))
        
        # 【修改2】slot 列名：sugg.select.slot → day_slot
        self.slot_stats = {
            slot: {
                'mean': s[s['day_slot'] == slot]['jbsteps30'].mean(),
                'zero_rate': (s[s['day_slot'] == slot]['jbsteps30'] == 0).mean(),
            }
            for slot in [1, 2, 3, 4, 5]
        }
        # 【修改6】
        self.global_mean_steps = s['jbsteps30'].mean()  # 246.5 而非 276
        # 【修改7】
        self.global_zero_rate = (s['jbsteps30'] == 0).mean()  # 0.371 而非 0.27
        
        # 【修改4+14】按 cluster 提取 location 分布
        # 5 个 cluster 的用户分组（从聚类分析得到）
        self.cluster_uids = {
            0: [1,2,5,6,8,9,10,11,13,15,16,18,20,21,22,24,25,29,30,31,33,34,35,36,37],  # 典型上班族
            1: [4,14,26,27,28,32],        # 社区活动型
            2: [3],                        # 运动爱好者
            3: [7,17],                     # 宅家晚活族
            4: [12,19,23],                 # 电子店员工
        }
        self.cluster_weights = {0: 25, 1: 6, 2: 1, 3: 2, 4: 3}  # 按人数比例
        self.cluster_names = {
            0: "office_worker", 1: "community_active", 2: "fitness_enthusiast",
            3: "homebody_night_active", 4: "retail_worker",
        }
        
        s['datetime_parsed'] = pd.to_datetime(s['datetime'])
        s['is_weekday'] = s['datetime_parsed'].dt.dayofweek < 5
        
        # 为每个 cluster × weekday/weekend × slot 计算 location 分布
        self.cluster_location_dist = {}  # {cluster_id: {(weekday_bool, slot): {loc: prob}}}
        for cl_id, cl_uids in self.cluster_uids.items():
            cl_data = s[s['uid'].isin(cl_uids)]
            self.cluster_location_dist[cl_id] = {}
            for is_wd in [True, False]:
                for slot in [1, 2, 3, 4, 5]:
                    subset = cl_data[(cl_data['is_weekday'] == is_wd) & (cl_data['day_slot'] == slot)]
                    if len(subset) > 0:
                        dist = subset['location'].value_counts(normalize=True).to_dict()
                    else:
                        # fallback 到全局分布
                        fallback = s[s['day_slot'] == slot]['location'].value_counts(normalize=True).to_dict()
                        dist = fallback
                    self.cluster_location_dist[cl_id][(is_wd, slot)] = dist
        
        # 保留全局 location_by_slot 作为 fallback
        self.location_categories = sorted(s['location'].dropna().unique().tolist())
        self.location_by_slot = {}
        for slot in [1, 2, 3, 4, 5]:
            slot_data = s[s['day_slot'] == slot]
            self.location_by_slot[slot] = slot_data['location'].value_counts(normalize=True).to_dict()
        
        # 按 cluster 的 avail 率
        self.cluster_avail_by_slot = {}
        for cl_id, cl_uids in self.cluster_uids.items():
            cl_data = s[s['uid'].isin(cl_uids)]
            self.cluster_avail_by_slot[cl_id] = {}
            for slot in [1, 2, 3, 4, 5]:
                subset = cl_data[cl_data['day_slot'] == slot]
                self.cluster_avail_by_slot[cl_id][slot] = subset['avail'].mean() if len(subset) > 0 else 0.87
        
        # 【修改8】activity 名称用 cleaned 的
        self.activity_probs = s['activity'].value_counts(normalize=True).to_dict()
        
        # 【修改9】发送概率：基于 is_randomized 的三值分布
        rand_data = s[s['is_randomized'] == True]
        self.send_probs = rand_data['send'].value_counts(normalize=True).to_dict()
        # {0: 0.124, 1: 0.489, 2: 0.386}
        self.randomization_rate = s['is_randomized'].mean()  # ~59%
        
        # 【修改11】response 分布
        sent_data = s[s['send'] > 0]
        self.response_probs = sent_data['response'].value_counts(normalize=True).to_dict()
        
        # 按 location 的平均步数（用于步数生成）
        self.steps_by_location = s.groupby('location')['jbsteps30'].mean().to_dict()
        
        self.gender_probs = u['gender'].value_counts(normalize=True).to_dict()
        
        print(f"V1 extracted from cleaned data: {len(u)} users, {len(s)} decisions")
        print(f"  global_mean_steps={self.global_mean_steps:.1f}, global_zero_rate={self.global_zero_rate:.1%}")
        print(f"  send_probs={self.send_probs}")
        print(f"  location_categories: {len(self.location_categories)} types")


def generate_baseline_vectors(ext: V1DataExtractor, n: int) -> pd.DataFrame:
    """Cholesky生成基线 + 按比例分配cluster"""
    L = np.linalg.cholesky(ext.baseline_sigma)
    z = np.random.standard_normal((n, len(ext.baseline_cols)))
    bl = ext.baseline_mu + z @ L.T
    alpha = np.random.uniform(-2, 2, len(ext.baseline_cols))
    delta = alpha / np.sqrt(1 + alpha @ alpha)
    for i in range(n):
        u, v = np.random.standard_normal(len(ext.baseline_cols)), np.random.standard_normal(len(ext.baseline_cols))
        bl[i] += (u > 0).astype(float) * (delta * np.abs(v))
    df = pd.DataFrame(bl, columns=ext.baseline_cols)
    df['selfeff.intake'] = df['selfeff.intake'].clip(5, 25)
    df['consc'] = df['consc'].clip(12, 30)
    df['age'] = df['age'].clip(19, 65).round().astype(int)
    gv, gp = list(ext.gender_probs.keys()), list(ext.gender_probs.values())
    df['gender'] = np.random.choice(gv, n, p=gp)
    df['user_id'] = range(1, n + 1)
    X = df[ext.baseline_cols].values
    df['predicted_mean_steps'] = (ext.model_steps.predict(X) + np.random.normal(0, ext.resid_std_steps * 0.5, n)).clip(50, 800)
    df['predicted_te'] = (ext.model_te.predict(X) + np.random.normal(0, ext.resid_std_te * 0.5, n)).clip(-200, 300)
    
    # 【修改14】按 25:6:1:2:3 的比例分配 cluster
    cluster_ids = list(ext.cluster_weights.keys())
    cluster_counts = list(ext.cluster_weights.values())
    cluster_probs = [c / sum(cluster_counts) for c in cluster_counts]
    df['cluster'] = np.random.choice(cluster_ids, n, p=cluster_probs)
    df['cluster_name'] = df['cluster'].map(ext.cluster_names)
    
    print(f"  Cluster分配: {dict(Counter(df['cluster'].values))}")
    print(f"  名称: {dict(df.groupby('cluster')['cluster_name'].first())}")
    
    return df


# ================================================================
# 第二部分：记忆流（不变）
# ================================================================

class Memory:
    def __init__(self, timestamp: str, content: str, mem_type: str, importance: int):
        self.timestamp = timestamp
        self.content = content
        self.mem_type = mem_type
        self.importance = importance
    def to_dict(self):
        return {"timestamp": self.timestamp, "content": self.content,
                "type": self.mem_type, "importance": self.importance}

class MemoryStream:
    def __init__(self):
        self.memories: List[Memory] = []
        self.importance_since_reflection = 0
        self.reflection_threshold = 50
    def add(self, memory: Memory):
        self.memories.append(memory)
        if memory.mem_type != "background":
            self.importance_since_reflection += memory.importance
    def should_reflect(self) -> bool:
        return self.importance_since_reflection >= self.reflection_threshold
    def reset_reflection_counter(self):
        self.importance_since_reflection = 0
    def get_recent(self, n: int = 10, mem_type: Optional[str] = None) -> List[Memory]:
        filtered = self.memories if mem_type is None else [m for m in self.memories if m.mem_type == mem_type]
        return filtered[-n:]
    def format_memories(self, memories: List[Memory]) -> str:
        return "\n".join(f"[{m.timestamp}] {m.content}" for m in memories)


def create_observation(day: int, slot: int, ctx: dict, action: int, steps: int) -> str:
    """
    【修改3】action 现在是 0/1/2
    【修改8】activity 用 ON_FOOT 而非 WALKING
    【修改11】加入 response 信息
    【修改13】区分 active suggestion (send=1) 和 sedentary suggestion (send=2)
    """
    ts = f"Day{day}_Slot{slot}"
    obs = f"[{ts}] Location: {ctx['location']}. "
    obs += f"Weather: {ctx['weather']}, {ctx['temperature']}C. "
    obs += f"Steps in prior 30min: {ctx['prior_30min_steps']}. Activity: {ctx['activity']}. "
    
    if action == 1:
        obs += "System sent an ACTIVE walking suggestion. "
        obs += f"Steps in following 30min: {steps}. "
        if ctx.get('response'):
            obs += f"Participant response: {ctx['response']}. "
        if steps == 0:
            obs += "Participant did NOT respond to suggestion. "
        elif steps > 200:
            obs += "Participant responded positively with walking. "
    elif action == 2:
        obs += "System sent a SEDENTARY/stand-up suggestion. "
        obs += f"Steps in following 30min: {steps}. "
        if ctx.get('response'):
            obs += f"Participant response: {ctx['response']}. "
    else:
        obs += f"No suggestion sent. Steps in following 30min: {steps}. "
    
    return obs


# ================================================================
# 第三部分：Prompt 模板（不变，略）
# ================================================================

SYSTEM_SCORE = "You are a behavioral scoring system. Output ONLY a single digit (1-5). No explanation."
IMPORTANCE_SYSTEM = "You are a behavioral analysis system. Rate importance from 1-10. Output ONLY the number."
IMPORTANCE_USER = """Rate the importance of this event for understanding the participant's exercise behavior and notification response patterns.
1 = completely routine (sitting at desk as usual)
10 = extremely important (first time responding to a notification, or ignoring notifications for many days straight)

Event: {observation}

Output ONLY a single number (1-10)."""

REFLECTION_Q_PROMPT = """Below are the participant's recent behavior records:

{recent_memories}

List exactly 3 questions worth investigating to understand:
- Whether the participant's response to activity suggestions is changing
- The participant's daily behavioral patterns
- What factors may influence whether they respond to suggestions

List 3 questions, one per line. Be specific and evidence-based."""

REFLECTION_GEN_PROMPT = """Question: "{question}"

Relevant evidence:
{relevant_memories}

Write a 1-2 sentence high-level inference based on the evidence above. Be specific and cite patterns you observe."""

MOTIVATION_PROMPT = """## Participant Background
{seed_memory}

## Recent Reflections
{recent_reflections}

## Recent Behavior Records (newest first)
{recent_observations}

## Current State
Study day: {study_day}, time slot {slot}
Steps in prior 30min: {pre30_steps}
Suggestion sent: {action_desc}
Steps in following 30min: {post_steps}

## Task
On a 1-5 scale, rate this participant's current INTRINSIC MOTIVATION — the self-directed drive to walk, independent of external prompts.

1 = Very low. Participant only walks when prompted, or shows declining activity over days.
2 = Low. Mostly sedentary without prompts, occasional activity.
3 = Moderate. Some self-initiated walking but inconsistent.
4 = High. Regularly walks without prompts, stable or increasing trend.
5 = Very high. Consistently active regardless of suggestions, increasing self-initiated activity.

IMPORTANT: Compare steps at prompted vs unprompted decision points. If participant walks 200+ steps WITHOUT a suggestion, motivation is likely HIGH.

Output ONLY a single digit (1-5)."""

HABIT_PROMPT = """## Participant Background
{seed_memory}

## Recent Behavior Records (newest first)
{recent_observations}

## Behavioral Regularity Indicators
Steps in past 7 days by slot: {slot_pattern}
Day-to-day consistency: {consistency_desc}

## Current State
Study day: {study_day}

## Task
On a 1-5 scale, rate this participant's current HABIT STRENGTH — how automatic and regular their walking behavior has become.

1 = No habit. Walking is entirely effortful and irregular.
2 = Weak. Occasionally walks at similar times but mostly inconsistent.
3 = Forming. Some regular patterns emerging (e.g., walks at same slot most days).
4 = Moderate. Consistent patterns across multiple days, walks at predictable times.
5 = Strong. Highly regular, walks at same times daily regardless of context.

Output ONLY a single digit (1-5)."""

RECEPTIVITY_PROMPT = """## Participant Background
{seed_memory}

## Recent Reflections
{recent_reflections}

## Recent Behavior Records (newest first)
{recent_observations}

## Current State
Study day: {study_day}, time slot {slot}
Current dosage: {dosage:.2f}

## Task
On a 1-5 scale, how likely is this participant to open and respond to an activity suggestion RIGHT NOW by walking in the next 30 minutes?

1 = Very unlikely. Participant has a pattern of ignoring suggestions, or recent response rate is very low.
2 = Unlikely. Participant occasionally responds but mostly ignores.
3 = Uncertain. Not enough information, or response pattern is inconsistent.
4 = Likely. Participant has been responding positively to suggestions recently.
5 = Very likely. Participant actively responds and shows positive attitude.

Output ONLY a single digit (1-5)."""

SYS_ADJUSTMENT = "You are a behavioral simulation module. Output ONLY a single integer from -50 to 100."

PROMPT_ADJUSTMENT = """User: {persona}
Psychological state: motivation={motivation}/5, habit={habit}/5, receptivity={receptivity}/5
Context: Day {study_day}, slot {slot}, location={location}. Suggestion type: {action_desc}.
Base predicted steps: {base_steps}.

Recent behavior:
{recent_obs}

How does psychology adjust actual steps vs base prediction?
Output integer from -50 to +100 (percentage).
Output ONLY a single integer."""


# ================================================================
# 第四部分：LLM 接口（不变）
# ================================================================

class Qwen3BLLM:
    def __init__(self, model_path: str = "Qwen/Qwen3-8B-AWQ"):
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        print(f"Loading model: {model_path}")
        self.llm = LLM(model=model_path, quantization="awq",
                        gpu_memory_utilization=0.90, max_model_len=4096,
                        trust_remote_code=True)
        self.score_guided = GuidedDecodingParams(choice=["1","2","3","4","5"])
        self.score_params = SamplingParams(temperature=0.3, max_tokens=1, guided_decoding=self.score_guided)
        self.importance_guided = GuidedDecodingParams(choice=[str(i) for i in range(1,11)])
        self.importance_params = SamplingParams(temperature=0.1, max_tokens=2, guided_decoding=self.importance_guided)
        self.text_params = SamplingParams(temperature=0.3, max_tokens=200)
        adj_choices = [str(i) for i in range(-50, 101)]
        self.adj_guided = GuidedDecodingParams(choice=adj_choices)
        self.adj_params = SamplingParams(temperature=0.3, max_tokens=4, guided_decoding=self.adj_guided)
        self.call_count = 0
        print("Model loaded!")
    
    def _prompt(self, system: str, user: str) -> str:
        return (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n/no_think\n")
    def score_importance(self, system, user):
        out = self.llm.generate([self._prompt(system, user)], self.importance_params)
        self.call_count += 1; return out[0].outputs[0].text.strip()
    def generate_text(self, system, user):
        out = self.llm.generate([self._prompt(system, user)], self.text_params)
        self.call_count += 1; return out[0].outputs[0].text.strip()
    def batch_score(self, prompts):
        fmt = [self._prompt(p["system"], p["user"]) for p in prompts]
        out = self.llm.generate(fmt, self.score_params)
        self.call_count += len(prompts); return [o.outputs[0].text.strip() for o in out]
    def judge_adjustment(self, system, user):
        out = self.llm.generate([self._prompt(system, user)], self.adj_params)
        self.call_count += 1
        try: return int(out[0].outputs[0].text.strip())
        except: return 0
    def batch_importance(self, prompts):
        fmt = [self._prompt(p["system"], p["user"]) for p in prompts]
        out = self.llm.generate(fmt, self.importance_params)
        self.call_count += len(prompts); return [o.outputs[0].text.strip() for o in out]
    def batch_adjustment(self, prompts):
        fmt = [self._prompt(p["system"], p["user"]) for p in prompts]
        out = self.llm.generate(fmt, self.adj_params)
        self.call_count += len(prompts)
        return [int(o.outputs[0].text.strip()) if o.outputs[0].text.strip().lstrip('-').isdigit() else 0 for o in out]
    def batch_text(self, prompts):
        fmt = [self._prompt(p["system"], p["user"]) for p in prompts]
        out = self.llm.generate(fmt, self.text_params)
        self.call_count += len(prompts); return [o.outputs[0].text.strip() for o in out]


class SimulatedLLM:
    def __init__(self): self.call_count = 0
    def score_importance(self, s, u): self.call_count += 1; return str(np.random.randint(3, 8))
    def generate_text(self, s, u): self.call_count += 1; return "Simulated inference."
    def batch_score(self, p): self.call_count += len(p); return [str(np.random.randint(1, 6)) for _ in p]
    def judge_adjustment(self, s, u): self.call_count += 1; return np.random.randint(-20, 30)
    def batch_importance(self, p): self.call_count += len(p); return [str(np.random.randint(3, 8)) for _ in p]
    def batch_adjustment(self, p): self.call_count += len(p); return [np.random.randint(-20, 30) for _ in p]
    def batch_text(self, p): self.call_count += len(p); return ["Simulated inference." for _ in p]


# ================================================================
# 第五部分：上下文和步数生成
# ================================================================

def generate_context(ext: V1DataExtractor, params: dict, day: int, slot: int, traj: list) -> dict:
    """
    【修改14】location 从该用户所属 cluster 的 weekday/weekend × slot 分布中采样
    """
    is_weekday = (day - 1) % 7 < 5
    cluster_id = int(params.get('cluster', 0))
    
    # 【修改14】从 cluster 的地点分布采样
    loc_dist = ext.cluster_location_dist.get(cluster_id, {}).get((is_weekday, slot), None)
    if loc_dist is None:
        # fallback 到全局分布
        loc_dist = ext.location_by_slot.get(slot, {'Home': 1.0})
    
    locs = list(loc_dist.keys())
    probs = list(loc_dist.values())
    probs = [p / sum(probs) for p in probs]  # 归一化
    location = np.random.choice(locs, p=probs)
    
    # avail 率也用 cluster 的
    cl_avail = ext.cluster_avail_by_slot.get(cluster_id, {})
    avail_prob = cl_avail.get(slot, ext.avail_by_slot.get(slot, 0.87))
    avail = int(np.random.random() < avail_prob)
    
    # 天气
    weather_options = ['Clear', 'Partly Cloudy', 'Mostly Cloudy', 'Overcast', 'Rain']
    weather = np.random.choice(weather_options, p=[0.3, 0.3, 0.2, 0.1, 0.1])
    temperature = round(np.random.normal(22, 8), 1)
    
    # 前30分钟步数
    user_scale = params['predicted_mean_steps'] / max(ext.global_mean_steps, 1)
    ss = ext.slot_stats.get(slot, {'mean': 200})
    prior = max(0, int(np.random.exponential(ss['mean'] * user_scale * 0.8)))
    
    yesterday = sum(t['jbsteps30'] for t in traj[-5:]) if len(traj) >= 5 else int(params['predicted_mean_steps'] * 5)
    
    # activity
    if prior > 200:
        activity = 'ON_FOOT'
    elif np.random.random() < 0.08:
        activity = 'IN_VEHICLE'
    else:
        activity = 'STILL'
    
    # 【修改9】send: 三值概率
    # 先决定是否 randomized（~59%），然后按三值概率分配
    if avail and np.random.random() < ext.randomization_rate:
        # is_randomized=True: 按 send_probs 采样
        send_vals = list(ext.send_probs.keys())
        send_ps = list(ext.send_probs.values())
        action = int(np.random.choice(send_vals, p=send_ps))
    else:
        action = 0  # not randomized → no send
    
    if not avail:
        action = 0
    
    # 【修改11】生成 response
    response = 'no_send'
    if action > 0:
        resp_vals = list(ext.response_probs.keys())
        resp_ps = list(ext.response_probs.values())
        response = np.random.choice(resp_vals, p=resp_ps)
    
    return dict(study_day=day, slot=slot, location=location, avail=avail,
                temperature=temperature, weather=weather,
                prior_30min_steps=prior, yesterday_steps=yesterday,
                activity=activity, weekday=(day - 1) % 7 < 5,
                response=response)


def generate_base_steps(params: dict, ctx: dict, action: int, dosage: float, ext: V1DataExtractor) -> int:
    """
    【修改3】action=1 和 action=2 都有治疗效果，但可能不同
    【修改6】用 cleaned 的 global_mean=246.5
    【修改7】零值率用 cleaned 的 0.371
    """
    # slot 比例（从 cleaned 数据计算：141/247, 307/247, 260/247, 268/247, 256/247）
    slot_ratios = {1: 0.57, 2: 1.24, 3: 1.06, 4: 1.09, 5: 1.04}
    baseline = params['predicted_mean_steps'] * slot_ratios.get(ctx['slot'], 1.0)
    
    # 地点效应（从 cleaned 的 location 平均步数）
    loc_mean = ext.steps_by_location.get(ctx['location'], ext.global_mean_steps)
    loc_ratio = loc_mean / max(ext.global_mean_steps, 1)
    baseline *= (0.5 + 0.5 * loc_ratio)  # 地点效应混合
    
    if not ctx['weekday']:
        baseline *= 0.9
    
    # 【修改3】治疗效果：send=1 和 send=2 都有效
    week = ctx['study_day'] // 7
    decay = max(0, 1.0 - 0.22 * week)
    # 【修改12】send=1 和 send=2 都算发送
    if action > 0:
        te = params['predicted_te'] * decay * max(0, 1.0 - 0.05 * dosage)
        if action == 2:
            te *= 0.8  # sedentary suggestion 效果略弱
    else:
        te = 0
    
    raw = max(0, (baseline + te) * np.random.lognormal(0, 0.5))
    
    # 【修改7】零值率从 cleaned 数据
    zero_prob = ext.global_zero_rate + 0.01 * max(0, ctx['study_day'] - 20)
    if ctx['activity'] == 'IN_VEHICLE':
        zero_prob = 0.7
    
    return 0 if np.random.random() < zero_prob else int(raw)


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
    
    logger = TraceLogger(output_dir)
    logger.log_seed_memory(persona, len(stream.memories))
    
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
                base, adj, steps = 0, 0, 0
            
            reward = np.log(steps + 0.5)
            # 【修改12】dosage: send=1 和 send=2 都累加
            dosage = 0.95 * dosage + (1 if action > 0 else 0)
            
            observation = create_observation(day, slot, ctx, action, steps)
            
            imp_prompt = IMPORTANCE_USER.format(observation=observation)
            imp_str = llm.score_importance(IMPORTANCE_SYSTEM, imp_prompt)
            try: importance = max(1, min(10, int(imp_str)))
            except: importance = 5
            
            stream.add(Memory(f"Day{day}_Slot{slot}", observation, "observation", importance))
            logger.log_observation(step_idx,
                pd.Series({'timestamp': f"Day{day}_Slot{slot}", 'send': action,
                           'jbsteps30': steps, 'jbsteps30pre': ctx['prior_30min_steps'],
                           'dec.location.category': ctx['location'],
                           'recognized.activity': ctx['activity'], 'dosage': dosage}),
                observation, importance)
            
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
                logger.log_reflection(step_idx, f"Day{day}_Slot{slot}", q_prompt_text, q_response, reflection_details)
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
            
            logger.log_scoring(idx=step_idx,
                row=pd.Series({'timestamp': f"Day{day}_Slot{slot}", 'send': action, 'jbsteps30': steps,
                    'jbsteps30pre': ctx['prior_30min_steps'], 'dec.location.category': ctx['location'],
                    'recognized.activity': ctx['activity'], 'dosage': dosage, 'slot': slot}),
                prompts={"receptivity": r_user, "feasibility": h_user, "energy": m_user},
                raw_samples={"receptivity": all_scores[2*n_samples:], "feasibility": all_scores[n_samples:2*n_samples], "energy": all_scores[:n_samples]},
                final_scores={"receptivity": r_score, "feasibility": h_score, "energy": m_score},
                memory_context={"n_observations": len(stream.get_recent(999, "observation")),
                    "n_reflections": len(stream.get_recent(999, "reflection")),
                    "recent_observations": recent_obs_text, "recent_reflections": recent_ref_text})
            
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
    
    logger.close()
    memory_log = [m.to_dict() for m in stream.memories]
    with open(os.path.join(output_dir, f"user{user_id}_memory_log.json"), 'w', encoding='utf-8') as f:
        json.dump(memory_log, f, ensure_ascii=False, indent=2)
    
    total_time = time.time() - start_time
    print(f"\nUser {user_id} done: {len(df)} points, {llm.call_count} LLM calls, {total_time:.0f}s")
    return df


# ================================================================
# 第七部分：主入口
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['llm', 'sim'], default='sim')
    parser.add_argument('--model', default='Qwen/Qwen3-8B-AWQ')
    parser.add_argument('--n_users', type=int, default=1)
    parser.add_argument('--n_days', type=int, default=42)
    parser.add_argument('--n_samples', type=int, default=3)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--users_csv', default='/mnt/user-data/uploads/users.csv')
    # 【修改1】输入改为 cleaned_output.csv
    parser.add_argument('--cleaned_csv', default='/mnt/user-data/uploads/cleaned_output.csv')
    parser.add_argument('--output_dir', default='/mnt/user-data/outputs/synth_output')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("=" * 60)
    print(f"HeartSteps Synthetic User Generator v3-cleaned (mode={args.mode})")
    print("=" * 60)
    
    # 【修改1】用 cleaned_output.csv
    ext = V1DataExtractor(args.users_csv, args.cleaned_csv)
    
    if args.mode == 'llm':
        llm = Qwen3BLLM(args.model)
    else:
        llm = SimulatedLLM()
    
    baseline_df = generate_baseline_vectors(ext, max(args.n_users, 10))
    
    params = baseline_df.iloc[0].to_dict()
    user_output_dir = os.path.join(args.output_dir, "user1")
    df = simulate_user(user_id=1, user_params=params, ext=ext, llm=llm,
                       n_days=args.n_days, slots_per_day=5, n_samples=args.n_samples,
                       output_dir=user_output_dir, verbose=True)
    df.to_csv(os.path.join(user_output_dir, "user1_trajectory.csv"), index=False)
    
    print(f"\nTotal LLM calls: {llm.call_count}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
