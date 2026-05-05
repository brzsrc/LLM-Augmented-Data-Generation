"""
run_user2_annotation.py
=======================
在 AutoDL A100 上用 Qwen3-8B-AWQ + vLLM 对 User 2 全部决策点进行心理状态标注

使用方式：
  cd /root/data
  python3 run_user2_annotation.py

输入文件（放在同目录下）：
  - suggestions.csv
  - users.csv

输出文件：
  - /root/output/user2_annotated.csv      （标注结果）
  - /root/output/user2_memory_log.json     （记忆流日志）
  - /root/output/user2_checkpoint.json     （断点续跑用）

预计：
  - ~3,400 次 vLLM 调用
  - A100-80GB 上约 1-2 小时
  - A100-40GB 上约 2-3 小时
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from collections import Counter
from pathlib import Path

# ================================================================
# 第一部分：数据预处理（从 approach2_pipeline.py 提取）
# ================================================================

def load_and_clean(path: str = "suggestions.csv") -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df['timestamp'] = pd.to_datetime(df['sugg.select.utime'])
    df['date'] = df['timestamp'].dt.date
    df['slot'] = df['sugg.select.slot'].astype(int)
    df['avail'] = df['avail'].map({True: 1, 'True': 1, False: 0, 'False': 0}).fillna(0).astype(int)
    df['send'] = df['send'].map({True: 1, 'True': 1, False: 0, 'False': 0}).fillna(0).astype(int)
    
    def encode_location(loc):
        if pd.isna(loc): return 2
        loc = str(loc).lower()
        if 'home' in loc: return 0
        elif 'work' in loc: return 1
        return 2
    
    df['location'] = df['dec.location.category'].apply(encode_location)
    df['temperature'] = pd.to_numeric(df['dec.temperature'], errors='coerce').fillna(20)
    df['jbsteps30'] = pd.to_numeric(df['jbsteps30'], errors='coerce').fillna(0)
    df['jbsteps30pre'] = pd.to_numeric(df['jbsteps30pre'], errors='coerce').fillna(0)
    df['reward'] = np.log(df['jbsteps30'] + 0.5)
    return df


def compute_dosage(df: pd.DataFrame, lam: float = 0.95) -> pd.DataFrame:
    df = df.sort_values(['user.index', 'timestamp']).reset_index(drop=True)
    dosage_values = []
    for uid, group in df.groupby('user.index'):
        X = 0.0
        for idx, row in group.iterrows():
            dosage_values.append(X)
            X = lam * X + (1.0 if row['send'] == 1 else 0.0)
    df['dosage'] = dosage_values
    return df


# ================================================================
# 第二部分：用户画像构建（从 user_profile_builder.py 提取）
# ================================================================

def build_user_profile(users_path: str, user_id: int) -> str:
    df = pd.read_csv(users_path)
    row = df[df['user.index'] == user_id].iloc[0]
    
    def safe(val, default="unknown"):
        return str(val) if pd.notna(val) else default
    
    def likert5(val):
        m = {1: "very low", 2: "low", 3: "moderate", 4: "high", 5: "very high"}
        try: return m.get(int(val), str(val))
        except: return "unknown"
    
    text = f"Participant is {safe(row.get('age'))}-year-old {safe(row.get('gender'))}, "
    text += f"{safe(row.get('marital'))}, {safe(row.get('ethnicity'))}. "
    text += f"Education: {safe(row.get('education'))}. Occupation: {safe(row.get('occupation'))}. "
    text += f"Household size: {safe(row.get('household.size'))}, children: {safe(row.get('children'), '0')}. "
    text += f"Study duration: {safe(row.get('totaldays'))} days. "
    
    # Conscientiousness
    consc = row.get('consc', None)
    if pd.notna(consc):
        text += f"Conscientiousness score: {int(consc)}/28. "
    
    # IPAQ
    ipaq_labels = {1: "low", 2: "moderate", 3: "high (HEPA)"}
    ipaq_cat = row.get('ipaq.hepa.intake', None)
    if pd.notna(ipaq_cat):
        text += f"IPAQ activity level: {ipaq_labels.get(int(ipaq_cat), 'unknown')}. "
    
    walk_time = row.get('walk.time.intake', 0)
    sit_time = row.get('sit.time.intake', 0)
    if pd.notna(walk_time): text += f"Daily walking: ~{int(walk_time)} min. "
    if pd.notna(sit_time): text += f"Daily sitting: ~{int(sit_time)} min. "
    
    # Self-efficacy
    selfeff = row.get('selfeff.intake', None)
    if pd.notna(selfeff):
        text += f"Exercise self-efficacy: {int(selfeff)}/25 "
        items = []
        for col, label in [('selfeff.tired.intake', 'when tired'),
                           ('selfeff.badmood.intake', 'bad mood'),
                           ('selfeff.notime.intake', 'no time'),
                           ('selfeff.vaca.intake', 'vacation'),
                           ('selfeff.precip.intake', 'bad weather')]:
            v = row.get(col, None)
            if pd.notna(v): items.append(f"{label}={likert5(v)}")
        if items: text += f"({', '.join(items)}). "
    
    # Environment
    env_items = []
    for col, label in [('sidewalkhome', 'sidewalks'), ('recfacilities', 'rec facilities'),
                        ('seeactive', 'see active people'), ('unsafenight', 'unsafe at night')]:
        v = row.get(col, None)
        if pd.notna(v): env_items.append(f"{label}={likert5(v)}")
    if env_items: text += f"Home environment: {', '.join(env_items)}. "
    
    # Travel
    travel_start = row.get('travel.start')
    travel_end = row.get('travel.end')
    if pd.notna(travel_start) and pd.notna(travel_end):
        text += f"Travel period during study: {travel_start} to {travel_end}. "
    
    return text


# ================================================================
# 第三部分：记忆流
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


def create_observation(row: pd.Series) -> str:
    time = row['timestamp']
    location = row.get('dec.location.exact', 'unknown')
    location_cat = row.get('dec.location.category', 'unknown')
    weather = row.get('dec.weather.condition', 'unknown')
    temp = row.get('temperature', 'N/A')
    pre_steps = int(row.get('jbsteps30pre', 0))
    action = row.get('send', 0)
    post_steps = int(row.get('jbsteps30', 0))
    response = row.get('response', None)
    activity = row.get('recognized.activity', 'UNKNOWN')
    message = row.get('returned.message', '')
    
    obs = f"[{time}] Location: {location} ({location_cat}). "
    obs += f"Weather: {weather}, {temp}C. "
    obs += f"Steps in prior 30min: {pre_steps}. Activity: {activity}. "
    
    if action == 1:
        obs += "System sent an activity suggestion. "
        if pd.notna(message) and message:
            obs += f"Message: '{str(message)[:60]}'. "
        obs += f"Steps in following 30min: {post_steps}. "
        if pd.notna(response):
            if response == 'good': obs += "Participant gave positive feedback. "
            elif response == 'bad': obs += "Participant gave negative feedback. "
    else:
        obs += f"No suggestion sent. Steps in following 30min: {post_steps}. "
    
    return obs


# ================================================================
# 第四部分：vLLM Qwen3-8B 接口
# ================================================================

class Qwen3BLLM:
    """
    通过 vLLM 离线推理调用 Qwen3-8B-AWQ
    使用 constrained decoding 保证输出为 1-5 的整数
    """

    def __init__(self, model_path: str = "Qwen/Qwen3-8B-AWQ"):
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        
        print(f"Loading model: {model_path}")
        print("This may take 1-2 minutes on first load...")
        
        self.llm = LLM(
            model=model_path,
            quantization="awq",
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            trust_remote_code=True,
            # Qwen3 默认启用 thinking mode，我们需要关闭它以获得直接输出
            # 通过在 prompt 中加 /no_think 标记来关闭
        )
        
        # 评分任务：constrained decoding，只能输出 1-5
        self.score_guided = StructuredOutputsParams(choice=["1", "2", "3", "4", "5"])
        self.score_params = SamplingParams(
            temperature=0.3,
            max_tokens=1,
            guided_decoding=self.score_guided,
        )
        
        # 重要性评分：constrained decoding，1-10
        self.importance_guided = StructuredOutputsParams(
            choice=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        )
        self.importance_params = SamplingParams(
            temperature=0.1,
            max_tokens=2,
            guided_decoding=self.importance_guided,
        )
        
        # 反思生成：自由文本，限制长度
        self.text_params = SamplingParams(
            temperature=0.3,
            max_tokens=200,
        )
        
        self.call_count = 0
        self.total_input_tokens = 0
        print("Model loaded successfully!")
    
    def _build_chat_prompt(self, system: str, user: str) -> str:
        """构建 Qwen3 chat 格式的 prompt，使用 /no_think 关闭 thinking mode"""
        # Qwen3 chat template:
        # <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{user}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n/no_think\n"
        return prompt
    
    def score(self, system: str, user: str) -> str:
        """评分调用：输出 1-5"""
        prompt = self._build_chat_prompt(system, user)
        outputs = self.llm.generate([prompt], sampling_params=self.score_params)
        self.call_count += 1
        result = outputs[0].outputs[0].text.strip()
        return result
    
    def score_importance(self, system: str, user: str) -> str:
        """重要性评分：输出 1-10"""
        prompt = self._build_chat_prompt(system, user)
        outputs = self.llm.generate([prompt], sampling_params=self.importance_params)
        self.call_count += 1
        return outputs[0].outputs[0].text.strip()
    
    def generate_text(self, system: str, user: str) -> str:
        """文本生成（反思等）"""
        prompt = self._build_chat_prompt(system, user)
        outputs = self.llm.generate([prompt], sampling_params=self.text_params)
        self.call_count += 1
        return outputs[0].outputs[0].text.strip()
    
    def batch_score(self, prompts: List[Dict[str, str]]) -> List[str]:
        """
        批量评分：一次提交多个 prompt，利用 vLLM 的 continuous batching
        prompts: [{"system": "...", "user": "..."}, ...]
        """
        formatted = [self._build_chat_prompt(p["system"], p["user"]) for p in prompts]
        outputs = self.llm.generate(formatted, sampling_params=self.score_params)
        self.call_count += len(prompts)
        return [o.outputs[0].text.strip() for o in outputs]


# ================================================================
# 第五部分：英文版 Prompt（优化 7B 模型）
# ================================================================

SYSTEM_SCORE = "You are a behavioral scoring system. Output ONLY a single digit (1-5). No explanation."

IMPORTANCE_SYSTEM = "You are a behavioral analysis system. Rate importance from 1-10. Output ONLY the number."

IMPORTANCE_USER = """Rate the importance of this event for understanding the participant's exercise behavior and notification response patterns.
1 = completely routine (sitting at desk as usual)
10 = extremely important (first time responding to a notification, or ignoring notifications for many days straight)

Event: {observation}

Output ONLY a single number (1-10)."""

RECEPTIVITY_PROMPT = """## Participant Background
{seed_memory}

## Recent Reflections
{recent_reflections}

## Recent Behavior Records (newest first)
{recent_observations}

## Current State
Time: {current_time}
Location: {current_location}
Activity: {current_activity}
Steps in prior 30min: {pre30_steps}
Current dosage: {current_dosage:.2f}

## Task
On a 1-5 scale, how likely is this participant to open and respond to an activity suggestion RIGHT NOW by walking in the next 30 minutes?

1 = Very unlikely. Participant has a pattern of ignoring suggestions, or recent response rate is very low.
2 = Unlikely. Participant occasionally responds but mostly ignores.
3 = Uncertain. Not enough information, or response pattern is inconsistent.
4 = Likely. Participant has been responding positively to suggestions recently.
5 = Very likely. Participant actively responds and shows positive attitude.

IMPORTANT: Base your judgment on the specific behavioral evidence above. If the participant ignored 4 out of the last 5 suggestions (post-suggestion steps = 0), receptivity should be LOW.

Output ONLY a single digit (1-5)."""

FEASIBILITY_PROMPT = """## Participant Background
{seed_memory}

## Current State
Time: {current_time}
Location: {current_location} (detail: {location_exact})
Location category: {location_category}
Activity: {current_activity}
Weather: {weather}, {temperature}C

## Same-location history
{location_history}

## Task
On a 1-5 scale, how FEASIBLE is it for this participant to stop what they are doing and go for a 10-minute walk RIGHT NOW?

This is NOT about willingness — it's about whether the physical and social environment allows walking.

1 = Not feasible. Likely in a meeting, driving, or otherwise unable to leave.
2 = Barely feasible. Could physically stand up but high social/task cost.
3 = Somewhat feasible. Can leave but requires effort to overcome inertia.
4 = Feasible. In a flexible state (at desk alone, at home, on break).
5 = Very feasible. Already outdoors, just finished a task, or on a break.

IMPORTANT: If history shows the participant NEVER walks at this location (steps always 0), feasibility is likely LOW.

Output ONLY a single digit (1-5)."""

ENERGY_PROMPT = """## Participant Background
{seed_memory}

## Today's Timeline
{today_timeline}

## Same time-slot history (past 3 days)
{same_slot_history}

## Current State
Time: {current_time}
Decision point {slot_number} of 5 today
Activity: {current_activity}
Steps in prior 30min: {pre30_steps}
Study day: {study_day}

## Task
On a 1-5 scale, what is this participant's current PSYCHOLOGICAL ENERGY level — the mental bandwidth to deviate from their current inertia?

1 = Exhausted. End of long day, or prolonged STILL activity periods.
2 = Low. Has been in the same activity for a long time, strong inertia.
3 = Moderate. No clear high or low energy signals.
4 = High. May have just completed a task, started a new activity, or beginning of day.
5 = Very high. Fresh start, just had a pleasant activity, or showing initiative.

IMPORTANT: If this is decision point 4-5 (evening), energy tends to be lower. If study day > 30, overall engagement may have declined.

Output ONLY a single digit (1-5)."""

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


# ================================================================
# 第六部分：主标注逻辑
# ================================================================

def annotate_user(
    user_id: int,
    user_data: pd.DataFrame,
    seed_text: str,
    llm: Qwen3BLLM,
    n_samples: int = 5,
    checkpoint_path: str = "/root/output/user2_checkpoint.json",
    resume_from: int = 0,  # 断点续跑的起始索引
) -> pd.DataFrame:
    """
    对一个参与者的所有决策点进行标注
    支持断点续跑：每10个时间点保存一次 checkpoint
    """
    from trace_logger import TraceLogger
    
    print(f"\n{'='*60}")
    print(f"Annotating User {user_id} ({len(user_data)} decision points)")
    print(f"n_samples={n_samples}, resume_from={resume_from}")
    print(f"{'='*60}")
    
    # 初始化记忆流
    stream = MemoryStream()
    for line in seed_text.strip().split('. '):
        if line.strip():
            stream.add(Memory("study_start", line.strip() + ".", "background", 7))
    
    # 初始化 trace logger
    logger = TraceLogger("/root/output")
    logger.log_seed_memory(seed_text, len(stream.memories))
    
    user_data = user_data.sort_values('timestamp').reset_index(drop=True)
    first_date = user_data['date'].iloc[0]
    T = len(user_data)
    
    # 加载已有结果（断点续跑）
    receptivity_raw = []
    feasibility_raw = []
    energy_raw = []
    
    if resume_from > 0 and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            ckpt = json.load(f)
        receptivity_raw = ckpt.get('receptivity_raw', [])[:resume_from]
        feasibility_raw = ckpt.get('feasibility_raw', [])[:resume_from]
        energy_raw = ckpt.get('energy_raw', [])[:resume_from]
        print(f"  Resumed from checkpoint: {resume_from} points loaded")
        
        # 重建记忆流（快速重放前 resume_from 个点，不调用 LLM）
        for idx in range(resume_from):
            row = user_data.iloc[idx]
            obs = create_observation(row)
            stream.add(Memory(str(row['timestamp']), obs, "observation", 5))
            if stream.should_reflect():
                stream.add(Memory(str(row['timestamp']),
                    f"[Auto-reflection at point {idx}] Participant behavior patterns noted.",
                    "reflection", 8))
                stream.reset_reflection_counter()
    
    start_time = time.time()
    
    for idx in range(resume_from, T):
        row = user_data.iloc[idx]
        
        # ── a. 生成观察记录 ──
        observation = create_observation(row)
        
        # ── b. 评估重要性 ──
        imp_prompt = IMPORTANCE_USER.format(observation=observation)
        imp_str = llm.score_importance(IMPORTANCE_SYSTEM, imp_prompt)
        try:
            importance = max(1, min(10, int(imp_str)))
        except:
            importance = 5
        
        stream.add(Memory(str(row['timestamp']), observation, "observation", importance))
        logger.log_observation(idx, row, observation, importance)
        
        # ── c. 反思触发 ──
        if stream.should_reflect():
            recent = stream.get_recent(15)
            recent_text = stream.format_memories(recent)
            
            # 生成反思问题
            q_prompt_text = REFLECTION_Q_PROMPT.format(recent_memories=recent_text)
            q_response = llm.generate_text(
                "You are a behavioral analysis system.",
                q_prompt_text
            )
            questions = [q.strip() for q in q_response.strip().split('\n') if q.strip()][:3]
            
            # 对每个问题生成反思
            reflection_details = []
            for q in questions:
                if len(q) < 5: continue
                ref_gen_prompt = REFLECTION_GEN_PROMPT.format(question=q, relevant_memories=recent_text)
                ref_text = llm.generate_text(
                    "You are a behavioral analysis system. Write concise inferences.",
                    ref_gen_prompt
                )
                stream.add(Memory(str(row['timestamp']), ref_text.strip(), "reflection", 8))
                reflection_details.append({
                    "question": q,
                    "prompt": ref_gen_prompt[:500],
                    "response": ref_text.strip(),
                })
            
            stream.reset_reflection_counter()
            logger.log_reflection(idx, str(row['timestamp']), q_prompt_text, q_response, reflection_details)
            print(f"  [{idx}/{T}] Reflection triggered ({len(questions)} questions)")
        
        # ── d. 准备上下文 ──
        recent_obs = stream.get_recent(10, "observation")
        recent_ref = stream.get_recent(3, "reflection")
        recent_obs_text = stream.format_memories(recent_obs)
        recent_ref_text = stream.format_memories(recent_ref) if recent_ref else "No reflections yet."
        
        study_day = (row['date'] - first_date).days + 1
        
        # 同地点历史
        same_loc = user_data.iloc[:idx]
        loc_cat = row.get('dec.location.category', '')
        same_loc_f = same_loc[same_loc.get('dec.location.category', '') == loc_cat] if loc_cat else same_loc.head(0)
        loc_hist = ""
        for _, lr in same_loc_f.tail(5).iterrows():
            loc_hist += f"[{lr['timestamp']}] steps={int(lr['jbsteps30'])}\n"
        if not loc_hist: loc_hist = "No history at this location."
        
        # 今天时间线
        today = user_data[(user_data['date'] == row['date']) & (user_data.index <= idx)]
        today_text = ""
        for _, tr in today.iterrows():
            today_text += f"[{tr['timestamp']}] activity={tr.get('recognized.activity','?')}, "
            today_text += f"pre30={int(tr['jbsteps30pre'])}steps, post30={int(tr['jbsteps30'])}steps\n"
        
        # 同时段历史
        same_slot = user_data[(user_data['slot'] == row['slot']) & (user_data.index < idx)].tail(3)
        slot_text = ""
        for _, sr in same_slot.iterrows():
            slot_text += f"[{sr['timestamp']}] steps={int(sr['jbsteps30'])}\n"
        if not slot_text: slot_text = "No same-slot history."
        
        # ── e. 批量评分（3个维度 × n_samples 次 = 一批提交） ──
        # 构建所有需要评分的 prompt
        score_prompts = []
        
        # 推送接受度 × n_samples
        r_user = RECEPTIVITY_PROMPT.format(
            seed_memory=seed_text[:400],
            recent_reflections=recent_ref_text,
            recent_observations=recent_obs_text,
            current_time=row['timestamp'],
            current_location=row.get('dec.location.category', 'unknown'),
            current_activity=row.get('recognized.activity', 'UNKNOWN'),
            pre30_steps=int(row['jbsteps30pre']),
            current_dosage=row.get('dosage', 0),
        )
        for _ in range(n_samples):
            score_prompts.append({"system": SYSTEM_SCORE, "user": r_user})
        
        # 情境可行性 × n_samples
        f_user = FEASIBILITY_PROMPT.format(
            seed_memory=seed_text[:400],
            current_time=row['timestamp'],
            current_location=row.get('dec.location.category', 'unknown'),
            location_exact=row.get('dec.location.exact', 'unknown'),
            location_category=row.get('dec.location.category', 'unknown'),
            current_activity=row.get('recognized.activity', 'UNKNOWN'),
            weather=row.get('dec.weather.condition', 'unknown'),
            temperature=row.get('temperature', 'N/A'),
            location_history=loc_hist,
        )
        for _ in range(n_samples):
            score_prompts.append({"system": SYSTEM_SCORE, "user": f_user})
        
        # 心理能量 × n_samples
        e_user = ENERGY_PROMPT.format(
            seed_memory=seed_text[:400],
            today_timeline=today_text,
            same_slot_history=slot_text,
            current_time=row['timestamp'],
            slot_number=row['slot'],
            current_activity=row.get('recognized.activity', 'UNKNOWN'),
            pre30_steps=int(row['jbsteps30pre']),
            study_day=study_day,
        )
        for _ in range(n_samples):
            score_prompts.append({"system": SYSTEM_SCORE, "user": e_user})
        
        # 一次性批量推理（3 × n_samples 个 prompt）
        all_scores = llm.batch_score(score_prompts)
        
        # 解析结果
        def parse_scores(raw_list):
            parsed = []
            for s in raw_list:
                try: parsed.append(max(1, min(5, int(s))))
                except: parsed.append(3)
            return Counter(parsed).most_common(1)[0][0]
        
        r_score = parse_scores(all_scores[0:n_samples])
        f_score = parse_scores(all_scores[n_samples:2*n_samples])
        e_score = parse_scores(all_scores[2*n_samples:3*n_samples])
        
        receptivity_raw.append(r_score)
        feasibility_raw.append(f_score)
        energy_raw.append(e_score)
        
        # ── trace logging ──
        logger.log_scoring(
            idx=idx, row=row,
            prompts={"receptivity": r_user, "feasibility": f_user, "energy": e_user},
            raw_samples={
                "receptivity": all_scores[0:n_samples],
                "feasibility": all_scores[n_samples:2*n_samples],
                "energy": all_scores[2*n_samples:3*n_samples],
            },
            final_scores={"receptivity": r_score, "feasibility": f_score, "energy": e_score},
            memory_context={
                "n_observations": len(stream.get_recent(999, "observation")),
                "n_reflections": len(stream.get_recent(999, "reflection")),
                "recent_observations": recent_obs_text,
                "recent_reflections": recent_ref_text,
            },
        )
        
        # ── f. 进度和断点 ──
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1 - resume_from) / elapsed * 3600
            eta = (T - idx - 1) / (rate / 3600) if rate > 0 else 0
            print(f"  [{idx+1}/{T}] R={r_score} F={f_score} E={e_score} | "
                  f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining | "
                  f"LLM calls: {llm.call_count}")
            
            # 保存断点
            checkpoint = {
                'user_id': user_id,
                'completed': idx + 1,
                'total': T,
                'receptivity_raw': receptivity_raw,
                'feasibility_raw': feasibility_raw,
                'energy_raw': energy_raw,
                'llm_calls': llm.call_count,
            }
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f)
    
    # ── 时序平滑 ──
    def smooth(scores, alpha=0.7):
        result = [float(scores[0])]
        for i in range(1, len(scores)):
            result.append(alpha * scores[i] + (1 - alpha) * result[-1])
        return [max(1, min(5, round(s))) for s in result]
    
    user_data = user_data.copy()
    user_data['receptivity_raw'] = receptivity_raw
    user_data['feasibility_raw'] = feasibility_raw
    user_data['energy_raw'] = energy_raw
    user_data['receptivity'] = smooth(receptivity_raw)
    user_data['feasibility'] = smooth(feasibility_raw)
    user_data['energy'] = smooth(energy_raw)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Annotation complete!")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  LLM calls: {llm.call_count}")
    print(f"  Receptivity: mean={np.mean(receptivity_raw):.2f}, dist={dict(Counter(receptivity_raw))}")
    print(f"  Feasibility: mean={np.mean(feasibility_raw):.2f}, dist={dict(Counter(feasibility_raw))}")
    print(f"  Energy:      mean={np.mean(energy_raw):.2f}, dist={dict(Counter(energy_raw))}")
    print(f"{'='*60}")
    
    # 保存记忆流日志
    memory_log = [m.to_dict() for m in stream.memories]
    with open("/root/output/user2_memory_log.json", 'w', encoding='utf-8') as f:
        json.dump(memory_log, f, ensure_ascii=False, indent=2)
    print(f"Memory log saved: {len(memory_log)} memories")
    
    # 关闭 trace logger
    logger.close()
    
    return user_data


# ================================================================
# 第七部分：主入口
# ================================================================

def main():
    print("=" * 60)
    print("HeartSteps User 2 Full Annotation")
    print("Qwen3-8B-AWQ via vLLM on AutoDL A100")
    print("=" * 60)
    
    # ── 检查文件 ──
    for f in ["suggestions.csv", "users.csv"]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found! Please upload to current directory.")
            sys.exit(1)
    
    os.makedirs("/root/output", exist_ok=True)
    
    # ── 数据预处理 ──
    print("\n[Step 1/4] Loading and preprocessing data...")
    df = load_and_clean("suggestions.csv")
    df = compute_dosage(df)
    user2_data = df[df['user.index'] == 2].copy()
    print(f"  User 2: {len(user2_data)} decision points")
    
    # ── 用户画像 ──
    print("\n[Step 2/4] Building user profile...")
    seed_text = build_user_profile("users.csv", user_id=2)
    print(f"  Profile length: {len(seed_text)} chars")
    print(f"  Preview: {seed_text[:200]}...")
    
    # ── 加载模型 ──
    print("\n[Step 3/4] Loading Qwen3-8B-AWQ via vLLM...")
    
    # 检查是否有本地模型缓存
    model_path = "/root/models/Qwen3-8B-AWQ"
    if not os.path.exists(model_path):
        model_path = "Qwen/Qwen3-8B-AWQ"  # 从 HuggingFace 下载
        print(f"  Model not cached locally, will download from HuggingFace")
    else:
        print(f"  Using cached model: {model_path}")
    
    llm = Qwen3BLLM(model_path=model_path)
    
    # ── 检查断点 ──
    checkpoint_path = "/root/output/user2_checkpoint.json"
    resume_from = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        resume_from = ckpt.get('completed', 0)
        if resume_from > 0:
            print(f"\n  Found checkpoint: {resume_from}/{ckpt.get('total', '?')} points completed")
            resp = input("  Resume from checkpoint? (y/n): ").strip().lower()
            if resp != 'y':
                resume_from = 0
                print("  Starting fresh.")
    
    # ── 运行标注 ──
    print(f"\n[Step 4/4] Running annotation (n_samples=5)...")
    result = annotate_user(
        user_id=2,
        user_data=user2_data,
        seed_text=seed_text,
        llm=llm,
        n_samples=5,
        checkpoint_path=checkpoint_path,
        resume_from=resume_from,
    )
    
    # ── 保存结果 ──
    output_path = "/root/output/user2_annotated.csv"
    result.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # ── 打印摘要 ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Output file: {output_path}")
    print(f"Columns added: receptivity, feasibility, energy (smoothed)")
    print(f"              receptivity_raw, feasibility_raw, energy_raw (raw)")
    
    # 显示前20行预览
    preview_cols = ['timestamp', 'send', 'jbsteps30', 'dosage',
                    'receptivity', 'feasibility', 'energy']
    print(f"\nFirst 20 rows:")
    print(result[preview_cols].head(20).to_string(index=False))
    
    # 统计
    print(f"\nDescriptive statistics:")
    for col in ['receptivity', 'feasibility', 'energy']:
        vals = result[col].values
        print(f"  {col}: mean={np.mean(vals):.2f}, std={np.std(vals):.2f}, "
              f"min={np.min(vals)}, max={np.max(vals)}")
    
    print(f"\nDone! Download /root/output/user2_annotated.csv to continue with TS + OPE.")


if __name__ == "__main__":
    main()
