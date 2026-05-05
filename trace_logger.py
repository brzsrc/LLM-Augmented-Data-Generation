"""
trace_logger.py
===============
记录合成用户生成过程中的每一步细节，方便检查 LLM 的思考质量和心理状态演化

保存四个文件：
  1. user{id}_trace.jsonl       — 每个决策点的完整推理记录（逐行JSON）
  2. user{id}_memory.jsonl      — 记忆流变化历史（每次add记一条）
  3. user{id}_reflections.md    — 所有反思内容的人类可读 Markdown
  4. user{id}_psych_timeline.csv — 心理状态时间线（方便画图）

使用方式：
  logger = TraceLogger("./output", user_id=1)
  
  # 在模拟循环中：
  logger.log_decision_point(day, slot, ctx, action, base_steps, adj, final_steps, state)
  logger.log_observation(timestamp, obs_text, importance)
  logger.log_reflection(day, reflection_text, old_state, new_state, raw_llm, prompt)
  logger.log_memory_event(timestamp, content, mem_type, importance)
  
  # 结束时：
  logger.finalize()
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np


def _json_safe(obj):
    """把 numpy 类型转为 Python 原生类型"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _safe_dump(record: dict) -> str:
    """安全的 JSON 序列化"""
    def convert(o):
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        if isinstance(o, list):
            return [convert(v) for v in o]
        if isinstance(o, tuple):
            return [convert(v) for v in o]
        return _json_safe(o)
    return json.dumps(convert(record), ensure_ascii=False)


class TraceLogger:
    def __init__(self, output_dir: str, user_id: int = 1, user_params: dict = None):
        self.output_dir = output_dir
        self.user_id = user_id
        os.makedirs(output_dir, exist_ok=True)
        
        prefix = f"user{user_id}"
        self.trace_path = os.path.join(output_dir, f"{prefix}_trace.jsonl")
        self.memory_path = os.path.join(output_dir, f"{prefix}_memory.jsonl")
        self.reflection_path = os.path.join(output_dir, f"{prefix}_reflections.md")
        self.timeline_path = os.path.join(output_dir, f"{prefix}_psych_timeline.csv")
        
        # 打开文件句柄
        self.trace_f = open(self.trace_path, 'w', encoding='utf-8')
        self.memory_f = open(self.memory_path, 'w', encoding='utf-8')
        self.reflection_f = open(self.reflection_path, 'w', encoding='utf-8')
        self.timeline_f = open(self.timeline_path, 'w', encoding='utf-8')
        
        # 写 headers
        self.reflection_f.write(f"# Synthetic User {user_id} — Reflection Log\n\n")
        if user_params:
            self.reflection_f.write(f"## User Profile\n\n")
            for k, v in user_params.items():
                if k not in ('user_id',):
                    self.reflection_f.write(f"- **{k}**: {v}\n")
            self.reflection_f.write(f"\n---\n\n")
        
        self.timeline_f.write("day,slot,step_index,motivation,habit,receptivity,"
                              "avail,send,jbsteps30,base_steps,llm_adj_pct,dosage,"
                              "importance_acc,reflection_triggered\n")
        
        self.step_index = 0
        self.reflection_count = 0
        self.decision_count = 0
        self.importance_acc = 0
    
    # ─────────────────────────────────────────────
    # 决策点级别记录
    # ─────────────────────────────────────────────
    
    def log_decision_point(self, day: int, slot: int, ctx: dict, 
                           action: int, base_steps: int, llm_adj_pct: int,
                           final_steps: int, state: dict, dosage: float,
                           prompt_text: str = "", llm_raw_output: str = "",
                           importance: int = 0, importance_acc: int = 0,
                           reflection_triggered: bool = False):
        """
        记录一个决策点的完整推理过程
        """
        self.step_index += 1
        self.decision_count += 1
        self.importance_acc = importance_acc
        
        record = {
            "step_index": self.step_index,
            "day": day,
            "slot": slot,
            "context": {
                "location": ctx.get('location', ''),
                "avail": ctx.get('avail', 0),
                "temperature": ctx.get('temperature', 0),
                "activity": ctx.get('activity', ''),
                "prior_30min_steps": ctx.get('prior_30min_steps', 0),
                "yesterday_steps": ctx.get('yesterday_steps', 0),
                "weekday": ctx.get('weekday', True),
            },
            "action": {
                "send": action,
                "dosage_before": round(dosage, 3),
            },
            "step_generation": {
                "base_steps": base_steps,
                "llm_adjustment_pct": llm_adj_pct,
                "final_steps": final_steps,
                "reward": round(float(__import__('numpy').log(final_steps + 0.5)), 4),
            },
            "psychological_state": {
                "motivation": state['motivation'],
                "habit": state['habit'],
                "receptivity": state['receptivity'],
            },
            "importance": importance,
            "importance_accumulated": importance_acc,
            "reflection_triggered": reflection_triggered,
            "llm_prompt": prompt_text[:500] if prompt_text else "",
            "llm_raw_output": str(llm_raw_output),
        }
        
        self.trace_f.write(_safe_dump(record) + "\n")
        
        # 时间线 CSV
        self.timeline_f.write(
            f"{day},{slot},{self.step_index},{state['motivation']},{state['habit']},"
            f"{state['receptivity']},{ctx.get('avail',0)},{action},{final_steps},"
            f"{base_steps},{llm_adj_pct},{dosage:.3f},{importance_acc},"
            f"{int(reflection_triggered)}\n"
        )
    
    # ─────────────────────────────────────────────
    # 记忆流事件
    # ─────────────────────────────────────────────
    
    def log_memory_event(self, timestamp: str, content: str, 
                          mem_type: str, importance: int,
                          memory_size: int = 0):
        """记录每次记忆流的变化"""
        record = {
            "timestamp": timestamp,
            "mem_type": mem_type,
            "importance": importance,
            "content": content[:300],
            "memory_size_after": memory_size,
            "step_index": self.step_index,
        }
        self.memory_f.write(_safe_dump(record) + "\n")
    
    # ─────────────────────────────────────────────
    # 观察记录
    # ─────────────────────────────────────────────
    
    def log_observation(self, day: int, slot: int, obs_text: str, 
                        importance: int, importance_acc: int):
        """记录一条观察（同时写入 memory 日志）"""
        self.log_memory_event(
            timestamp=f"D{day}S{slot}",
            content=obs_text,
            mem_type="observation",
            importance=importance,
        )
    
    # ─────────────────────────────────────────────
    # 反思记录（最详细）
    # ─────────────────────────────────────────────
    
    def log_reflection(self, day: int, 
                       reflection_text: str,
                       old_state: dict, new_state: dict,
                       raw_llm_state: Tuple[int, int, int],
                       constrained_state: dict,
                       reflect_prompt: str = "",
                       state_prompt: str = "",
                       recent_obs_text: str = "",
                       recent_ref_text: str = "",
                       questions: List[str] = None,
                       inferences: List[str] = None):
        """
        记录一次完整的多步反思过程
        
        参数:
            day: 研究天数
            reflection_text: 所有推断合并的文本
            old_state / new_state: 反思前后状态
            raw_llm_state: LLM 原始输出 (mot, hab, rec)
            constrained_state: 约束后的状态
            reflect_prompt: 生成问题的 prompt
            state_prompt: 状态更新的 prompt
            recent_obs_text: 喂给 LLM 的近期观察
            recent_ref_text: 喂给 LLM 的近期反思
            questions: Step1 生成的 3 个问题
            inferences: Step2 针对每个问题的推断
        """
        self.reflection_count += 1
        questions = questions or []
        inferences = inferences or []
        
        # --- JSONL 记录 ---
        record = {
            "reflection_index": self.reflection_count,
            "day": day,
            "step_index": self.step_index,
            "old_state": old_state.copy(),
            "llm_raw_output": list(raw_llm_state),
            "constrained_state": constrained_state.copy(),
            "state_change": {
                "motivation": constrained_state['motivation'] - old_state['motivation'],
                "habit": constrained_state['habit'] - old_state['habit'],
                "receptivity": constrained_state['receptivity'] - old_state['receptivity'],
            },
            "questions": questions,
            "inferences": inferences,
            "reflection_text": reflection_text[:800],
            "reflect_prompt": reflect_prompt[:500],
            "state_prompt": state_prompt[:500],
        }
        self.trace_f.write(_safe_dump({"type": "reflection", **record}) + "\n")
        self.log_memory_event(f"D{day}R", reflection_text[:200], "reflection", 6)
        self.log_memory_event(f"D{day}U", 
            f"State: mot={constrained_state['motivation']},hab={constrained_state['habit']},"
            f"rec={constrained_state['receptivity']}", "reflection", 5)
        
        # --- Markdown 可读记录 ---
        md = f"### Reflection #{self.reflection_count} — Day {day}\n\n"
        
        # 状态变化摘要
        md += f"**State before**: motivation={old_state['motivation']}, "
        md += f"habit={old_state['habit']}, receptivity={old_state['receptivity']}\n\n"
        
        # Step 1: 问题
        if questions:
            md += f"**Step 1 — Generated Questions:**\n\n"
            for i, q in enumerate(questions, 1):
                md += f"{i}. {q}\n"
            md += "\n"
        
        # Step 2: 推断
        if inferences:
            md += f"**Step 2 — Evidence-Based Inferences:**\n\n"
            for i, (q, inf) in enumerate(zip(questions, inferences), 1):
                md += f"**Q{i}**: {q[:100]}\n"
                md += f"> {inf}\n\n"
        
        # Step 3: 状态更新
        md += f"**Step 3 — State Update:**\n\n"
        md += f"- LLM raw output: ({raw_llm_state[0]}, {raw_llm_state[1]}, {raw_llm_state[2]})\n"
        md += f"- After constraint: motivation={constrained_state['motivation']}, "
        md += f"habit={constrained_state['habit']}, receptivity={constrained_state['receptivity']}\n"
        
        changes = []
        for k in ['motivation', 'habit', 'receptivity']:
            delta = constrained_state[k] - old_state[k]
            if delta != 0:
                changes.append(f"{k} {'+' if delta>0 else ''}{delta}")
        md += f"- **Changes**: {', '.join(changes) if changes else 'No change'}\n\n"
        
        # 折叠的详细信息
        if recent_obs_text:
            md += f"<details>\n<summary>Recent observations fed to LLM ({len(recent_obs_text)} chars)</summary>\n\n"
            md += f"```\n{recent_obs_text[:800]}\n```\n\n</details>\n\n"
        
        if state_prompt:
            md += f"<details>\n<summary>State update prompt</summary>\n\n"
            md += f"```\n{state_prompt[:1000]}\n```\n\n</details>\n\n"
        
        md += "---\n\n"
        self.reflection_f.write(md)
    
    # ─────────────────────────────────────────────
    # 用户初始化记录
    # ─────────────────────────────────────────────
    
    def log_user_init(self, params: dict, initial_state: dict, persona: str):
        """记录用户初始化信息"""
        record = {
            "type": "user_init",
            "user_id": self.user_id,
            "params": {k: _json_safe(v) for k, v in params.items()},
            "initial_state": initial_state.copy(),
            "persona": persona,
        }
        self.trace_f.write(_safe_dump(record) + "\n")
    
    # ─────────────────────────────────────────────
    # 结束
    # ─────────────────────────────────────────────
    
    def finalize(self, final_state: dict = None, summary_stats: dict = None):
        """写入摘要并关闭文件"""
        
        # 写摘要到 reflection markdown
        self.reflection_f.write(f"\n## Summary\n\n")
        self.reflection_f.write(f"- Total decision points: {self.decision_count}\n")
        self.reflection_f.write(f"- Total reflections triggered: {self.reflection_count}\n")
        self.reflection_f.write(f"- Average decisions between reflections: "
                                f"{self.decision_count / max(self.reflection_count, 1):.1f}\n")
        if final_state:
            self.reflection_f.write(f"- Final state: motivation={final_state.get('motivation','?')}, "
                                    f"habit={final_state.get('habit','?')}, "
                                    f"receptivity={final_state.get('receptivity','?')}\n")
        if summary_stats:
            self.reflection_f.write(f"\n### Behavioral Statistics\n\n")
            for k, v in summary_stats.items():
                self.reflection_f.write(f"- {k}: {v}\n")
        
        # 写摘要到 trace
        record = {
            "type": "summary",
            "total_decisions": self.decision_count,
            "total_reflections": self.reflection_count,
            "final_state": final_state,
            "summary_stats": summary_stats,
        }
        self.trace_f.write(_safe_dump(record) + "\n")
        
        # 关闭
        self.trace_f.close()
        self.memory_f.close()
        self.reflection_f.close()
        self.timeline_f.close()
        
        print(f"TraceLogger finalized for user {self.user_id}:")
        print(f"  {self.trace_path}")
        print(f"  {self.memory_path}")
        print(f"  {self.reflection_path}")
        print(f"  {self.timeline_path}")
