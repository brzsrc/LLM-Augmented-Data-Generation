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
# 第二部分：记忆流（不变）
# ================================================================

class Memory:
    def __init__(self, timestamp: str, content: str, mem_type: str, importance: int):
        self.timestamp = timestamp
        self.content = content
        #reflection, observation, background
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
    ts = f"Day{day}_Slot{slot}"
    obs = f"[{ts}] Location: {ctx['location']}. "
    obs += f"Weather: {ctx['weather']}, {ctx['temperature']}. "
    obs += f"Steps in prior 30min: {ctx['prior_30min_steps']}. Activity: {ctx['activity']}. "

    if action == 1:
        obs += "System sent an ACTIVE walking suggestion. "
        obs += f"Steps in following 30min: {steps} after suggestion. "
        if ctx.get('response'):
            obs += f"Participant response: {ctx['response']}. "
    elif action == 2:
        obs += "System sent a SEDENTARY/stand-up suggestion. "
        obs += f"Steps in following 30min: {steps}. "
        if ctx.get('response'):
            obs += f"Participant response: {ctx['response']}. "
    else:
        obs += f"No suggestion sent. Steps in following 30min: {steps}. "

    return obs
