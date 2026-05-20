from typing import List, Optional
import warnings

warnings.filterwarnings('ignore')


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

    def get_new_observations_since_last_reflection(self,
                                                    fallback_n: int = 15) -> List[Memory]:
        """Observations added since the most recent reflection memory.

        Used as the 'new evidence' input to reflection reassessment.
        If there is no prior reflection (this is the first one), fall back
        to the most recent `fallback_n` observations.
        """
        # 找最后一条 reflection 的位置
        last_refl_idx = -1
        for i, m in enumerate(self.memories):
            if m.mem_type == "reflection":
                last_refl_idx = i

        if last_refl_idx == -1:
            # 还没有任何反思,返回最近 N 条 observation 作 fallback
            return [m for m in self.memories
                    if m.mem_type == "observation"][-fallback_n:]

        # 取 last_refl_idx 之后的所有 observation
        return [m for m in self.memories[last_refl_idx + 1:]
                if m.mem_type == "observation"]

    def reset_reflection_counter(self):
        self.importance_since_reflection = 0

    def get_recent(self, n: Optional[int] = 10, mem_type: Optional[str] = None) -> List[Memory]:
        filtered = self.memories if mem_type is None else [m for m in self.memories if m.mem_type == mem_type]
        if n is None:
            return filtered
        return filtered[-n:]

    def format_memories(self, memories: List[Memory]) -> str:
        return "\n".join(f"[{m.timestamp}] {m.content}" for m in memories)

    def get_recent_weighted_obs(self, n: int = 10,
                            importance_floor: int = 7,
                            recent_window: int = 20) -> List[Memory]:
        """Importance-aware retrieval for step generation and reflection prompts.

        Strategy:
          1. Take ALL observations from the whole history with importance >=
             importance_floor (these are "permanent highlights" — high-signal
             past events that should never be forgotten).
          2. Fill remaining slots with the most recent observations not
             already chosen.
          3. Preserve chronological order in the returned list (LLMs read
             time-ordered memories more easily).

        If no memory has been re-scored yet, behaves close to get_recent.
        """
        all_obs = [m for m in self.memories if m.mem_type == 'observation']
        if not all_obs:
            return []

        # 1. 高重要性事件 (跨整个历史)
        highlights = [m for m in all_obs if m.importance >= importance_floor]

        # 高重要性事件已经够多 → 按时间序取最近的 n 个
        if len(highlights) >= n:
            return highlights[-n:]

        # 2. 补充最近的低重要性事件 (跳过已选 highlights)
        chosen_ids = {id(m) for m in highlights}
        recent_pool = all_obs[-recent_window:]
        fillers = [m for m in recent_pool if id(m) not in chosen_ids]
        needed = n - len(highlights)
        fillers = fillers[-needed:] if needed > 0 else []

        # 3. 按时间序合并 (memories 列表本身是按插入序的, 所以用 index 排)
        all_idx = {id(m): i for i, m in enumerate(all_obs)}
        chosen = highlights + fillers
        chosen.sort(key=lambda m: all_idx[id(m)])
        return chosen

    def get_recent_weighted_ref(self, n: int = 3,
                            reflection_floor: int = 5) -> List[Memory]:
        """Importance-aware retrieval.

        kinds: which memory types to consider as candidates
        reflection_floor: if 'reflection' in kinds, drop reflections below this
        """
        # 反思的过滤逻辑: 低于 floor 直接淘汰
        all_candidates = [m for m in self.memories if m.mem_type == 'reflection' and m.importance >= reflection_floor]
        if not all_candidates:
            return []

        if len(all_candidates) >= n:
            return all_candidates[-n:]

        return all_candidates

def create_observation(day: int, slot: int, ctx: dict, action: int, steps: int) -> str:
    ts = f"Day{day}_Slot{slot}"
    obs = f"[{ts}] Location: {ctx['location']}. "
    obs += f"Weather: {ctx['weather']}, {ctx['temperature']}. "
    obs += f"Steps in prior 30min: {ctx['prior_30min_steps']}. Activity: {ctx['activity']}. "

    # Map raw response label to clearer phrasing that won't confuse the LLM
    response_map = {
        'good': 'I rated the suggestion as helpful',
        'bad':  'I rated the suggestion as unhelpful',
        'no_response': 'I did not rate the suggestion',
    }

    if action == 1:
        obs += "System sent an ACTIVE walking suggestion. "
        obs += f"Steps in following 30min: {steps}. "
        if ctx.get('response'):
            desc = response_map.get(ctx['response'], ctx['response'])
            obs += f"My rating of the suggestion: {desc} (this is my subjective opinion, NOT a measure of how much I walked). "
    elif action == 2:
        obs += "System sent a SEDENTARY/stand-up suggestion. "
        obs += f"Steps in following 30min: {steps}. "
        if ctx.get('response'):
            desc = response_map.get(ctx['response'], ctx['response'])
            obs += f"My rating of the suggestion: {desc} (this is my subjective opinion, NOT a measure of how much I walked). "
    else:
        obs += f"No suggestion sent. Steps in following 30min: {steps}. "

    return obs