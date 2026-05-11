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
# 第四部分：LLM 接口（不变）
# ================================================================

class Qwen3BLLM:
    def __init__(self, model_path: str = "../models/Qwen3-8B-AWQ"):
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        print(f"Loading model: {model_path}")
        self.llm = LLM(model=model_path, quantization="awq",
                       gpu_memory_utilization=0.90, max_model_len=4096,
                       trust_remote_code=True)
        self.score_guided = StructuredOutputsParams(choice=[f"##{i}##" for i in range(1, 6)])
        self.score_params = SamplingParams(temperature=0.3, max_tokens=4, structured_outputs=self.score_guided)
        self.importance_guided = StructuredOutputsParams(choice=[f"##{i}##" for i in range(1, 11)])
        self.importance_params = SamplingParams(temperature=0.1, max_tokens=4, structured_outputs=self.importance_guided)
        self.text_params = SamplingParams(temperature=0.3, max_tokens=500)
        # Plan B (ARMMAN-style): LLM 输出 ##N## 包裹的步数, 例如 ##250##.
        # structured_outputs 强制采 choice list 中的一个完整字符串.
        # 稀疏 choice 控制规模 (4-digit max + ## wrapper, 单选 7-8 字符):
        #   0-999 全保留, 1000-3000 每 5 步, 3000-9999 每 100 步
        steps_choices = (
            [f"##{i}##" for i in range(0, 1000)] +
            [f"##{i}##" for i in range(1000, 3000, 5)] +
            [f"##{i}##" for i in range(3000, 10000, 100)]
        )
        self.steps_guided = StructuredOutputsParams(choice=steps_choices)
        # ##N## 最长是 ##9900## = 8 字符 ≈ 5-6 token, max_tokens=12 给足安全余量
        self.steps_params = SamplingParams(temperature=0.3, max_tokens=12,
                                           structured_outputs=self.steps_guided)
        # 调试用: 跑头几次时把原始 text 存一下方便诊断
        self.debug_steps_outputs = []
        self.debug_capture_n = 20
        self.call_count = 0
        print(f"Model loaded! steps choices = {len(steps_choices)}")

    def _prompt(self, system: str, user: str) -> str:
        return (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n/no_think\n")

    def score_importance(self, system, user):
        out = self.llm.generate([self._prompt(system, user)], self.importance_params)
        self.call_count += 1
        return self._parse_int(out[0].outputs[0].text, lo=1, hi=10, default=5)

    def generate_text(self, system, user):
        out = self.llm.generate([self._prompt(system, user)], self.text_params)
        self.call_count += 1;
        return out[0].outputs[0].text.strip()

    def batch_score(self, prompts):
        fmt = [self._prompt(p["system"], p["user"]) for p in prompts]
        out = self.llm.generate(fmt, self.score_params)
        self.call_count += len(prompts)
        return [self._parse_int(o.outputs[0].text, lo=1, hi=5, default=3) for o in out]

    def judge_steps(self, system, user):
        out = self.llm.generate([self._prompt(system, user)], self.steps_params)
        self.call_count += 1
        raw = out[0].outputs[0].text
        if len(self.debug_steps_outputs) < self.debug_capture_n:
            self.debug_steps_outputs.append(raw)
        return self._parse_steps(raw)

    def batch_importance(self, prompts):
        fmt = [self._prompt(p["system"], p["user"]) for p in prompts]
        out = self.llm.generate(fmt, self.importance_params)
        self.call_count += len(prompts)
        return [self._parse_int(o.outputs[0].text, lo=1, hi=10, default=5) for o in out]

    def batch_steps(self, prompts):
        fmt = [self._prompt(p["system"], p["user"]) for p in prompts]
        out = self.llm.generate(fmt, self.steps_params)
        self.call_count += len(prompts)
        results = []
        for o in out:
            raw = o.outputs[0].text
            if len(self.debug_steps_outputs) < self.debug_capture_n:
                self.debug_steps_outputs.append(raw)
            results.append(self._parse_steps(raw))
        return results

    @staticmethod
    def _parse_steps(text: str) -> int:
        """Robust parse: handle ##N## wrapper or any text with digits.
        Strategy: extract first contiguous digit run, clip to [0, 9999].
        Examples:
          '##250##' → 250
          '250'     → 250
          '##0##'   → 0
          '## 0 ##' → 0  (whitespace tolerant)
          'I think ##300##' → 300
          ''        → 0
        """
        s = text.strip()
        # find first digit
        n = 0
        found_digit = False
        for ch in s:
            if ch.isdigit():
                found_digit = True
                n = n * 10 + int(ch)
                if n > 9999:
                    return 9999
            elif found_digit:
                # past the first digit run, stop
                break
        return min(n, 9999) if found_digit else 0

    @staticmethod
    def _parse_int(text: str, lo: int, hi: int, default: int) -> int:
        """Parse first integer from text (handles ##N## or bare digits),
        clip to [lo, hi]. Returns `default` if no digits found."""
        s = text.strip()
        n = 0
        found = False
        for ch in s:
            if ch.isdigit():
                found = True
                n = n * 10 + int(ch)
                if n > hi:
                    return hi
            elif found:
                break
        if not found:
            return default
        return max(lo, min(hi, n))

    def batch_text(self, prompts):
        fmt = [self._prompt(p["system"], p["user"]) for p in prompts]
        out = self.llm.generate(fmt, self.text_params)
        self.call_count += len(prompts);
        return [o.outputs[0].text.strip() for o in out]


class SimulatedLLM:
    def __init__(self): self.call_count = 0

    def score_importance(self, s, u): self.call_count += 1; return int(np.random.randint(3, 8))

    def generate_text(self, s, u): self.call_count += 1; return "Simulated inference."

    def batch_score(self, p): self.call_count += len(p); return [int(np.random.randint(1, 6)) for _ in p]

    def judge_steps(self, s, u): self.call_count += 1; return int(np.random.lognormal(4.5, 1.2))

    def batch_importance(self, p): self.call_count += len(p); return [int(np.random.randint(3, 8)) for _ in p]

    def batch_steps(self, p): self.call_count += len(p); return [int(np.random.lognormal(4.5, 1.2)) for _ in p]

    def batch_text(self, p): self.call_count += len(p); return ["Simulated inference." for _ in p]
