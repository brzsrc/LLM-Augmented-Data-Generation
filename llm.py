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
        self.score_guided = StructuredOutputsParams(choice=["1", "2", "3", "4", "5"])
        self.score_params = SamplingParams(temperature=0.3, max_tokens=1, structured_outputs=self.score_guided)
        self.importance_guided = StructuredOutputsParams(choice=[str(i) for i in range(1, 11)])
        self.importance_params = SamplingParams(temperature=0.1, max_tokens=2, structured_outputs=self.importance_guided)
        self.text_params = SamplingParams(temperature=0.3, max_tokens=500)
        # Plan B: absolute step prediction.
        # 关键修复: 用 structured_outputs 强制输出整数 (避免模型先吐换行/标签前缀),
        # 同时 max_tokens 给足 (5 位数 最长 4 token + 安全余量)
        # 用稀疏 choice set 加速 vLLM constrained decoding 构建:
        #   0-999 step 全保留, 1000-3000 step 2 间隔, 3000-9999 step 100 间隔
        steps_choices = (
            [str(i) for i in range(0, 1000)] +
            [str(i) for i in range(1000, 3000, 5)] +
            [str(i) for i in range(3000, 10000, 100)]
        )
        self.steps_guided = StructuredOutputsParams(choice=steps_choices)
        self.steps_params = SamplingParams(temperature=0.3, max_tokens=8,
                                           structured_outputs=self.steps_guided)
        # 调试用: 跑头几次时把原始 text 存一下方便诊断
        self.debug_steps_outputs = []
        self.debug_capture_n = 20  # 只存前 20 次的原始输出
        self.call_count = 0
        print(f"Model loaded! steps choices = {len(steps_choices)}")

    def _prompt(self, system: str, user: str) -> str:
        return (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n/no_think\n")

    def score_importance(self, system, user):
        out = self.llm.generate([self._prompt(system, user)], self.importance_params)
        self.call_count += 1;
        return out[0].outputs[0].text.strip()

    def generate_text(self, system, user):
        out = self.llm.generate([self._prompt(system, user)], self.text_params)
        self.call_count += 1;
        return out[0].outputs[0].text.strip()

    def batch_score(self, prompts):
        fmt = [self._prompt(p["system"], p["user"]) for p in prompts]
        out = self.llm.generate(fmt, self.score_params)
        self.call_count += len(prompts);
        return [o.outputs[0].text.strip() for o in out]

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
        self.call_count += len(prompts);
        return [o.outputs[0].text.strip() for o in out]

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
        """Robust parse: take leading digits, clip to [0, 9999]."""
        s = text.strip()
        # leading digits only (may be followed by garbage)
        n = 0
        for ch in s:
            if ch.isdigit():
                n = n * 10 + int(ch)
                if n > 9999:
                    return 9999
            else:
                break
        return min(n, 9999)

    def batch_text(self, prompts):
        fmt = [self._prompt(p["system"], p["user"]) for p in prompts]
        out = self.llm.generate(fmt, self.text_params)
        self.call_count += len(prompts);
        return [o.outputs[0].text.strip() for o in out]


class SimulatedLLM:
    def __init__(self): self.call_count = 0

    def score_importance(self, s, u): self.call_count += 1; return str(np.random.randint(3, 8))

    def generate_text(self, s, u): self.call_count += 1; return "Simulated inference."

    def batch_score(self, p): self.call_count += len(p); return [str(np.random.randint(1, 6)) for _ in p]

    def judge_steps(self, s, u): self.call_count += 1; return int(np.random.lognormal(4.5, 1.2))

    def batch_importance(self, p): self.call_count += len(p); return [str(np.random.randint(3, 8)) for _ in p]

    def batch_steps(self, p): self.call_count += len(p); return [int(np.random.lognormal(4.5, 1.2)) for _ in p]

    def batch_text(self, p): self.call_count += len(p); return ["Simulated inference." for _ in p]
