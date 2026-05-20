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
        self.score_params = SamplingParams(temperature=0.7, max_tokens=4, structured_outputs=self.score_guided)

        self.importance_guided = StructuredOutputsParams(choice=[f"##{i}##" for i in range(1, 11)])
        self.importance_params = SamplingParams(temperature=0.1, max_tokens=4, structured_outputs=self.importance_guided)

        self.text_params = SamplingParams(temperature=0.7, max_tokens=500)
        steps_choices = (
            [f"##{i}##" for i in range(0, 1000)] +
            [f"##{i}##" for i in range(1000, 3000, 5)] +
            [f"##{i}##" for i in range(3000, 10000, 100)]
        )
        self.steps_guided = StructuredOutputsParams(choice=steps_choices)
        # ##N## 最长是 ##9900## = 8 字符 ≈ 5-6 token, max_tokens=12 给足安全余量
        self.steps_params = SamplingParams(temperature=0.7, max_tokens=12,
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

    def generate_text(self, system, user, thinking: bool=False):
        out = self.llm.generate([self._prompt(system, user)], self.text_params)
        self.call_count += 1
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

    def batch_text(self, prompts, thinking: bool=False):
        fmt = [self._prompt(p["system"], p["user"]) for p in prompts]
        out = self.llm.generate(fmt, self.text_params)
        self.call_count += len(prompts)
        return [o.outputs[0].text.strip() for o in out]

# ================================================================
# Qwen3-32B-AWQ
# ================================================================

class Qwen32BLLM:
    def __init__(self, model_path: str = "../models/Qwen3-32B-AWQ"):
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        print(f"Loading model: {model_path}")
        self.llm = LLM(
            model=model_path,
            quantization="awq_marlin",         # ← CHANGED: 从 "awq" 改 "awq_marlin",更快
            dtype="float16",                    # ← NEW: 明确 fp16
            gpu_memory_utilization=0.90,
            max_model_len=8192,                 # ← CHANGED: 从 4096 提到 8192 (reassess prompt 可能到 2K)
            trust_remote_code=True,
            # enable_prefix_caching=True,  # ← 加这行
            # enable_chunked_prefill=True,  # ← 加这行
            tensor_parallel_size=2,
        )

        # score (1-5): 反思打分用
        self.score_guided = StructuredOutputsParams(choice=[f"##{i}##" for i in range(1, 6)])
        self.score_params = SamplingParams(
            temperature=0.7, top_p=0.8, top_k=20,    # ← NEW: Qwen3 官方推荐采样参数
            max_tokens=4,
            structured_outputs=self.score_guided
        )

        # importance (1-10): 单点 importance 打分用
        self.importance_guided = StructuredOutputsParams(choice=[f"##{i}##" for i in range(1, 11)])
        self.importance_params = SamplingParams(
            temperature=0.1, top_p=0.8, top_k=20,    # ← NEW
            max_tokens=4,
            structured_outputs=self.importance_guided
        )

        # # text: 反思生成、reassess 重打分都用这个
        # self.text_params = SamplingParams(
        #     temperature=0.7, top_p=0.8, top_k=20,    # ← NEW
        #     max_tokens=1000
        # )

        # ⚠️ thinking 任务用大 max_tokens
        self.text_params_thinking = SamplingParams(
            temperature=0.7, top_p=0.8, top_k=20,
            max_tokens=2048,  # ← thinking 需要足够空间
        )

        # ⚠️ no-think 任务用小 max_tokens
        self.text_params_no_think = SamplingParams(
            temperature=0.7, top_p=0.8, top_k=20,
            max_tokens=300,  # ← 直接答案,不需要太长
        )

        # steps choices (跟 8B 完全一样,不动)
        steps_choices = (
            [f"##{i}##" for i in range(0, 1000)] +
            [f"##{i}##" for i in range(1000, 3000, 5)] +
            [f"##{i}##" for i in range(3000, 10000, 100)]
        )
        self.steps_guided = StructuredOutputsParams(choice=steps_choices)
        self.steps_params = SamplingParams(
            temperature=0.7, top_p=0.8, top_k=20,    # ← NEW
            max_tokens=12,
            structured_outputs=self.steps_guided
        )

        self.debug_steps_outputs = []
        self.debug_capture_n = 20
        self.call_count = 0
        print(f"Model loaded! steps choices = {len(steps_choices)}")

    # def _prompt(self, system: str, user: str) -> str:
    #     # Qwen3-32B 支持 /no_think 模式开关,跟 8B 一样保留
    #     return (f"<|im_start|>system\n{system}<|im_end|>\n"
    #             f"<|im_start|>user\n{user}<|im_end|>\n"
    #             f"<|im_start|>assistant\n/no_think\n")

    # 两种 prompt 风格
    def _prompt_thinking(self, system: str, user: str) -> str:
        """强制 LLM 进入 thinking 模式 — 预填 <think>"""
        return (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n")

    def _prompt_no_think(self, system: str, user: str) -> str:
        """跳过 thinking — 预填空 <think></think>"""
        return (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def score_importance(self, system, user):
        out = self.llm.generate([self._prompt_no_think(system, user)], self.importance_params)
        self.call_count += 1
        return self._parse_int(out[0].outputs[0].text, lo=1, hi=10, default=5)

    def generate_text(self, system, user, thinking: bool=False):
        """
        thinking=False (default): 跳过思考, 适合 questions 生成、短答案
        thinking=True: 允许思考, 适合 reflection inference 和 reassess
        """
        if thinking:
            out = self.llm.generate([self._prompt_thinking(system, user)], self.text_params_thinking)
        else:
            out = self.llm.generate([self._prompt_no_think(system, user)], self.text_params_no_think)
        self.call_count += 1
        return out[0].outputs[0].text.strip()

    def batch_score(self, prompts):
        fmt = [self._prompt_no_think(p["system"], p["user"]) for p in prompts]
        out = self.llm.generate(fmt, self.score_params)
        self.call_count += len(prompts)
        return [self._parse_int(o.outputs[0].text, lo=1, hi=5, default=3) for o in out]

    def judge_steps(self, system, user):
        out = self.llm.generate([self._prompt_no_think(system, user)], self.steps_params)
        self.call_count += 1
        raw = out[0].outputs[0].text
        if len(self.debug_steps_outputs) < self.debug_capture_n:
            self.debug_steps_outputs.append(raw)
        return self._parse_steps(raw)

    def batch_importance(self, prompts):
        fmt = [self._prompt_no_think(p["system"], p["user"]) for p in prompts]
        out = self.llm.generate(fmt, self.importance_params)
        self.call_count += len(prompts)
        return [self._parse_int(o.outputs[0].text, lo=1, hi=10, default=5) for o in out]

    def batch_steps(self, prompts):
        fmt = [self._prompt_no_think(p["system"], p["user"]) for p in prompts]
        out = self.llm.generate(fmt, self.steps_params)
        self.call_count += len(prompts)
        results = []
        for o in out:
            raw = o.outputs[0].text
            if len(self.debug_steps_outputs) < self.debug_capture_n:
                self.debug_steps_outputs.append(raw)
            results.append(self._parse_steps(raw))
        return results

        # text 任务: 有两个版本 - thinking 和 no_think
    def batch_text(self, prompts, thinking: bool=False):
        """
        thinking=False (default): 跳过思考, 适合 questions 生成、短答案
        thinking=True: 允许思考, 适合 reflection inference 和 reassess
        """
        if thinking:
            fmt = [self._prompt_thinking(p["system"], p["user"]) for p in prompts]
            params = self.text_params_thinking
        else:
            fmt = [self._prompt_no_think(p["system"], p["user"]) for p in prompts]
            params = self.text_params_no_think

        out = self.llm.generate(fmt, params)
        self.call_count += len(prompts)
        return [o.outputs[0].text.strip() for o in out]

    # parse 方法跟 8B 完全一样,直接复用
    @staticmethod
    def _parse_steps(text: str) -> int:
        s = text.strip()
        n = 0
        found_digit = False
        for ch in s:
            if ch.isdigit():
                found_digit = True
                n = n * 10 + int(ch)
                if n > 9999:
                    return 9999
            elif found_digit:
                break
        return min(n, 9999) if found_digit else 0

    @staticmethod
    def _parse_int(text: str, lo: int, hi: int, default: int) -> int:
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




class SimulatedLLM:
    def __init__(self): self.call_count = 0

    def score_importance(self, s, u): self.call_count += 1; return int(np.random.randint(3, 8))

    def generate_text(self, s, u, thinking: bool=False): self.call_count += 1; return "Simulated inference."

    def batch_score(self, p): self.call_count += len(p); return [int(np.random.randint(1, 6)) for _ in p]

    def judge_steps(self, s, u): self.call_count += 1; return int(np.random.lognormal(4.5, 1.2))

    def batch_importance(self, p): self.call_count += len(p); return [int(np.random.randint(3, 8)) for _ in p]

    def batch_steps(self, p): self.call_count += len(p); return [int(np.random.lognormal(4.5, 1.2)) for _ in p]

    def batch_text(self, p, thinking: bool=False): self.call_count += len(p); return ["Simulated inference." for _ in p]


if __name__ == '__main__':
    llm = Qwen32BLLM(model_path="/root/autodl-tmp/models/Qwen3-32B-AWQ")
    # 验证 1: structured 输出
    score = llm.score_importance(
        "You score events 1-10.",
        "An event: walked 1500 steps after a sedentary morning. Score: ##N##"
    )
    print(f"score: {score}")  # 应该是 1-10 的数字

    # 验证 2: text 输出 (reassess 用的)
    texts = llm.batch_text([{
        "system": "Output 50 integers comma-separated wrapped in ##...##.",
        "user": "Generate 50 random integers 1-10. Output: ##N1,N2,N3,N4,N5...##, where each Ni is an integer 1-10"
    }])
    print(f"text: {texts[0]}")  # 应该形如 "##3,7,2,9,4##"