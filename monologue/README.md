# Qwen3-32B HeartSteps 心理揣测预测 (v3:支持任意用户数)

## 改造点(v3)

相比 v2,核心变化:
1. **uid 不再硬编码**——所有脚本支持 `--uids` / `--uids-file` / CSV 自动全集
2. **`common.py` 模块**——所有 prompt 拼装、文本解析、UserState 等共享代码集中
3. **`--max-batch` 限制**——用户数变大时,单 batch 太大会爆显存,可手动切片
4. **缺画像/缺数据用户自动跳过**——不会因为某个用户出问题挂掉整轮
5. **`step3 --all-from-jsonl`**——自动发现哪些用户已经跑完,合并它们

## 文件结构

```
qwen_v3/
├── common.py                          ← 共享工具,所有 step 引用
├── llm.py                             ← 你的 Qwen32BLLM (或测试桩)
├── data/cleaned_output_to_predict.csv ← 输入数据
├── prompts/                           ← 4 个 prompt 模板,无需改
│   ├── step1_system.txt
│   ├── step1_user.txt
│   ├── step2_monologue_system.txt
│   ├── step2_monologue_user.txt
│   ├── step2_post30_system.txt
│   └── step2_post30_user.txt
├── outputs/
│   ├── personas/user_*.md             ← step1 产物
│   ├── monologues/user_*.jsonl        ← step2 产物
│   └── predictions.csv                ← step3 产物
├── step1_build_persona.py
├── step2_predict_rows.py              ← 串行版(调试用)
├── step2_predict_rows_parallel.py     ← 并行版(主推)
└── step3_merge_csv.py
```

## 三种 uid 指定方式(所有 step 通用)

```bash
# 1. 跑 CSV 中全部用户(默认)
python step1_build_persona.py

# 2. 指定几个用户
python step1_build_persona.py --uids 1 11 37

# 3. 从 JSON 文件读 uid 列表
echo '[1, 11, 37, 42]' > train_uids.json
python step1_build_persona.py --uids-file train_uids.json
```

## 完整流程示例

### A. 跑 User 1, 11, 37 三个用户(等价于 v2 默认行为)

```bash
python step1_build_persona.py --uids 1 11 37
python step2_predict_rows_parallel.py --uids 1 11 37
python step3_merge_csv.py --uids 1 11 37
```

### B. 跑训练集全部用户(假设 train_uids.json 有 30 个 uid)

```bash
python step1_build_persona.py --uids-file train_uids.json
python step2_predict_rows_parallel.py --uids-file train_uids.json --max-batch 16
python step3_merge_csv.py --uids-file train_uids.json
```

### C. 跑 CSV 全部用户(自动发现)

```bash
python step1_build_persona.py
python step2_predict_rows_parallel.py --max-batch 8
python step3_merge_csv.py --all-from-jsonl
```

## --max-batch 怎么选

vLLM 的 continuous batching 默认会自己控制并发量,但 prompt 数量过多时:
- 30 个用户同时构造独白 prompt → 单次输入 token 总量 ≈ 30 × 25k = 750k
- max_model_len=32k 时,vLLM 内部会自动分批,但 prompt prefill 阶段显存压力大

经验值:
- **10 用户以内**:`--max-batch 0`(无限制,vLLM 自动)
- **10-30 用户**:`--max-batch 8 ~ 16`
- **30+ 用户**:`--max-batch 4 ~ 8`
- 出现 OOM 时,直接减半

## 跑之前必须确认的 3 件事(同 v2)

1. **`max_model_len=32768`**(你接口默认 8192,装不下)
2. **改 `from llm import Qwen32BLLM`** 成你项目实际路径
3. **脚本会自动调** `text_params_no_think.max_tokens=600`(默认 400 容易截断)

## 断点续跑 + 不同步用户

- 每个用户独立写 jsonl(append-only),中断后再跑会跳过已完成行
- 如果不同用户的 jsonl 进度不一致(罕见,比如手动删了某个文件),并行版会**自动串行追平**落后用户,然后进入并行模式
- 这意味着可以**逐步添加新用户**:已经跑过的不重做,新加的从头开始

```bash
# 第一天跑了 3 个
python step2_predict_rows_parallel.py --uids 1 11 37

# 第二天想加 5 个新用户
python step2_predict_rows_parallel.py --uids 1 11 37 42 55 66 77 88
# 已有的 1/11/37 完全跳过,新加的 5 个会被串行追平后进入并行
```

## 性能估算

| 配置 | rows | 时间(no_think) | 时间(thinking) |
|---|---|---|---|
| 3 用户 569 行 | 569 | ~35 min | ~2 hr |
| 10 用户 ~2000 行 | 2000 | ~70 min | ~4 hr |
| 30 用户 ~6000 行 | 6000 | ~3 hr | ~10 hr |

(线性 scale,因为 vLLM batch 加速主要来自跨用户并行)
