"""
公用工具:
- resolve_uids(): 从命令行参数 / 配置文件 / CSV 自动解析 uid 列表
- paths: 统一的目录结构
- UserState: 并行版用的用户状态类
- 文本解析、画像原则提取、历史滑窗

被 step1 / step2 / step3 同时引用,避免 uid 列表在 4 个地方重复定义。
"""

import json
import os
import re
from pathlib import Path

import pandas as pd

# ============================================
# 路径(全部相对当前工作目录)
# ============================================
DEFAULT_INPUT_CSV = "data/cleaned_output_to_predict.csv"
DEFAULT_PERSONA_DIR = "evaluation/outputs/personas"
DEFAULT_MONOLOGUE_DIR = "evaluation/outputs/monologues"
DEFAULT_PREDICTIONS_CSV = "outputs/predictions.csv"

PROMPT_PATHS = {
    "step1_sys": "prompts/step1_system.txt",
    "step1_usr": "prompts/step1_user.txt",
    "mono_sys": "prompts/step2_monologue_system.txt",
    "mono_usr": "prompts/step2_monologue_user.txt",
    "post_sys": "prompts/step2_post30_system.txt",
    "post_usr": "prompts/step2_post30_user.txt",
}

HISTORY_FULL_RECENT = 20
DOW_MAP_CN = {0: "周一", 1: "周二", 2: "周三", 3: "周四",
              4: "周五", 5: "周六", 6: "周日"}


# ============================================
# UID 解析:支持 4 种来源,按优先级
# ============================================
def resolve_uids(
    cli_uids: list = None,
    uids_file: str = None,
    csv_path: str = DEFAULT_INPUT_CSV,
    mode: str = "all",
) -> list:
    """
    决定要处理哪些 uid。优先级:cli > uids_file > csv 全集

    参数
    ----
    cli_uids: 命令行直接传的 list,如 [1, 11, 37]
    uids_file: 一个 json 文件路径,内容应是 [int, int, ...]
    csv_path: 输入 CSV 路径,用于 'all' 模式取全集
    mode: 'all' = csv 中全部 uid,'list' = 用 cli_uids 或 uids_file

    返回
    ----
    sorted 升序 uid list (int)
    """
    if cli_uids is not None and len(cli_uids) > 0:
        uids = sorted(set(int(u) for u in cli_uids))
        print(f"[uids] from CLI: {len(uids)} users -> {uids[:10]}{'...' if len(uids)>10 else ''}")
        return uids

    if uids_file is not None and os.path.exists(uids_file):
        with open(uids_file) as f:
            raw = json.load(f)
        uids = sorted(set(int(u) for u in raw))
        print(f"[uids] from file {uids_file}: {len(uids)} users")
        return uids

    # 默认:从 csv 取全部
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")
    df = pd.read_csv(csv_path, usecols=["uid"])
    uids = sorted(set(int(u) for u in df["uid"].unique()))
    print(f"[uids] from CSV {csv_path}: {len(uids)} users")
    return uids


def add_uid_args(parser):
    """给 argparse 加统一的 uid 参数。3 个文件都用它,行为一致。"""
    parser.add_argument(
        "--uids", type=int, nargs="+", default=None,
        help="直接指定 uid 列表,如 --uids 1 11 37"
    )
    parser.add_argument(
        "--uids-file", default=None,
        help="JSON 文件路径,内容是 uid 列表(如 [1,11,37])"
    )
    parser.add_argument(
        "--input-csv", default=DEFAULT_INPUT_CSV,
        help="输入数据 CSV 路径"
    )
    parser.add_argument(
        "--persona-dir", default=DEFAULT_PERSONA_DIR,
        help="画像输出/读取目录"
    )
    parser.add_argument(
        "--monologue-dir", default=DEFAULT_MONOLOGUE_DIR,
        help="逐行揣测 jsonl 输出目录"
    )
    return parser


# ============================================
# 文本解析:独白输出 / 画像原则 / 历史滑窗
# ============================================
def strip_think_tags(text: str) -> str:
    """处理 thinking 模式的 <think>...</think> 残留(no_think 模式也兼容)。"""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def parse_monologue_output(text: str) -> dict:
    """从 LLM 输出抽取 MONOLOGUE / REASONING / SUMMARY 三段。"""
    sections = {"MONOLOGUE": "", "REASONING": "", "SUMMARY": ""}
    text = strip_think_tags(text)
    pat = re.compile(
        r"###\s*(MONOLOGUE|REASONING|SUMMARY)\s*\n(.*?)(?=###\s*(?:MONOLOGUE|REASONING|SUMMARY)|\Z)",
        re.DOTALL,
    )
    for m in pat.finditer(text):
        sections[m.group(1).strip()] = m.group(2).strip()
    return sections


def extract_principles(persona: str) -> str:
    """从画像中只提取第 6 部分(预测原则)。"""
    m = re.search(r"###\s*6[\.、][^\n]*\n(.*?)(?=\n###|\Z)", persona, re.DOTALL)
    return m.group(1).strip() if m else persona


def build_history(prior_rows: list) -> str:
    """近 K 行带完整独白,更早行只带 SUMMARY。"""
    if not prior_rows:
        return "(这是该用户的第一个时段,还没有之前的揣测。)"
    n = len(prior_rows)
    early = prior_rows[: max(0, n - HISTORY_FULL_RECENT)]
    recent = prior_rows[max(0, n - HISTORY_FULL_RECENT):]
    lines = []
    if early:
        lines.append("【早期时段摘要】")
        for r in early:
            lines.append(
                f"- d{r['study_day']} s{r['slot']}: {r['summary']} "
                f"(pre30={r['pre30']:.0f}, 预测post30={r['post30']})"
            )
        lines.append("")
    if recent:
        lines.append(f"【最近 {len(recent)} 个时段(完整独白)】")
        for r in recent:
            lines.append(
                f"--- d{r['study_day']} s{r['slot']} "
                f"(loc={r['location']}, send={r['send']}, resp={r['response']}, "
                f"pre30={r['pre30']:.0f}, 预测post30={r['post30']}) ---"
            )
            lines.append(r["monologue"])
            lines.append("")
    return "\n".join(lines)


def build_rows_table(df: pd.DataFrame) -> str:
    """把一个用户的数据压缩成紧凑文本表,给 step1 画像 prompt 用。"""
    df = df.sort_values(["study_day", "day_slot"]).reset_index(drop=True).copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek
    dow_en = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    lines = []
    for _, r in df.iterrows():
        lines.append(
            f"d{int(r['study_day']):>2} {dow_en[r['dow']]} "
            f"s{int(r['day_slot'])}({int(r['hour']):02d}h) | "
            f"loc={r['location']:<14} | act={r['activity']:<7} | "
            f"wx={r['weather']:<12} {r['temperature']:<13} | "
            f"send={int(r['send'])} avail={r['avail']} rand={r['is_randomized']} | "
            f"resp={r['response']:<11} | pre30={r['jbsteps30pre']:.0f}"
        )
    return "\n".join(lines)


# ============================================
# 用户状态(并行版用)
# ============================================
class UserState:
    """一个用户的运行时状态:数据 + 已揣测的 prior_rows + 输出 jsonl 句柄。"""

    def __init__(self, uid: int, sub_df: pd.DataFrame, persona: str, jsonl_path: str):
        self.uid = uid
        self.persona = persona
        self.principles = extract_principles(persona)

        sub_df = sub_df.copy()
        sub_df["datetime"] = pd.to_datetime(sub_df["datetime"])
        self.df = sub_df.sort_values(["study_day", "day_slot"]).reset_index(drop=True)
        self.N = len(self.df)

        self.prior_rows = []
        self.done_keys = set()
        self.jsonl_path = jsonl_path

        # 断点续跑
        if os.path.exists(jsonl_path):
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    self.prior_rows.append(rec)
                    self.done_keys.add((rec["study_day"], rec["slot"]))

        self.cursor = len(self.prior_rows)
        self.f_out = None

    def open_out(self):
        os.makedirs(os.path.dirname(self.jsonl_path) or ".", exist_ok=True)
        self.f_out = open(self.jsonl_path, "a", encoding="utf-8")

    def close_out(self):
        if self.f_out:
            self.f_out.close()
            self.f_out = None

    def current_row(self):
        if self.cursor >= self.N:
            return None
        return self.df.iloc[self.cursor]

    def append_result(self, rec: dict):
        self.f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f_out.flush()
        self.prior_rows.append(rec)
        self.cursor += 1


# ============================================
# 加载 + 校验:给定 uids 列表,返回每个能加载的 (uid, sub_df, persona_path)
# ============================================
def load_users_for_prediction(
    uids: list, input_csv: str, persona_dir: str,
    skip_missing_persona: bool = True,
) -> list:
    """
    返回 list of (uid, sub_df, persona_text)。
    自动跳过:
      - 在 CSV 中没数据的 uid
      - 没有画像文件的 uid(如果 skip_missing_persona=True)
    """
    df = pd.read_csv(input_csv)
    available_uids = set(df["uid"].unique())

    out = []
    skipped_no_data = []
    skipped_no_persona = []

    for uid in uids:
        if uid not in available_uids:
            skipped_no_data.append(uid)
            continue

        persona_path = os.path.join(persona_dir, f"user_{uid}.md")
        if not os.path.exists(persona_path):
            if skip_missing_persona:
                skipped_no_persona.append(uid)
                continue
            persona_text = None
        else:
            persona_text = open(persona_path, encoding="utf-8").read()
            persona_text = strip_think_tags(persona_text)

        sub = df[df["uid"] == uid].copy()
        out.append((uid, sub, persona_text))

    if skipped_no_data:
        print(f"[load] 跳过 {len(skipped_no_data)} 个 uid(CSV 中无数据): "
              f"{skipped_no_data[:10]}{'...' if len(skipped_no_data)>10 else ''}")
    if skipped_no_persona:
        print(f"[load] 跳过 {len(skipped_no_persona)} 个 uid(无画像文件): "
              f"{skipped_no_persona[:10]}{'...' if len(skipped_no_persona)>10 else ''}")
    print(f"[load] 实际加载 {len(out)}/{len(uids)} users")
    return out


# ============================================
# Prompt 拼装(供并行/串行版共用)
# ============================================
def build_monologue_prompt(state: UserState, mono_usr_tmpl: str) -> tuple:
    r = state.current_row()
    hour = r["datetime"].hour
    dow = DOW_MAP_CN[r["datetime"].dayofweek]

    subs = {
        "{UID}": str(state.uid),
        "{STUDY_DAY}": str(int(r["study_day"])),
        "{WEEKDAY}": dow,
        "{SLOT}": str(int(r["day_slot"])),
        "{HOUR}": str(hour),
        "{LOCATION}": str(r["location"]),
        "{ACTIVITY}": str(r["activity"]),
        "{WEATHER}": str(r["weather"]),
        "{TEMPERATURE}": str(r["temperature"]),
        "{IS_WEEKDAY}": str(r["is_weekday"]),
        "{SEND}": str(int(r["send"])),
        "{RESPONSE}": str(r["response"]),
        "{PRE30}": str(r["jbsteps30pre"]),
    }

    user_prompt = mono_usr_tmpl\
        .replace("{PERSONA}", state.persona)\
        .replace("{HISTORY}", build_history(state.prior_rows))
    for k, v in subs.items():
        user_prompt = user_prompt.replace(k, v)
    return subs, user_prompt


def build_post30_prompt(state: UserState, subs: dict, mono_text: str, reasoning: str,
                        post_usr_tmpl: str) -> str:
    user_prompt = post_usr_tmpl\
        .replace("{PERSONA_PRINCIPLES}", state.principles)\
        .replace("{MONOLOGUE}", mono_text)\
        .replace("{REASONING}", reasoning)
    for k, v in subs.items():
        user_prompt = user_prompt.replace(k, v)
    return user_prompt


def make_result_record(uid: int, row, post30: int, parsed: dict, raw: str = "",
                       persona: str = None, principles: str = None) -> dict:
    """构造一条 jsonl 记录。统一字段名,供所有版本使用。

    可选 persona / principles:传入时会写入每条 jsonl,便于事后单条复盘
    (不传则不写入,保持向后兼容)
    """
    rec = {
        "uid": uid,
        "study_day": int(row["study_day"]),
        "slot": int(row["day_slot"]),
        "location": str(row["location"]),
        "send": int(row["send"]),
        "response": str(row["response"]),
        "pre30": float(row["jbsteps30pre"]),
        "post30": int(max(0, post30)),
        "monologue": parsed["MONOLOGUE"],
        "reasoning": parsed["REASONING"],
        "summary": parsed["SUMMARY"] or f"d{int(row['study_day'])} s{int(row['day_slot'])}",
        "raw_mono": raw,
    }
    if persona is not None:
        rec["persona"] = persona
    if principles is not None:
        rec["principles"] = principles
    return rec
