"""Extract per-(slot, action) reward structure from real data.

This is the data-driven content that goes into LLM prompts — replacing the
LLM's default prior ("messages always help") with what the data actually says.
"""
from __future__ import annotations
import os
from typing import Dict
import pandas as pd
from src import config as cfg

def extract_state_action_signal(df: pd.DataFrame,
                                out_dir: str = None) -> Dict:
    """Returns a dict structure consumable by prompt_builder."""
    action_col = cfg.COL_ACTION
    slot_col = cfg.COL_SLOT
    avail_col = cfg.COL_AVAIL

    # Only use randomized subset for causal estimates
    use = df[df[avail_col]] if avail_col and avail_col in df.columns else df

    per_action = use.groupby(action_col)["reward"].agg(["count", "mean"]).round(3)
    per_slot_action = (use.groupby([slot_col, action_col])["reward"]
                          .agg(["count", "mean"]).round(3))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        per_action.to_csv(os.path.join(out_dir, "signal_per_action.csv"))
        per_slot_action.to_csv(os.path.join(out_dir, "signal_per_slot_action.csv"))

    # turn into a structured dict for prompts
    per_slot_dict = {}
    means_unstacked = per_slot_action["mean"].unstack(action_col)
    for slot, row in means_unstacked.iterrows():
        per_slot_dict[int(slot)] = {int(a): float(row[a]) for a in row.index if pd.notna(row[a])}

    return {
        "global_means": per_action["mean"].to_dict(),
        "per_slot": per_slot_dict,
    }
