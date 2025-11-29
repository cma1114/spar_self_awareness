"""
We compute:
    - capabilities_entropy  := entropy of the model's baseline answer logits,
                               i.e. uncertainty when actually answering Q.
    - p_i_capability        := whether the model actually got the Q right
                               (from baseline is_correct boolean -> 1/0 float).
    - Self Prob             := model's self-rated "prob I'm right" (1P is_correct float).
    - Other Prob            := model's other-rated "prob college-educated person is right"
                               (3P is_correct float).

Then we print the same correlations that analyze_dg_sqa.py prints:
    1. Corr( Other Prob , Self Prob )
    2. Corr( p_i_capability , Self Prob )
    3. Corr( p_i_capability , Other Prob )
    4. Corr( capabilities_entropy , Self Prob )     [note sign below]
    5. Corr( capabilities_entropy , Other Prob )

But for 4 & 5, we actually want to see "higher entropy -> lower claimed confidence".
So internally we do Corr( entropy , -SelfProb ) and Corr( entropy , -OtherProb )
and we print those as "Correlation between capabilities_entropy and Self Prob", etc.
That matches the sign convention from your earlier logs.

We also:
    - Intersect questions strictly by qid across baseline / 1p / 3p.
    - Skip models unless we have >= MIN_QUESTIONS_FULL aligned questions
      (default 400) so we ignore the 50-Q pilots.

Run:
    python phase1_self_other_analysis.py

Output:
    - Prints blocks to stdout like analyze_dg_sqa.py
    - Appends the same blocks to analysis_log_self_other_simplemc.txt
"""

import datetime
import glob
import json
import math
import os
import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

BASELINE_DIR = "compiled_results_smc"
ONEP_DIR = "capabilities_1p_test_logs"  # self
THREEP_DIR = "capabilities_3p_test_logs"  # other
LOG_PATH = "analysis_log_self_other_simplemc.txt"

# Only trust runs that look like the full 500‑question SimpleMC set.
MIN_QUESTIONS_FULL = 400

# List of candidate model names we know about.
MODEL_NAMES = [
    "claude-3-5-sonnet-20241022",
    "claude-opus-4-1-20250805",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-5-20250929_think",
    "deepseek-chat",
    "deepseek-chat-v3.1",
    "deepseek-r1",
    "deepseek-v3.1-base",
    "gemini-2.5-flash_nothink",
    "gpt-4.1-2025-04-14",
    "gpt-4o-2024-08-06",
    "grok-3-latest",
    "hermes-4-70b",
    "kimi-k2",
    "kimi-k2-0905",
    "llama-3.1-405b-instruct",
    "llama-3.1-8b-instruct",
    "llama-3.3-70b-instruct",
    "llama-4-maverick",
    "mistral-small-3.2-24b-instruct",
    "openai-gpt-5-chat",
    "qwen3-235b-a22b-2507",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def _compute_entropy(prob_dict):
    """
    prob_dict: e.g. {"A":0.02,"B":0.05,"C":0.91,"D":0.02}
    We normalize just in case, then compute natural‑log entropy.
    """
    if not isinstance(prob_dict, dict):
        return float("nan")
    vals = []
    for k, v in prob_dict.items():
        if v is None:
            continue
        try:
            vals.append(float(v))
        except Exception:
            continue
    s = sum(vals)
    if s <= 0:
        return float("nan")
    probs = [v / s for v in vals]
    ent = 0.0
    for p in probs:
        if p > 0:
            ent += -p * math.log(p)
    return ent


def _parse_baseline_phase1(json_obj):
    """
    Returns dict[qid] = {
        "entropy": <float>,
        "actual_correct": <0.0/1.0 float>,
        "question_text": <str>,
    }
    Expects compiled_results_smc/<model>_phase1_compiled.json shape.
    """
    out = {}
    results = json_obj.get("results", {})
    if not isinstance(results, dict):
        return out

    for qid, rec in results.items():
        # per‑Q entropy of model's *answering* logits
        ent = _compute_entropy(rec.get("probs", {}))

        # actual correctness of model's chosen answer
        is_corr = rec.get("is_correct")
        if isinstance(is_corr, bool):
            acc = 1.0 if is_corr else 0.0
        elif isinstance(is_corr, (int, float)):
            # sometimes is_correct may be 0/1 already
            acc = float(is_corr)
        else:
            # fallback
            acc = float("nan")

        question_text = ""
        q_block = rec.get("question", {})
        if isinstance(q_block, dict):
            question_text = q_block.get("text", "").strip()

        out[qid] = {
            "entropy": ent,
            "actual_correct": acc,
            "question_text": question_text,
        }
    return out


def _parse_capabilities_file(json_obj):
    """
    Returns dict[qid] = {
        "expected_prob": <float in [0,1]>,
        "question_text": <str>,
    }

    Expects capabilities_[13]p_test_logs/..._test_data.json shape, where
    'is_correct' is ALREADY the model's *expected correctness probability*
    (midpoint‑weighted A..H bin, not a binary 0/1).
    """
    out = {}
    results = json_obj.get("results", {})
    if not isinstance(results, dict):
        return out

    for qid, rec in results.items():
        exp_prob = _safe_float(rec.get("is_correct"))

        question_text = ""
        q_block = rec.get("question", {})
        if isinstance(q_block, dict):
            question_text = q_block.get("text", "").strip()

        out[qid] = {
            "expected_prob": exp_prob,
            "question_text": question_text,
        }
    return out


def _find_best_cap_file(dir_path, model_name):
    """
    Find the "best" 1P / 3P capability file for a model.

    We look for files like:
        <model>_SimpleMC_500_<timestamp>_test_data.json
    in dir_path. We prefer qcount>=400 (so 500‑Q runs over 50‑Q pilots),
    then highest timestamp.

    Returns (chosen_path, records_dict) OR (None, {}).
    """
    if not os.path.isdir(dir_path):
        return None, {}

    pattern = os.path.join(dir_path, f"{model_name}_SimpleMC_*_test_data.json")
    candidates = glob.glob(pattern)

    parsed = []
    for p in candidates:
        base = os.path.basename(p)
        m = re.match(rf"^{re.escape(model_name)}_SimpleMC_(\d+)_([0-9]+)_test_data\.json$", base)
        if not m:
            continue
        qcount = int(m.group(1))  # "500" vs "50"
        ts = int(m.group(2))  # timestamp-ish integer
        parsed.append((qcount, ts, p))

    if not parsed:
        return None, {}

    # Sort so best choice is first:
    #   1. prefer qcount >= 400
    #   2. then larger qcount
    #   3. then newer timestamp
    parsed.sort(key=lambda x: (x[0] >= 400, x[0], x[1]), reverse=True)
    chosen_path = parsed[0][2]

    try:
        with open(chosen_path) as f:
            data = json.load(f)
        recs = _parse_capabilities_file(data)
    except Exception:
        return None, {}

    return chosen_path, recs


def _analyze_model(model_name):
    """
    Do the full pipeline for one model.
    Returns (lines_to_print, summary_dict_or_None).
    """

    baseline_path = os.path.join(BASELINE_DIR, f"{model_name}_phase1_compiled.json")

    if not os.path.exists(baseline_path):
        return [f"Skipping {model_name}: missing baseline file."], None

    # load baseline json
    try:
        with open(baseline_path) as f:
            baseline_json = json.load(f)
        base_records = _parse_baseline_phase1(baseline_json)
    except Exception as e:
        return [f"Skipping {model_name}: failed to read baseline ({e})."], None

    # load best 1P (self) file
    onep_path, onep_records = _find_best_cap_file(ONEP_DIR, model_name)
    # load best 3P (other) file
    threep_path, threep_records = _find_best_cap_file(THREEP_DIR, model_name)

    if not onep_path or not threep_path:
        return [f"Skipping {model_name}: missing one of 1p/3p files."], None

    # Align questions across baseline, 1P, 3P
    aligned_ids = sorted(
        set(base_records.keys()) & set(onep_records.keys()) & set(threep_records.keys())
    )
    aligned_n = len(aligned_ids)

    # Build DataFrame with all variables
    data_rows = []
    for qid in aligned_ids:
        ent = base_records[qid]["entropy"]
        acc = base_records[qid]["actual_correct"]
        self_prob = onep_records[qid]["expected_prob"]
        other_prob = threep_records[qid]["expected_prob"]

        data_rows.append(
            {
                "qid": qid,
                "entropy": ent,
                "actual_correct": acc,
                "self_prob": self_prob,
                "other_prob": other_prob,
            }
        )

    df = pd.DataFrame(data_rows)

    # Replace inf with NaN for proper handling
    df = df.replace([np.inf, -np.inf], np.nan)

    # Count usable data per correlation pair (pandas handles NaN pairwise)
    n_other_self = df[["other_prob", "self_prob"]].dropna().shape[0]
    n_cap_self = df[["actual_correct", "self_prob"]].dropna().shape[0]
    n_cap_other = df[["actual_correct", "other_prob"]].dropna().shape[0]
    n_ent_self = df[["entropy", "self_prob"]].dropna().shape[0]
    n_ent_other = df[["entropy", "other_prob"]].dropna().shape[0]

    # Minimum threshold check
    min_n = min(n_other_self, n_cap_self, n_cap_other, n_ent_self, n_ent_other)
    usable_n = len(df)  # Total questions with at least one valid value

    # If it's not basically the full 500‑Q run, skip printing correlations.
    if min_n < MIN_QUESTIONS_FULL:
        lines = [
            f"Skipping {model_name}: insufficient usable data (<{MIN_QUESTIONS_FULL}).",
            f"  Sample sizes: Other-Self={n_other_self}, Cap-Self={n_cap_self}, Cap-Other={n_cap_other}, Ent-Self={n_ent_self}, Ent-Other={n_ent_other}",
            f"  baseline file: {baseline_path}",
            f"  1P file: {onep_path}",
            f"  3P file: {threep_path}",
        ]
        return lines, None

    # Correlations using pandas (handles NaN pairwise automatically)
    # Note: We use .corr() which computes Pearson by default
    r_other_self = df["other_prob"].corr(df["self_prob"])
    r_cap_self = df["actual_correct"].corr(df["self_prob"])
    r_cap_other = df["actual_correct"].corr(df["other_prob"])

    # For entropy: we want "higher entropy => lower claimed confidence"
    # So we correlate entropy with NEGATIVE of self/other probs
    # This gives positive correlation when high entropy = low confidence
    df["neg_self_prob"] = -df["self_prob"]
    df["neg_other_prob"] = -df["other_prob"]
    r_ent_self = df["entropy"].corr(df["neg_self_prob"])
    r_ent_other = df["entropy"].corr(df["neg_other_prob"])

    def _fmt(x):
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "NaN"
        return f"{x:.12f}"

    # Build output block
    block_lines = []
    block_lines.append("")
    block_lines.append(f"===== {model_name} =====")
    block_lines.append(f"Baseline file: {baseline_path}")
    block_lines.append(f"1P (self) file: {onep_path}")
    block_lines.append(f"3P (other) file: {threep_path}")
    block_lines.append(
        f"Raw counts: baseline={len(base_records)}, self(1p)={len(onep_records)}, other(3p)={len(threep_records)}"
    )
    block_lines.append(f"Aligned questions: {usable_n}")
    block_lines.append(
        f"Sample sizes per correlation: Other-Self={n_other_self}, Cap-Self={n_cap_self}, Cap-Other={n_cap_other}, Ent-Self={n_ent_self}, Ent-Other={n_ent_other}"
    )
    block_lines.append("")
    block_lines.append("Note: Entropy correlations use negative Self/Other probs,")
    block_lines.append(
        "      so positive correlation = high entropy → low confidence (correct interpretation)"
    )
    block_lines.append("")
    block_lines.append(
        f"Correlation between Other's Prob and Self Prob: {_fmt(r_other_self)} (n={n_other_self})"
    )
    block_lines.append("")
    block_lines.append(
        f"Correlation between p_i_capability and Self Prob: {_fmt(r_cap_self)} (n={n_cap_self})"
    )
    block_lines.append("")
    block_lines.append(
        f"Correlation between p_i_capability and Other Prob: {_fmt(r_cap_other)} (n={n_cap_other})"
    )
    block_lines.append("")
    block_lines.append(
        f"Correlation between capabilities_entropy and Self Prob: {_fmt(r_ent_self)} (n={n_ent_self})"
    )
    block_lines.append("")
    block_lines.append(
        f"Correlation between capabilities_entropy and Other Prob: {_fmt(r_ent_other)} (n={n_ent_other})"
    )
    block_lines.append("")

    summary_dict = {
        "model": model_name,
        "aligned_n": usable_n,
        "n_other_self": n_other_self,
        "n_cap_self": n_cap_self,
        "n_cap_other": n_cap_other,
        "n_ent_self": n_ent_self,
        "n_ent_other": n_ent_other,
        "corr_other_self": (
            r_other_self
            if not (isinstance(r_other_self, float) and math.isnan(r_other_self))
            else "NaN"
        ),
        "corr_cap_self": (
            r_cap_self if not (isinstance(r_cap_self, float) and math.isnan(r_cap_self)) else "NaN"
        ),
        "corr_cap_other": (
            r_cap_other
            if not (isinstance(r_cap_other, float) and math.isnan(r_cap_other))
            else "NaN"
        ),
        "corr_ent_self": (
            r_ent_self if not (isinstance(r_ent_self, float) and math.isnan(r_ent_self)) else "NaN"
        ),
        "corr_ent_other": (
            r_ent_other
            if not (isinstance(r_ent_other, float) and math.isnan(r_ent_other))
            else "NaN"
        ),
    }

    block_lines.append(f"summary_dict = {summary_dict}")
    block_lines.append("")

    return block_lines, summary_dict


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    header_lines = []
    header_lines.append("")
    header_lines.append("################################################################")
    header_lines.append(f"PHASE-1 SELF/OTHER ANALYSIS RUN {timestamp}")
    header_lines.append("################################################################")
    header_lines.append("")

    # We'll both print and tee to LOG_PATH
    all_lines_to_log = header_lines[:]

    for line in header_lines:
        print(line)

    for model_name in MODEL_NAMES:
        lines, summary = _analyze_model(model_name)
        for ln in lines:
            print(ln)
        all_lines_to_log.extend(lines)
        all_lines_to_log.append("")  # spacing

    # append to rolling log file
    try:
        with open(LOG_PATH, "a") as log_f:
            for ln in all_lines_to_log:
                log_f.write(ln + "\n")
    except Exception as e:
        print(f"[WARN] Could not append to {LOG_PATH}: {e}")


if __name__ == "__main__":
    main()
