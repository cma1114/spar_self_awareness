#!/usr/bin/env python3
"""
generate_contrastive_pairs.py

Zero-config miner for contrastive pairs using Phase-1 (SimpleMC-500) artifacts.

Assumed repo layout (from repo root):
  - compiled_results_smc/<model>_phase1_compiled.json        (baseline A/B/C/D with probs dict)
  - capabilities_1p_test_logs/<model>_SimpleMC_500_*_test_data.json    (1P/self confidence)
  - capabilities_3p_test_logs/<model>_SimpleMC_500_*_test_data.json    (3P/other confidence)

Outputs (in contrastive_pairs/<model>/):
  EXISTING:
  - <model>_unified.csv                    (all aligned data with margin + gap_abs)
  - <model>_self_other_gap.csv             (top ~100 by |SelfProb-OtherProb| with pmax>=0.55)
  - <model>_calibration_extremes.csv       (≤50 pairs: overconf-wrong ↔ underconf-right)
  - <model>_easy_vs_hard.csv               (≤100 pairs by answer letter; easy vs hard)

  NEW:
  - <model>_same_perspective.csv           (Self ≈ Other, gap ≤ 0.10)
  - <model>_same_perspective_train.csv     (70% split for interp discovery)
  - <model>_same_perspective_test.csv      (30% split for interp validation)
  - <model>_different_perspective.csv      (Self ≠ Other, gap ≥ 0.30)
  - <model>_different_perspective_train.csv
  - <model>_different_perspective_test.csv
  - <model>_opposite_extremes_AB.csv       (paired: confident-self vs hesitant-self)
  - <model>_opposite_extremes_AB_train.csv
  - <model>_opposite_extremes_AB_test.csv

Run:
  python generate_contrastive_pairs.py

Configuration via environment variables (optional):
  EXISTING:
  - P_MAX_MIN_FOR_GAP=0.55         # Minimum pmax filter
  - EASY_PMAX_TH=0.80              # Threshold for "easy"
  - HARD_PMAX_TH=0.40              # Threshold for "hard"
  - OC_WRONG_SELF_TH=0.80          # Overconfident-wrong threshold
  - UC_RIGHT_SELF_TH=0.20          # Underconfident-right threshold
  - TOP_GAP_N=100                  # Number of self-other gap pairs
  - CALIBRATION_PAIRS=50           # Number of calibration pairs
  - EASY_HARD_PER_LETTER=25        # Pairs per answer letter
  - MIN_ALIGNED=480                # Minimum aligned questions

  NEW - SAME vs DIFFERENT:
  - GAP_SAME_EPS=0.10              # SAME if gap ≤ this
  - GAP_DIFF_MIN=0.30              # DIFFERENT if gap ≥ this
  - GAP_SAME_Q=0.20                # Fallback: bottom 20% quantile
  - GAP_DIFF_Q=0.80                # Fallback: top 20% quantile
  - MIN_BUCKET_SIZE=20             # Minimum questions per bucket

  NEW - OPPOSITE EXTREMES:
  - SELF_HIGH=0.80                 # High self-confidence threshold
  - SELF_LOW=0.20                  # Low self-confidence threshold
  - OTHER_HIGH=0.70                # High other-confidence threshold
  - OTHER_LOW=0.30                 # Low other-confidence threshold
  - ENTROPY_LOW_Q=0.30             # Low entropy = bottom 30% quantile
  - ENTROPY_HIGH_Q=0.70            # High entropy = top 30% quantile
  - MIN_EXTREME_PAIRS=10           # Minimum pairs required

  TRAIN/TEST SPLIT:
  - TRAIN_SPLIT=0.70               # 70/30 train/test split for interp
"""

import glob
import hashlib
import json
import math
import os
import re
import time

import numpy as np
import pandas as pd


# ---------- Config ----------
# P_MAX_MIN_FOR_GAP: Minimum probability of the top answer to consider the question "settled".
# We use 0.55 (slightly > 0.5) to ensure the model has a clear majority winner.
# If p_max < 0.55, the model is "confused" between options, making confidence analysis noisy.
P_MAX_MIN_FOR_GAP = float(os.environ.get("P_MAX_MIN_FOR_GAP", "0.55"))

# EASY/HARD Thresholds (for Easy vs Hard contrast)
# Questions with p_max >= 0.80 are considered "Easy" (high certainty).
# Questions with p_max <= 0.40 are considered "Hard" (low certainty/confusion).
EASY_PMAX_TH = float(os.environ.get("EASY_PMAX_TH", "0.80"))
HARD_PMAX_TH = float(os.environ.get("HARD_PMAX_TH", "0.40"))

# Overconfidence/Underconfidence Thresholds
# OC: High confidence (>= 0.80) but WRONG answer.
# UC: Low confidence (<= 0.20) but RIGHT answer.
OC_WRONG_SELF_TH = float(os.environ.get("OC_WRONG_SELF_TH", "0.80"))
UC_RIGHT_SELF_TH = float(os.environ.get("UC_RIGHT_SELF_TH", "0.20"))

# Sampling Limits
TOP_GAP_N = int(os.environ.get("TOP_GAP_N", "100"))          # Max pairs for Self-Other gap
CALIBRATION_PAIRS = int(os.environ.get("CALIBRATION_PAIRS", "50")) # Max pairs for calibration
EASY_HARD_PER_LETTER = int(os.environ.get("EASY_HARD_PER_LETTER", "25")) # Balanced sampling per answer letter

# Minimum aligned questions required to process a model (avoids processing incomplete runs)
MIN_ALIGNED = int(os.environ.get("MIN_ALIGNED", "480"))

# NEW MINERS CONFIG:
# Gap Thresholds for "Same Perspective" vs "Different Perspective"
# GAP_SAME_EPS: Maximum gap (abs diff) to consider Self and Other as "Same" (e.g. 0.10)
GAP_SAME_EPS = float(os.environ.get("GAP_SAME_EPS", "0.10"))
# GAP_DIFF_MIN: Minimum gap (abs diff) to consider Self and Other as "Different" (e.g. 0.30)
GAP_DIFF_MIN = float(os.environ.get("GAP_DIFF_MIN", "0.30"))

# Quantiles for Gap Filtering (to avoid outliers)
GAP_SAME_Q = float(os.environ.get("GAP_SAME_Q", "0.20"))
GAP_DIFF_Q = float(os.environ.get("GAP_DIFF_Q", "0.80"))
MIN_BUCKET_SIZE = int(os.environ.get("MIN_BUCKET_SIZE", "20"))

# NEW: Opposite Extremes Config (Introspective Confidence)
# Defining "High Confidence" vs "Low Confidence" for Self and Other
SELF_HIGH = float(os.environ.get("SELF_HIGH", "0.80"))   # Self prob >= 0.80
SELF_LOW = float(os.environ.get("SELF_LOW", "0.20"))     # Self prob <= 0.20
OTHER_HIGH = float(os.environ.get("OTHER_HIGH", "0.70")) # Other prob >= 0.70
OTHER_LOW = float(os.environ.get("OTHER_LOW", "0.30"))   # Other prob <= 0.30

# Entropy Quantiles (for defining "Sharp" vs "Flat" distributions)
ENTROPY_LOW_Q = float(os.environ.get("ENTROPY_LOW_Q", "0.30"))  # Bottom 30% entropy (Sharp)
ENTROPY_HIGH_Q = float(os.environ.get("ENTROPY_HIGH_Q", "0.70")) # Top 30% entropy (Flat)
MIN_EXTREME_PAIRS = int(os.environ.get("MIN_EXTREME_PAIRS", "10"))

# NEW: Train/test split ratio
TRAIN_SPLIT = float(os.environ.get("TRAIN_SPLIT", "0.70"))

# NEW: Trivial question filtering
TRIVIAL_THRESHOLD = float(os.environ.get("TRIVIAL_THRESHOLD", "0.70"))

# List of models to process (same as phase1_self_other_analysis.py)
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


# ---------- Helpers ----------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def _compute_entropy(prob_dict):
    """
    Compute natural-log entropy from a probability dict.
    Handles normalization and NaN cases.
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
        "pmax": <float>,
        "margin": <float>,  # NEW: p_top1 - p_top2
        "actual_correct": <0.0/1.0 float>,
        "correct_answer": <str>,
        "question_text": <str>,
    }
    Expects compiled_results_smc/<model>_phase1_compiled.json shape.
    """
    out = {}
    results = json_obj.get("results", {})
    if not isinstance(results, dict):
        return out

    for qid, rec in results.items():
        # per-Q entropy of model's *answering* logits
        ent = _compute_entropy(rec.get("probs", {}))

        # pmax and margin from probs (normalize first, with softmax fallback for logits)
        prob_dict = rec.get("probs", {})
        if isinstance(prob_dict, dict) and prob_dict:
            # Extract values
            vals = []
            for v in prob_dict.values():
                if v is not None:
                    try:
                        vals.append(float(v))
                    except Exception:
                        continue

            if vals:
                # Check if values look like logits (any negative) or unnormalized probs
                has_negative = any(v < 0 for v in vals)
                s = sum(vals)
                is_unnormalized = s < 0.99 or s > 1.01

                if has_negative or is_unnormalized:
                    # Apply softmax (stable version)
                    max_val = max(vals)
                    exp_vals = [np.exp(v - max_val) for v in vals]
                    exp_sum = sum(exp_vals)
                    normalized = [e / exp_sum for e in exp_vals]
                else:
                    # Simple normalization
                    normalized = [v / s for v in vals]

                sorted_probs = sorted(normalized, reverse=True)
                pmax = sorted_probs[0]
                # margin = difference between top 2 probabilities
                if len(sorted_probs) >= 2:
                    margin = sorted_probs[0] - sorted_probs[1]
                else:
                    margin = pmax  # Only one option means margin = pmax
            else:
                pmax = float("nan")
                margin = float("nan")
        else:
            pmax = float("nan")
            margin = float("nan")

        # actual correctness of model's chosen answer
        is_corr = rec.get("is_correct")
        if isinstance(is_corr, bool):
            acc = 1.0 if is_corr else 0.0
        elif isinstance(is_corr, (int, float)):
            acc = float(is_corr)
        else:
            acc = float("nan")

        # correct answer label
        correct_answer = ""
        if "correct_answer_label" in rec:
            correct_answer = rec["correct_answer_label"]
        elif "correct_answer" in rec:
            correct_answer = rec["correct_answer"]
        elif "question" in rec and isinstance(rec["question"], dict):
            correct_answer = rec["question"].get(
                "correct_answer_label", rec["question"].get("correct_answer", "")
            )

        question_text = ""
        q_block = rec.get("question", {})
        if isinstance(q_block, dict):
            question_text = q_block.get("text", "").strip()
        elif "question_text" in rec:
            question_text = rec["question_text"]

        out[qid] = {
            "entropy": ent,
            "pmax": pmax,
            "margin": margin,
            "actual_correct": acc,
            "correct_answer": correct_answer,
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
    (midpoint-weighted A..H bin, not a binary 0/1).
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
    in dir_path. We prefer qcount>=400 (so 500-Q runs over 50-Q pilots),
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


def build_unified_df(model_name: str) -> pd.DataFrame | None:
    """
    Build unified dataframe for a single model by aligning baseline, 1P, and 3P data.
    Returns None if insufficient data.
    """
    baseline_path = os.path.join("compiled_results_smc", f"{model_name}_phase1_compiled.json")

    if not os.path.exists(baseline_path):
        log(f"  SKIP: Missing baseline file {baseline_path}")
        return None

    # Load baseline
    try:
        with open(baseline_path) as f:
            baseline_json = json.load(f)
        base_records = _parse_baseline_phase1(baseline_json)
    except Exception as e:
        log(f"  SKIP: Failed to read baseline ({e})")
        return None

    # Load best 1P (self) file
    onep_path, onep_records = _find_best_cap_file("capabilities_1p_test_logs", model_name)
    # Load best 3P (other) file
    threep_path, threep_records = _find_best_cap_file("capabilities_3p_test_logs", model_name)

    if not onep_path or not threep_path:
        log(f"  SKIP: Missing 1P or 3P files (1P={onep_path}, 3P={threep_path})")
        return None

    log(f"  Baseline: {baseline_path}")
    log(f"  1P/self:  {onep_path}")
    log(f"  3P/other: {threep_path}")

    # Align questions across baseline, 1P, 3P
    aligned_ids = sorted(
        set(base_records.keys()) & set(onep_records.keys()) & set(threep_records.keys())
    )
    aligned_n = len(aligned_ids)

    if aligned_n < MIN_ALIGNED:
        log(f"  SKIP: Insufficient aligned questions ({aligned_n} < {MIN_ALIGNED})")
        return None

    log(f"  Aligned questions: {aligned_n}")

    # Build DataFrame with all variables
    data_rows = []
    for qid in aligned_ids:
        ent = base_records[qid]["entropy"]
        pmax = base_records[qid]["pmax"]
        margin = base_records[qid]["margin"]
        acc = base_records[qid]["actual_correct"]
        corr_ans = base_records[qid]["correct_answer"]
        q_text = base_records[qid]["question_text"]
        self_prob = onep_records[qid]["expected_prob"]
        other_prob = threep_records[qid]["expected_prob"]

        # Fallback for question text
        if not q_text and onep_records[qid]["question_text"]:
            q_text = onep_records[qid]["question_text"]
        if not q_text and threep_records[qid]["question_text"]:
            q_text = threep_records[qid]["question_text"]

        data_rows.append(
            {
                "question_id": qid,
                "entropy": ent,
                "pmax": pmax,
                "margin": margin,
                "correct": acc,
                "correct_answer": corr_ans,
                "SelfProb": self_prob,
                "OtherProb": other_prob,
                "question_text": q_text,
            }
        )

    df = pd.DataFrame(data_rows)

    # Replace inf with NaN for proper handling
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaNs in critical fields
    df = df.dropna(subset=["entropy", "pmax", "margin", "correct", "SelfProb", "OtherProb"])

    # Add computed columns
    df["gap_abs"] = (df["SelfProb"] - df["OtherProb"]).abs()
    df["gap_signed"] = df["SelfProb"] - df["OtherProb"]
    df["direction"] = df["gap_signed"].apply(
        lambda x: "Self>Other" if x > 0 else ("Other>Self" if x < 0 else "Equal")
    )

    log(f"  Final rows after NaN removal: {len(df)}")

    # Quick sanity check correlations
    try:
        r_ent_self = df["entropy"].corr(-df["SelfProb"])
        r_ent_other = df["entropy"].corr(-df["OtherProb"])
        log(f"  Quick check: corr(H, -Self)={r_ent_self:.3f}, corr(H, -Other)={r_ent_other:.3f}")
    except Exception:
        pass

    return df


def preflight_diagnostics(df: pd.DataFrame, model_name: str) -> None:
    """Print diagnostic statistics before mining."""
    log("  ")
    log("  ═══ PREFLIGHT DIAGNOSTICS ═══")

    # Filter by pmax
    df_filtered = df[df["pmax"] >= P_MAX_MIN_FOR_GAP].copy()
    n_filtered = len(df_filtered)
    n_total = len(df)

    log(f"  Total questions: {n_total}")
    log(f"  After pmax ≥ {P_MAX_MIN_FOR_GAP} filter: {n_filtered} ({100*n_filtered/n_total:.1f}%)")

    if n_filtered == 0:
        log("  WARNING: No questions pass pmax filter!")
        return

    # Percentiles
    log("  ")
    log("  Percentiles (after pmax filter):")
    for col, name in [
        ("gap_abs", "Gap |Self-Other|"),
        ("entropy", "Entropy"),
        ("pmax", "Pmax"),
        ("margin", "Margin"),
    ]:
        p10 = df_filtered[col].quantile(0.10)
        p20 = df_filtered[col].quantile(0.20)
        p80 = df_filtered[col].quantile(0.80)
        p90 = df_filtered[col].quantile(0.90)
        log(f"    {name:20s}: 10%={p10:.3f}, 20%={p20:.3f}, 80%={p80:.3f}, 90%={p90:.3f}")

    # Direction counts
    log("  ")
    log("  Direction Balance:")
    n_self_gt = (df_filtered["SelfProb"] > df_filtered["OtherProb"]).sum()
    n_other_gt = (df_filtered["SelfProb"] < df_filtered["OtherProb"]).sum()
    n_equal = (df_filtered["SelfProb"] == df_filtered["OtherProb"]).sum()
    total = n_self_gt + n_other_gt + n_equal
    log(f"    Self > Other: {n_self_gt} ({100*n_self_gt/total:.1f}%)")
    log(f"    Other > Self: {n_other_gt} ({100*n_other_gt/total:.1f}%)")
    log(f"    Equal:        {n_equal} ({100*n_equal/total:.1f}%)")

    if n_self_gt / total > 0.80 or n_other_gt / total > 0.80:
        log("      WARNING: Direction imbalance detected (>80% one-sided)")

    log("  ═══════════════════════════════")
    log("  ")


def mine_same_vs_different(df: pd.DataFrame, outdir: str, model_name: str) -> None:
    """
    Miner 1: SAME vs DIFFERENT perspective alignment.

    SAME = questions where Self ≈ Other (small gap)
    DIFFERENT = questions where Self ≠ Other (large gap)
    """
    log("  ")
    log("  === Mining SAME vs DIFFERENT ===")

    # Filter by pmax
    df_filtered = df[df["pmax"] >= P_MAX_MIN_FOR_GAP].copy()
    n_filtered = len(df_filtered)

    if n_filtered == 0:
        log("  SKIP: No questions after pmax filter")
        return

    # Drop trivial "everyone thinks it's easy" cases (Chris's suggestion)
    mask_trivial = (df_filtered["SelfProb"] >= TRIVIAL_THRESHOLD) & (
        df_filtered["OtherProb"] >= TRIVIAL_THRESHOLD
    )
    n_trivial = mask_trivial.sum()
    if n_trivial > 0:
        log(
            f"  Dropping {n_trivial} trivial questions (Self>={TRIVIAL_THRESHOLD} & Other>={TRIVIAL_THRESHOLD})"
        )
        df_filtered = df_filtered[~mask_trivial]
        n_filtered = len(df_filtered)

    # Try strict thresholds first
    same_mask = df_filtered["gap_abs"] <= GAP_SAME_EPS
    diff_mask = df_filtered["gap_abs"] >= GAP_DIFF_MIN

    n_same_strict = same_mask.sum()
    n_diff_strict = diff_mask.sum()

    # Fallback to quantiles if needed
    same_threshold = GAP_SAME_EPS
    diff_threshold = GAP_DIFF_MIN
    same_method = "strict"
    diff_method = "strict"

    if n_same_strict < MIN_BUCKET_SIZE:
        log(
            f"  SAME: Strict threshold (≤{GAP_SAME_EPS}) yields {n_same_strict} < {MIN_BUCKET_SIZE}"
        )
        log(f"  SAME: Falling back to bottom {GAP_SAME_Q*100:.0f}% quantile")
        gap_q_low = df_filtered["gap_abs"].quantile(GAP_SAME_Q)
        same_mask = df_filtered["gap_abs"] <= gap_q_low
        same_threshold = gap_q_low
        same_method = "quantile"

    if n_diff_strict < MIN_BUCKET_SIZE:
        log(
            f"  DIFFERENT: Strict threshold (≥{GAP_DIFF_MIN}) yields {n_diff_strict} < {MIN_BUCKET_SIZE}"
        )
        log(f"  DIFFERENT: Falling back to top {(1-GAP_DIFF_Q)*100:.0f}% quantile")
        gap_q_high = df_filtered["gap_abs"].quantile(GAP_DIFF_Q)
        diff_mask = df_filtered["gap_abs"] >= gap_q_high
        diff_threshold = gap_q_high
        diff_method = "quantile"

    # Extract buckets (make DIFFERENT explicitly disjoint from SAME)
    df_same = df_filtered[same_mask].copy()
    diff_mask_disjoint = diff_mask & (~same_mask)  # Ensure no overlap
    df_different = df_filtered[diff_mask_disjoint].copy()

    # Middle range (discarded)
    middle_mask = ~same_mask & ~diff_mask
    n_middle = middle_mask.sum()

    # Verify disjoint (should always be true now with explicit disjoint mask)
    assert (same_mask & diff_mask_disjoint).sum() == 0, "SAME and DIFFERENT buckets overlap!"
    assert (
        same_mask.sum() + diff_mask_disjoint.sum() + n_middle == n_filtered
    ), "Bucket counts don't add up!"

    log(
        f"  SAME bucket: {len(df_same)} questions (using {same_method}, threshold={same_threshold:.3f})"
    )
    log(
        f"  DIFFERENT bucket: {len(df_different)} questions (using {diff_method}, threshold={diff_threshold:.3f})"
    )
    log(f"  Middle range (discarded): {n_middle} questions")

    if len(df_same) > 0:
        # Direction in SAME
        n_self_gt_same = (df_same["SelfProb"] > df_same["OtherProb"]).sum()
        n_other_gt_same = (df_same["SelfProb"] < df_same["OtherProb"]).sum()
        log(f"    SAME directions: Self>Other={n_self_gt_same}, Other>Self={n_other_gt_same}")
        log(
            f"    SAME medians: gap={df_same['gap_abs'].median():.3f}, entropy={df_same['entropy'].median():.3f}, Self={df_same['SelfProb'].median():.3f}, Other={df_same['OtherProb'].median():.3f}"
        )

        # Save SAME
        same_out = df_same[
            [
                "question_id",
                "SelfProb",
                "OtherProb",
                "gap_abs",
                "gap_signed",
                "direction",
                "pmax",
                "margin",
                "correct",
                "entropy",
                "correct_answer",
                "question_text",
            ]
        ].copy()
        same_path = os.path.join(outdir, f"{model_name}_same_perspective.csv")
        same_out.to_csv(same_path, index=False)
        log(f"  Saved: {same_path}")

        # Train/test split for SAME (deterministic shuffle via stable hash)
        def stable_hash(qid):
            return int(hashlib.md5(str(qid).encode()).hexdigest()[:8], 16)

        same_out["_sort_key"] = same_out["question_id"].apply(stable_hash)
        same_out = (
            same_out.sort_values("_sort_key").drop("_sort_key", axis=1).reset_index(drop=True)
        )
        n_train = int(len(same_out) * TRAIN_SPLIT)
        same_train = same_out.iloc[:n_train]
        same_test = same_out.iloc[n_train:]
        same_train.to_csv(same_path.replace(".csv", "_train.csv"), index=False)
        same_test.to_csv(same_path.replace(".csv", "_test.csv"), index=False)
        log(f"    Train: {len(same_train)}, Test: {len(same_test)}")

    if len(df_different) > 0:
        # Direction in DIFFERENT
        n_self_gt_diff = (df_different["SelfProb"] > df_different["OtherProb"]).sum()
        n_other_gt_diff = (df_different["SelfProb"] < df_different["OtherProb"]).sum()
        log(f"    DIFFERENT directions: Self>Other={n_self_gt_diff}, Other>Self={n_other_gt_diff}")
        log(
            f"    DIFFERENT medians: gap={df_different['gap_abs'].median():.3f}, entropy={df_different['entropy'].median():.3f}, Self={df_different['SelfProb'].median():.3f}, Other={df_different['OtherProb'].median():.3f}"
        )

        # Save DIFFERENT
        diff_out = df_different[
            [
                "question_id",
                "SelfProb",
                "OtherProb",
                "gap_abs",
                "gap_signed",
                "direction",
                "pmax",
                "margin",
                "correct",
                "entropy",
                "correct_answer",
                "question_text",
            ]
        ].copy()
        diff_path = os.path.join(outdir, f"{model_name}_different_perspective.csv")
        diff_out.to_csv(diff_path, index=False)
        log(f"  Saved: {diff_path}")

        # Train/test split for DIFFERENT (deterministic shuffle via stable hash)
        def stable_hash(qid):
            return int(hashlib.md5(str(qid).encode()).hexdigest()[:8], 16)

        diff_out["_sort_key"] = diff_out["question_id"].apply(stable_hash)
        diff_out = (
            diff_out.sort_values("_sort_key").drop("_sort_key", axis=1).reset_index(drop=True)
        )
        n_train = int(len(diff_out) * TRAIN_SPLIT)
        diff_train = diff_out.iloc[:n_train]
        diff_test = diff_out.iloc[n_train:]
        diff_train.to_csv(diff_path.replace(".csv", "_train.csv"), index=False)
        diff_test.to_csv(diff_path.replace(".csv", "_test.csv"), index=False)
        log(f"    Train: {len(diff_train)}, Test: {len(diff_test)}")


def mine_opposite_extremes(df: pd.DataFrame, outdir: str, model_name: str) -> None:
    """
    Miner 2: Opposite Extremes A vs B.

    Condition A: High Self + Low Entropy + Low Other (confident self, sharp answer, low humans)
    Condition B: Low Self + High Entropy + High Other (hesitant self, fuzzy answer, high humans)
    """
    log("  ")
    log("  === Mining Opposite Extremes ===")

    # Filter by pmax
    df_filtered = df[df["pmax"] >= P_MAX_MIN_FOR_GAP].copy()
    n_filtered = len(df_filtered)

    if n_filtered == 0:
        log("  SKIP: No questions after pmax filter")
        return

    # Compute entropy quantiles on filtered data
    ent_lo = df_filtered["entropy"].quantile(ENTROPY_LOW_Q)
    ent_hi = df_filtered["entropy"].quantile(ENTROPY_HIGH_Q)

    log(f"  Entropy thresholds: low ≤ {ent_lo:.3f}, high ≥ {ent_hi:.3f}")

    # Stage 1: Try strict thresholds
    cond_a = df_filtered[
        (df_filtered["SelfProb"] >= SELF_HIGH)
        & (df_filtered["entropy"] <= ent_lo)
        & (df_filtered["OtherProb"] <= OTHER_LOW)
    ].copy()

    cond_b = df_filtered[
        (df_filtered["SelfProb"] <= SELF_LOW)
        & (df_filtered["entropy"] >= ent_hi)
        & (df_filtered["OtherProb"] >= OTHER_HIGH)
    ].copy()

    n_a = len(cond_a)
    n_b = len(cond_b)
    stage_used = "strict"

    log(f"  Stage 1 (strict): A={n_a}, B={n_b}")

    # Stage 2: Relax Other constraints if needed
    if n_a < MIN_EXTREME_PAIRS or n_b < MIN_EXTREME_PAIRS:
        log("  Stage 2: Relaxing Other constraints by 0.1")
        cond_a = df_filtered[
            (df_filtered["SelfProb"] >= SELF_HIGH)
            & (df_filtered["entropy"] <= ent_lo)
            & (df_filtered["OtherProb"] <= OTHER_LOW + 0.1)
        ].copy()

        cond_b = df_filtered[
            (df_filtered["SelfProb"] <= SELF_LOW)
            & (df_filtered["entropy"] >= ent_hi)
            & (df_filtered["OtherProb"] >= OTHER_HIGH - 0.1)
        ].copy()

        n_a = len(cond_a)
        n_b = len(cond_b)
        stage_used = "relaxed_other"
        log(f"  Stage 2 result: A={n_a}, B={n_b}")

    # Stage 3: Relax Self constraints if still needed
    if n_a < MIN_EXTREME_PAIRS or n_b < MIN_EXTREME_PAIRS:
        log("  Stage 3: Relaxing Self constraints by 0.1")
        cond_a = df_filtered[
            (df_filtered["SelfProb"] >= SELF_HIGH - 0.1)
            & (df_filtered["entropy"] <= ent_lo)
            & (df_filtered["OtherProb"] <= OTHER_LOW + 0.1)
        ].copy()

        cond_b = df_filtered[
            (df_filtered["SelfProb"] <= SELF_LOW + 0.1)
            & (df_filtered["entropy"] >= ent_hi)
            & (df_filtered["OtherProb"] >= OTHER_HIGH - 0.1)
        ].copy()

        n_a = len(cond_a)
        n_b = len(cond_b)
        stage_used = "relaxed_both"
        log(f"  Stage 3 result: A={n_a}, B={n_b}")

    # Stage 4: Use quantiles as last resort
    if n_a < MIN_EXTREME_PAIRS or n_b < MIN_EXTREME_PAIRS:
        log("  Stage 4: Using quantiles (top/bottom 20%)")
        self_q_high = df_filtered["SelfProb"].quantile(0.80)
        self_q_low = df_filtered["SelfProb"].quantile(0.20)
        other_q_low = df_filtered["OtherProb"].quantile(0.30)
        other_q_high = df_filtered["OtherProb"].quantile(0.70)

        cond_a = df_filtered[
            (df_filtered["SelfProb"] >= self_q_high)
            & (df_filtered["entropy"] <= ent_lo)
            & (df_filtered["OtherProb"] <= other_q_low)
        ].copy()

        cond_b = df_filtered[
            (df_filtered["SelfProb"] <= self_q_low)
            & (df_filtered["entropy"] >= ent_hi)
            & (df_filtered["OtherProb"] >= other_q_high)
        ].copy()

        n_a = len(cond_a)
        n_b = len(cond_b)
        stage_used = "quantile"
        log(f"  Stage 4 result: A={n_a}, B={n_b}")

    log(f"  Final: Condition A={n_a}, Condition B={n_b} (using {stage_used} thresholds)")

    if n_a == 0 or n_b == 0:
        log("  ERROR: One condition has zero questions, cannot pair!")
        return

    if n_a < 5 or n_b < 5:
        log(f"  WARNING: Very few pairs (A={n_a}, B={n_b}), results may not be reliable")

    # Check for duplicate QIDs within each condition
    assert cond_a["question_id"].duplicated().sum() == 0, "Duplicate QIDs in Condition A"
    assert cond_b["question_id"].duplicated().sum() == 0, "Duplicate QIDs in Condition B"

    # Sort deterministically for pairing
    cond_a = cond_a.sort_values(
        ["SelfProb", "entropy", "OtherProb"], ascending=[False, True, True]
    ).reset_index(drop=True)
    cond_b = cond_b.sort_values(
        ["SelfProb", "entropy", "OtherProb"], ascending=[True, False, False]
    ).reset_index(drop=True)

    # Pair
    n_pairs = min(n_a, n_b)
    pairs = []

    for i in range(n_pairs):
        a = cond_a.iloc[i]
        b = cond_b.iloc[i]
        pairs.append(
            {
                "pair_id": f"opp_{i:03d}",
                "A_qid": a["question_id"],
                "A_Self": a["SelfProb"],
                "A_Other": a["OtherProb"],
                "A_entropy": a["entropy"],
                "A_pmax": a["pmax"],
                "A_margin": a["margin"],
                "A_correct": int(a["correct"]),
                "A_question_text": a["question_text"],
                "B_qid": b["question_id"],
                "B_Self": b["SelfProb"],
                "B_Other": b["OtherProb"],
                "B_entropy": b["entropy"],
                "B_pmax": b["pmax"],
                "B_margin": b["margin"],
                "B_correct": int(b["correct"]),
                "B_question_text": b["question_text"],
            }
        )

    # Log medians
    log(
        f"  Condition A medians: Self={cond_a['SelfProb'][:n_pairs].median():.3f}, Other={cond_a['OtherProb'][:n_pairs].median():.3f}, entropy={cond_a['entropy'][:n_pairs].median():.3f}"
    )
    log(
        f"  Condition B medians: Self={cond_b['SelfProb'][:n_pairs].median():.3f}, Other={cond_b['OtherProb'][:n_pairs].median():.3f}, entropy={cond_b['entropy'][:n_pairs].median():.3f}"
    )

    # Save pairs
    pairs_df = pd.DataFrame(pairs)
    pairs_path = os.path.join(outdir, f"{model_name}_opposite_extremes_AB.csv")
    pairs_df.to_csv(pairs_path, index=False)
    log(f"  Saved: {pairs_path} ({n_pairs} pairs)")

    # Train/test split (deterministic shuffle via stable hash of pair_id)
    def stable_hash(pid):
        return int(hashlib.md5(str(pid).encode()).hexdigest()[:8], 16)

    pairs_df["_sort_key"] = pairs_df["pair_id"].apply(stable_hash)
    pairs_df = pairs_df.sort_values("_sort_key").drop("_sort_key", axis=1).reset_index(drop=True)
    n_train = int(n_pairs * TRAIN_SPLIT)
    pairs_train = pairs_df.iloc[:n_train]
    pairs_test = pairs_df.iloc[n_train:]
    pairs_train.to_csv(pairs_path.replace(".csv", "_train.csv"), index=False)
    pairs_test.to_csv(pairs_path.replace(".csv", "_test.csv"), index=False)
    log(f"    Train: {len(pairs_train)} pairs, Test: {len(pairs_test)} pairs")

    # Assertions (only check numeric columns to avoid brittle text field checks)
    numeric_cols = pairs_df.select_dtypes(include=[np.number]).columns
    nan_count = pairs_df[numeric_cols].isna().sum().sum()
    assert nan_count == 0, f"NaN values found in numeric columns! Count: {nan_count}"
    log("  Assertions passed (no NaNs in numeric columns, no duplicates)")


def mine_pairs(df: pd.DataFrame, outdir: str, model_name: str) -> None:
    """
    Generate three types of contrastive pairs and save to CSV files.
    """
    os.makedirs(outdir, exist_ok=True)

    # Save unified data
    unified_path = os.path.join(outdir, f"{model_name}_unified.csv")
    df.to_csv(unified_path, index=False)
    log(f"  Saved unified data: {unified_path}")

    # 1) Self vs Other big gap pairs
    df_gap = df[df["pmax"] >= P_MAX_MIN_FOR_GAP].copy()
    df_gap = df_gap.sort_values("gap_abs", ascending=False).head(TOP_GAP_N)

    gap_out = df_gap[
        [
            "question_id",
            "SelfProb",
            "OtherProb",
            "gap_abs",
            "gap_signed",
            "direction",
            "pmax",
            "margin",
            "correct",
            "entropy",
            "correct_answer",
            "question_text",
        ]
    ].copy()
    gap_path = os.path.join(outdir, f"{model_name}_self_other_gap.csv")
    gap_out.to_csv(gap_path, index=False)
    log(f"  Saved self-other gap pairs: {gap_path} ({len(gap_out)} pairs)")

    # 2) Calibration extremes pairs
    overconf = df[(df["correct"] == 0) & (df["SelfProb"] >= OC_WRONG_SELF_TH)].copy()
    underconf = df[(df["correct"] == 1) & (df["SelfProb"] <= UC_RIGHT_SELF_TH)].copy()

    overconf = overconf.sort_values("SelfProb", ascending=False)
    underconf = underconf.sort_values("SelfProb", ascending=True)

    n_pairs = min(len(overconf), len(underconf), CALIBRATION_PAIRS)

    pairs = []
    for i in range(n_pairs):
        oc = overconf.iloc[i]
        uc = underconf.iloc[i]
        pairs.append(
            {
                "pair_id": f"calib_{i:03d}",
                "overconf_qid": oc["question_id"],
                "overconf_self": oc["SelfProb"],
                "overconf_pmax": oc["pmax"],
                "overconf_margin": oc["margin"],
                "overconf_entropy": oc["entropy"],
                "overconf_question": oc["question_text"],
                "underconf_qid": uc["question_id"],
                "underconf_self": uc["SelfProb"],
                "underconf_pmax": uc["pmax"],
                "underconf_margin": uc["margin"],
                "underconf_entropy": uc["entropy"],
                "underconf_question": uc["question_text"],
            }
        )

    if pairs:
        calib_path = os.path.join(outdir, f"{model_name}_calibration_extremes.csv")
        pd.DataFrame(pairs).to_csv(calib_path, index=False)
        log(f"  Saved calibration extreme pairs: {calib_path} ({len(pairs)} pairs)")
    else:
        log("  WARNING: No calibration extreme pairs found")

    # 3) Easy vs Hard pairs (matched by correct_answer)
    if "correct_answer" in df.columns and df["correct_answer"].notna().any():
        easy = df[df["pmax"] >= EASY_PMAX_TH].copy().sort_values("pmax", ascending=False)
        hard = df[df["pmax"] <= HARD_PMAX_TH].copy().sort_values("pmax", ascending=True)

        pairs_eh = []
        for letter in sorted(df["correct_answer"].dropna().unique()):
            easy_l = easy[easy["correct_answer"] == letter].reset_index(drop=True)
            hard_l = hard[hard["correct_answer"] == letter].reset_index(drop=True)

            c = min(len(easy_l), len(hard_l), EASY_HARD_PER_LETTER)
            for i in range(c):
                e = easy_l.iloc[i]
                h = hard_l.iloc[i]
                pairs_eh.append(
                    {
                        "pair_id": f"eh_{letter}_{i:03d}",
                        "answer_letter": letter,
                        "easy_qid": e["question_id"],
                        "easy_pmax": e["pmax"],
                        "easy_margin": e["margin"],
                        "easy_entropy": e["entropy"],
                        "easy_correct": int(e["correct"]),
                        "easy_question": e["question_text"],
                        "hard_qid": h["question_id"],
                        "hard_pmax": h["pmax"],
                        "hard_margin": h["margin"],
                        "hard_entropy": h["entropy"],
                        "hard_correct": int(h["correct"]),
                        "hard_question": h["question_text"],
                    }
                )

        if pairs_eh:
            eh_path = os.path.join(outdir, f"{model_name}_easy_vs_hard.csv")
            pd.DataFrame(pairs_eh).to_csv(eh_path, index=False)
            log(f"  Saved easy-vs-hard pairs: {eh_path} ({len(pairs_eh)} pairs)")
        else:
            log("  WARNING: No easy-vs-hard pairs found")
    else:
        log("  SKIP easy-vs-hard: No correct_answer labels available")


def mine_introspective_extremes(df: pd.DataFrame, outdir: str, model_name: str) -> None:
    """
    Miner 3: Introspective Extremes (Chris's definition - Relaxed).

    Focuses purely on Introspective Confidence (Self + Entropy).
    We remove the hard constraint on OtherProb because the model has a strong
    Self>Other bias, making "Low Self + High Other" (Group B) nearly impossible to find.

    Group A (High Introspective Confidence):
      - High Self Probability
      - Low Entropy

    Group B (Low Introspective Confidence):
      - Low Self Probability
      - High Entropy
    """
    log("  ")
    log("  === Mining Introspective Extremes (Self + Entropy only) ===")

    # Filter by pmax
    df_filtered = df[df["pmax"] >= P_MAX_MIN_FOR_GAP].copy()
    n_filtered = len(df_filtered)

    if n_filtered == 0:
        log("  SKIP: No questions after pmax filter")
        return

    # Compute entropy quantiles on filtered data
    ent_lo = df_filtered["entropy"].quantile(ENTROPY_LOW_Q)
    ent_hi = df_filtered["entropy"].quantile(ENTROPY_HIGH_Q)

    log(f"  Entropy thresholds: low ≤ {ent_lo:.3f}, high ≥ {ent_hi:.3f}")

    # Stage 1: Try strict thresholds (Self + Entropy only)
    cond_a = df_filtered[
        (df_filtered["SelfProb"] >= SELF_HIGH) & (df_filtered["entropy"] <= ent_lo)
    ].copy()

    cond_b = df_filtered[
        (df_filtered["SelfProb"] <= SELF_LOW) & (df_filtered["entropy"] >= ent_hi)
    ].copy()

    n_a = len(cond_a)
    n_b = len(cond_b)
    stage_used = "strict"

    log(f"  Stage 1 (strict): A={n_a}, B={n_b}")

    # Stage 2: Relax Self constraints if needed
    if n_a < MIN_EXTREME_PAIRS or n_b < MIN_EXTREME_PAIRS:
        log("  Stage 2: Relaxing Self constraints by 0.1")
        cond_a = df_filtered[
            (df_filtered["SelfProb"] >= SELF_HIGH - 0.1) & (df_filtered["entropy"] <= ent_lo)
        ].copy()

        cond_b = df_filtered[
            (df_filtered["SelfProb"] <= SELF_LOW + 0.1) & (df_filtered["entropy"] >= ent_hi)
        ].copy()

        n_a = len(cond_a)
        n_b = len(cond_b)
        stage_used = "relaxed_self"
        log(f"  Stage 2 result: A={n_a}, B={n_b}")

    # Stage 3: Use quantiles as last resort
    if n_a < MIN_EXTREME_PAIRS or n_b < MIN_EXTREME_PAIRS:
        log("  Stage 3: Using quantiles (top/bottom 20%)")
        self_q_high = df_filtered["SelfProb"].quantile(0.80)
        self_q_low = df_filtered["SelfProb"].quantile(0.20)

        cond_a = df_filtered[
            (df_filtered["SelfProb"] >= self_q_high) & (df_filtered["entropy"] <= ent_lo)
        ].copy()

        cond_b = df_filtered[
            (df_filtered["SelfProb"] <= self_q_low) & (df_filtered["entropy"] >= ent_hi)
        ].copy()

        n_a = len(cond_a)
        n_b = len(cond_b)
        stage_used = "quantile"
        log(f"  Stage 3 result: A={n_a}, B={n_b}")

    log(f"  Final: Condition A={n_a}, Condition B={n_b} (using {stage_used} thresholds)")

    if n_a == 0 or n_b == 0:
        log("  ERROR: One condition has zero questions, cannot pair!")
        return

    if n_a < 5 or n_b < 5:
        log(f"  WARNING: Very few pairs (A={n_a}, B={n_b}), results may not be reliable")

    # Check for duplicate QIDs within each condition
    assert cond_a["question_id"].duplicated().sum() == 0, "Duplicate QIDs in Condition A"
    assert cond_b["question_id"].duplicated().sum() == 0, "Duplicate QIDs in Condition B"

    # Sort deterministically for pairing
    cond_a = cond_a.sort_values(["SelfProb", "entropy"], ascending=[False, True]).reset_index(
        drop=True
    )
    cond_b = cond_b.sort_values(["SelfProb", "entropy"], ascending=[True, False]).reset_index(
        drop=True
    )

    # Pair
    n_pairs = min(n_a, n_b)
    pairs = []

    for i in range(n_pairs):
        a = cond_a.iloc[i]
        b = cond_b.iloc[i]
        pairs.append(
            {
                "pair_id": f"intro_{i:03d}",
                "A_qid": a["question_id"],
                "A_Self": a["SelfProb"],
                "A_Other": a["OtherProb"],
                "A_entropy": a["entropy"],
                "A_pmax": a["pmax"],
                "A_margin": a["margin"],
                "A_correct": int(a["correct"]),
                "A_question_text": a["question_text"],
                "B_qid": b["question_id"],
                "B_Self": b["SelfProb"],
                "B_Other": b["OtherProb"],
                "B_entropy": b["entropy"],
                "B_pmax": b["pmax"],
                "B_margin": b["margin"],
                "B_correct": int(b["correct"]),
                "B_question_text": b["question_text"],
            }
        )

    # Log medians
    log(
        f"  Condition A medians: Self={cond_a['SelfProb'][:n_pairs].median():.3f}, Other={cond_a['OtherProb'][:n_pairs].median():.3f}, entropy={cond_a['entropy'][:n_pairs].median():.3f}"
    )
    log(
        f"  Condition B medians: Self={cond_b['SelfProb'][:n_pairs].median():.3f}, Other={cond_b['OtherProb'][:n_pairs].median():.3f}, entropy={cond_b['entropy'][:n_pairs].median():.3f}"
    )

    # Save pairs
    pairs_df = pd.DataFrame(pairs)
    pairs_path = os.path.join(outdir, f"{model_name}_introspective_extremes_AB.csv")
    pairs_df.to_csv(pairs_path, index=False)
    log(f"  Saved: {pairs_path} ({n_pairs} pairs)")

    # Train/test split (deterministic shuffle via stable hash of pair_id)
    def stable_hash(pid):
        return int(hashlib.md5(str(pid).encode()).hexdigest()[:8], 16)

    pairs_df["_sort_key"] = pairs_df["pair_id"].apply(stable_hash)
    pairs_df = pairs_df.sort_values("_sort_key").drop("_sort_key", axis=1).reset_index(drop=True)
    n_train = int(n_pairs * TRAIN_SPLIT)
    pairs_train = pairs_df.iloc[:n_train]
    pairs_test = pairs_df.iloc[n_train:]
    pairs_train.to_csv(pairs_path.replace(".csv", "_train.csv"), index=False)
    pairs_test.to_csv(pairs_path.replace(".csv", "_test.csv"), index=False)
    log(f"    Train: {len(pairs_train)} pairs, Test: {len(pairs_test)} pairs")

    # Assertions (only check numeric columns to avoid brittle text field checks)
    numeric_cols = pairs_df.select_dtypes(include=[np.number]).columns
    nan_count = pairs_df[numeric_cols].isna().sum().sum()
    assert nan_count == 0, f"NaN values found in numeric columns! Count: {nan_count}"
    log("  Assertions passed (no NaNs in numeric columns, no duplicates)")


def main():
    log("=== Contrastive Pair Generation ===")
    log("Configuration:")
    log(f"  P_MAX_MIN_FOR_GAP = {P_MAX_MIN_FOR_GAP}")
    log(f"  EASY_PMAX_TH = {EASY_PMAX_TH}")
    log(f"  HARD_PMAX_TH = {HARD_PMAX_TH}")
    log(f"  OC_WRONG_SELF_TH = {OC_WRONG_SELF_TH}")
    log(f"  UC_RIGHT_SELF_TH = {UC_RIGHT_SELF_TH}")
    log(f"  TOP_GAP_N = {TOP_GAP_N}")
    log(f"  CALIBRATION_PAIRS = {CALIBRATION_PAIRS}")
    log(f"  EASY_HARD_PER_LETTER = {EASY_HARD_PER_LETTER}")
    log(f"  MIN_ALIGNED = {MIN_ALIGNED}")
    log("  ")
    log("  NEW MINERS:")
    log(f"  GAP_SAME_EPS = {GAP_SAME_EPS}")
    log(f"  GAP_DIFF_MIN = {GAP_DIFF_MIN}")
    log(f"  MIN_BUCKET_SIZE = {MIN_BUCKET_SIZE}")
    log(f"  SELF_HIGH = {SELF_HIGH}, SELF_LOW = {SELF_LOW}")
    log(f"  OTHER_HIGH = {OTHER_HIGH}, OTHER_LOW = {OTHER_LOW}")
    log(f"  ENTROPY_LOW_Q = {ENTROPY_LOW_Q}, ENTROPY_HIGH_Q = {ENTROPY_HIGH_Q}")
    log(f"  MIN_EXTREME_PAIRS = {MIN_EXTREME_PAIRS}")
    log(f"  TRAIN_SPLIT = {TRAIN_SPLIT}")
    log("")

    outroot = "contrastive_pairs"
    os.makedirs(outroot, exist_ok=True)

    processed_count = 0
    skipped_count = 0

    for model_name in MODEL_NAMES:
        log(f"Processing: {model_name}")

        df = build_unified_df(model_name)

        if df is None or len(df) < MIN_ALIGNED:
            skipped_count += 1
            continue

        outdir = os.path.join(outroot, model_name)

        # Run diagnostics
        preflight_diagnostics(df, model_name)

        # Mine existing pairs
        mine_pairs(df, outdir, model_name)

        # NEW: Mine SAME vs DIFFERENT
        mine_same_vs_different(df, outdir, model_name)

        # NEW: Mine Opposite Extremes
        mine_opposite_extremes(df, outdir, model_name)

        # NEW: Mine Introspective Extremes (Chris's definition)
        mine_introspective_extremes(df, outdir, model_name)

        processed_count += 1
        log("")

    log("=== Summary ===")
    log(f"Processed: {processed_count} models")
    log(f"Skipped: {skipped_count} models")
    log(f"Output directory: {outroot}/")
    log("Done!")


if __name__ == "__main__":
    main()
