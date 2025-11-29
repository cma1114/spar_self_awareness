#!/usr/bin/env python3
"""
Self-Other Direction Discovery and Evaluation

Computes a Self-Other direction from contrastive pairs and evaluates
how well it separates SAME vs DIFFERENT perspectives on held-out test data.

This implements first interpretability milestone:
- Load DIFFERENT_train and SAME_train contrastive pairs
- Extract hidden states at last token for Self and Other prompts
- Compute direction Δ = mean(Self - Other) over DIFFERENT_train
- Test if projection onto Δ separates SAME vs DIFFERENT on test set
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from prompt_utils import build_other_prompt, build_self_prompt


# --------- CONFIG ---------
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = "llama-3.3-70b-instruct"

# Paths
DATA_DIR = Path("contrastive_pairs") / MODEL_NAME
CSV_SAME_TRAIN = DATA_DIR / f"{MODEL_NAME}_same_perspective_train.csv"
CSV_DIFF_TRAIN = DATA_DIR / f"{MODEL_NAME}_different_perspective_train.csv"
CSV_SAME_TEST = DATA_DIR / f"{MODEL_NAME}_same_perspective_test.csv"
CSV_DIFF_TEST = DATA_DIR / f"{MODEL_NAME}_different_perspective_test.csv"

COMPILED_RESULTS = Path("compiled_results_smc") / f"{MODEL_NAME}_phase1_compiled.json"

# Which layer to analyze (-1 = last layer, can sweep later)
LAYER_INDEX = -1

# Output
OUTPUT_DIR = Path("interp/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Pair:
    """One Self vs Other contrast for a single question."""

    question_id: str
    self_prompt: str
    other_prompt: str
    label: Literal["same", "different"]


def load_question_data(model_name: str) -> dict:
    """Load question text from compiled results."""
    baseline_path = Path("compiled_results_smc") / f"{model_name}_phase1_compiled.json"
    print(f"Loading question data from {baseline_path}...")
    with open(baseline_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data['results'])} questions")
    return data["results"]





def load_pairs_from_csv(
    same_csv: Path,
    diff_csv: Path,
    question_data: dict,
) -> list[Pair]:
    """Load SAME and DIFFERENT items from CSVs and create prompt pairs."""
    print(f"Loading pairs from:\n  {same_csv}\n  {diff_csv}")

    df_same = pd.read_csv(same_csv)
    df_diff = pd.read_csv(diff_csv)

    pairs: list[Pair] = []

    # Process SAME perspective pairs
    for _, row in df_same.iterrows():
        qid = row["question_id"]
        if qid not in question_data:
            print(f"Warning: Question {qid} not found in compiled results, skipping")
            continue

        question_text = question_data[qid]["question"]
        pairs.append(
            Pair(
                question_id=qid,
                self_prompt=build_self_prompt(question_text),
                other_prompt=build_other_prompt(question_text),
                label="same",
            )
        )

    # Process DIFFERENT perspective pairs
    for _, row in df_diff.iterrows():
        qid = row["question_id"]
        if qid not in question_data:
            print(f"Warning: Question {qid} not found in compiled results, skipping")
            continue

        question_text = question_data[qid]["question"]
        pairs.append(
            Pair(
                question_id=qid,
                self_prompt=build_self_prompt(question_text),
                other_prompt=build_other_prompt(question_text),
                label="different",
            )
        )

    print(f"Loaded {len(pairs)} pairs ({len(df_same)} SAME + {len(df_diff)} DIFFERENT)")
    return pairs


def load_model_and_tokenizer():
    """Load Llama-3.3-70B in 4-bit quantization."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Setting up 4-bit quantization config...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    print("Loading model (this will use cached weights from smoke test)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.eval()
    print("✅ Model loaded successfully!")

    return tokenizer, model


@torch.no_grad()
def get_last_token_hidden(
    model,
    tokenizer,
    prompt: str,
    layer_index: int = -1,
) -> torch.Tensor:
    """
    Run model on prompt and return hidden state at last token.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input text
        layer_index: Which layer to extract (-1 = last layer)

    Returns:
        Hidden state vector of shape (hidden_size,)
    """
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(model.device)

    outputs = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
    )

    # hidden_states is tuple: (layer0, layer1, ..., final)
    # Each element is (batch, seq_len, hidden_size)
    hidden_states = outputs.hidden_states
    layer_h = hidden_states[layer_index]  # (1, seq_len, hidden_size)
    last_token_h = layer_h[:, -1, :]  # (1, hidden_size)

    return last_token_h.squeeze(0).detach().cpu()


def compute_direction_and_scores(
    pairs_train: list[Pair],
    pairs_test: list[Pair],
    tokenizer,
    model,
    layer_index: int,
):
    """
    Compute Self-Other direction from DIFFERENT_train and evaluate on test set.

    Algorithm:
    1. For each training pair, get h_self and h_other at specified layer
    2. Compute delta = h_self - h_other
    3. Direction = mean(delta) over DIFFERENT items only, then normalize
    4. For test pairs, project their deltas onto direction
    5. Compute AUC: can projection score separate SAME vs DIFFERENT?

    Returns:
        direction: The normalized Self-Other direction vector
        auc: Area under ROC curve on test set
    """
    print(f"\n{'='*60}")
    print(f"Computing Self-Other direction at layer {layer_index}")
    print(f"{'='*60}\n")

    # ---- 1. Extract deltas on TRAIN ----
    print(f"Extracting activations for {len(pairs_train)} train pairs...")
    deltas_train = []
    labels_train = []

    for i, pair in enumerate(pairs_train):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(pairs_train)} train pairs")

        h_self = get_last_token_hidden(model, tokenizer, pair.self_prompt, layer_index)
        h_other = get_last_token_hidden(model, tokenizer, pair.other_prompt, layer_index)
        delta = h_self - h_other  # Self - Other

        deltas_train.append(delta)
        labels_train.append(pair.label)

    deltas_train = torch.stack(deltas_train)  # (N_train, hidden_size)
    print(f"✅ Extracted {len(deltas_train)} train deltas, shape: {deltas_train.shape}")

    # ---- 2. Compute direction from DIFFERENT only ----
    mask_diff = np.array(labels_train) == "different"
    diff_deltas = deltas_train[mask_diff]

    print(f"\nComputing direction from {diff_deltas.shape[0]} DIFFERENT pairs...")
    direction = diff_deltas.mean(dim=0)
    direction = direction / direction.norm()  # normalize to unit length
    print(f"✅ Direction computed, norm: {direction.norm():.6f}")

    # ---- 3. Evaluate on TEST ----
    print(f"\nEvaluating on {len(pairs_test)} test pairs...")
    scores_test = []
    y_test = []

    for i, pair in enumerate(pairs_test):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(pairs_test)} test pairs")

        h_self = get_last_token_hidden(model, tokenizer, pair.self_prompt, layer_index)
        h_other = get_last_token_hidden(model, tokenizer, pair.other_prompt, layer_index)
        delta = h_self - h_other

        # Project onto direction
        score = float(torch.dot(direction, delta))
        scores_test.append(score)
        y_test.append(1 if pair.label == "different" else 0)

    # Compute AUC
    auc = roc_auc_score(y_test, scores_test)

    # The direction sign is arbitrary. If AUC < 0.5, flip the direction and scores.
    # This standardizes so DIFFERENT always has higher scores than SAME.
    if auc < 0.5:
        print("⚠️ AUC < 0.5, flipping direction sign so that DIFFERENT has higher scores...")
        direction = -direction
        scores_test = [-s for s in scores_test]
        auc = 1.0 - auc  # mirrored AUC

    print(f"\n{'='*60}")
    print("✅ RESULTS")
    print(f"{'='*60}")
    print(f"Layer: {layer_index}")
    print(f"Direction trained on: {mask_diff.sum()} DIFFERENT pairs")
    print(f"Test set: {sum(y_test)} DIFFERENT + {len(y_test) - sum(y_test)} SAME")
    print(f"AUC (SAME vs DIFFERENT): {auc:.4f}")
    print(f"{'='*60}\n")

    return direction, auc, scores_test, y_test


def main():
    print("\n" + "=" * 60)
    print("Self-Other Direction Discovery")
    print("=" * 60 + "\n")

    # Load question data
    question_data = load_question_data(MODEL_NAME)

    # Load contrastive pairs
    print("\nLoading contrastive pairs...")
    pairs_train = load_pairs_from_csv(CSV_SAME_TRAIN, CSV_DIFF_TRAIN, question_data)
    pairs_test = load_pairs_from_csv(CSV_SAME_TEST, CSV_DIFF_TEST, question_data)

    # FULL RUN MODE - using all pairs
    print("🚀 FULL RUN: Using all training and test pairs")

    print("\n📊 Dataset summary:")
    print(f"  Train: {len(pairs_train)} pairs")
    print(f"    - SAME: {sum(1 for p in pairs_train if p.label == 'same')}")
    print(f"    - DIFFERENT: {sum(1 for p in pairs_train if p.label == 'different')}")
    print(f"  Test: {len(pairs_test)} pairs")
    print(f"    - SAME: {sum(1 for p in pairs_test if p.label == 'same')}")
    print(f"    - DIFFERENT: {sum(1 for p in pairs_test if p.label == 'different')}")

    # Load model
    print("\n" + "=" * 60)
    tokenizer, model = load_model_and_tokenizer()

    # Compute direction and evaluate
    direction, auc, scores, labels = compute_direction_and_scores(
        pairs_train, pairs_test, tokenizer, model, LAYER_INDEX
    )

    # Save results
    output_path = OUTPUT_DIR / f"self_other_direction_layer{LAYER_INDEX}.pt"
    print(f"Saving direction to {output_path}...")
    torch.save(
        {
            "direction": direction,
            "layer_index": LAYER_INDEX,
            "auc": auc,
            "model_id": MODEL_ID,
            "test_scores": scores,
            "test_labels": labels,
        },
        output_path,
    )
    print("✅ Saved!")

    # Summary
    print("\n" + "=" * 60)
    print("🎉 FIRST INTERPRETABILITY MILESTONE COMPLETE!")
    print("=" * 60)
    print(f"✅ Found Self-Other direction at layer {LAYER_INDEX}")
    print(f"✅ AUC on held-out test: {auc:.4f}")
    print(f"✅ Direction saved to: {output_path}")
    print("\nNext steps:")
    print("  1. Scan across all layers to find best layer")
    print("  2. Visualize projections (SAME vs DIFFERENT distributions)")
    print("  3. Test causality via steering experiments")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
