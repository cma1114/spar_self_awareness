#!/usr/bin/env python3
"""
Layer Sweep for Self-Other Direction

Scans all layers (0-79) to find where the Self-Other signal is strongest.
Uses activation caching to run efficiently (one forward pass per prompt).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
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

# Output
OUTPUT_DIR = Path("interp/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Layers to scan (Llama-3.3-70B has 80 layers: 0-79)
LAYERS_TO_SCAN = list(range(0, 80, 1))  # Scan every layer for high resolution


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





def load_pairs_from_csv(same_csv: Path, diff_csv: Path, question_data: dict) -> list[Pair]:
    """Load SAME and DIFFERENT items from CSVs and create prompt pairs."""
    print(f"Loading pairs from:\n  {same_csv}\n  {diff_csv}")
    df_same = pd.read_csv(same_csv)
    df_diff = pd.read_csv(diff_csv)
    pairs: list[Pair] = []

    for label, df in [("same", df_same), ("different", df_diff)]:
        for _, row in df.iterrows():
            qid = row["question_id"]
            if qid not in question_data:
                continue
            q_text = question_data[qid]["question"]
            pairs.append(
                Pair(
                    question_id=qid,
                    self_prompt=build_self_prompt(q_text),
                    other_prompt=build_other_prompt(q_text),
                    label=label,
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

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.eval()
    print("✅ Model loaded successfully!")
    return tokenizer, model


@torch.no_grad()
def cache_all_activations(model, tokenizer, pairs: list[Pair]) -> dict:
    """
    Run model once per prompt and cache last-token hidden states for ALL layers.

    Returns:
        cache: dict structure:
        {
            qid: {
                "self": tensor(num_layers, hidden_size),
                "other": tensor(num_layers, hidden_size),
                "label": "same" | "different"
            }
        }
    """
    print(f"\nCaching activations for {len(pairs)} pairs (Self & Other)...")
    cache = {}

    for i, pair in enumerate(tqdm(pairs, desc="Caching")):
        # 1. Run Self Prompt
        messages_self = [{"role": "user", "content": pair.self_prompt}]
        formatted_self = tokenizer.apply_chat_template(
            messages_self, tokenize=False, add_generation_prompt=True
        )
        inputs_self = tokenizer(
            formatted_self, return_tensors="pt", add_special_tokens=True
        ).to(model.device)
        out_self = model(**inputs_self, output_hidden_states=True, use_cache=False)
        # Stack all layers: (num_layers, 1, seq_len, hidden) -> (num_layers, hidden)
        # We take the last token: [:, -1, :]
        # hidden_states is a tuple of length num_layers+1 (includes embeddings)
        # We want layers 0-79. Llama-3.3-70B has 80 layers.
        # hidden_states[0] is embeddings. hidden_states[1] is layer 0 output.
        # Let's store indices 1 to 80 to map to layers 0-79.
        # Actually, let's just store everything and index carefully later.
        # Stack into tensor: (81, hidden_size)
        h_self_stack = torch.stack([h[:, -1, :].squeeze(0).cpu() for h in out_self.hidden_states])

        # 2. Run Other Prompt
        messages_other = [{"role": "user", "content": pair.other_prompt}]
        formatted_other = tokenizer.apply_chat_template(
            messages_other, tokenize=False, add_generation_prompt=True
        )
        inputs_other = tokenizer(
            formatted_other, return_tensors="pt", add_special_tokens=True
        ).to(model.device)
        out_other = model(**inputs_other, output_hidden_states=True, use_cache=False)
        h_other_stack = torch.stack([h[:, -1, :].squeeze(0).cpu() for h in out_other.hidden_states])

        cache[pair.question_id] = {
            "self": h_self_stack,  # (81, 8192)
            "other": h_other_stack,  # (81, 8192)
            "label": pair.label,
        }

    return cache


def run_layer_sweep(cache: dict, layers_to_scan: list[int]):
    """
    Sweep through layers using cached activations.
    hidden_states tuple has 81 elements: [embeddings, layer0, ..., layer79].
    So layer N corresponds to index N+1.
    """
    results = []

    print(f"\nScanning {len(layers_to_scan)} layers...")

    for layer_idx in tqdm(layers_to_scan, desc="Sweeping"):
        # Map layer index to tuple index (0 -> 1, 79 -> 80)
        tuple_idx = layer_idx + 1

        # 1. Collect Train Deltas (DIFFERENT only)
        train_deltas = []
        for qid, data in cache.items():
            # We need to distinguish train vs test.
            # The cache contains ALL pairs. We need to know which are train/test.
            # Wait, the cache is just a dict. We need to pass train_pairs and test_pairs to this function
            # or split them here.
            pass

    return results


# Refactored to pass pairs explicitly
def run_sweep_with_split(
    cache: dict, train_pairs: list[Pair], test_pairs: list[Pair], layers_to_scan: list[int]
):
    results = []

    print(f"\nScanning {len(layers_to_scan)} layers...")

    for layer_idx in tqdm(layers_to_scan, desc="Sweeping"):
        tuple_idx = layer_idx + 1

        # 1. Compute Direction from DIFFERENT Train Pairs
        diff_deltas = []
        for p in train_pairs:
            if p.label == "different":
                h_self = cache[p.question_id]["self"][tuple_idx]
                h_other = cache[p.question_id]["other"][tuple_idx]
                diff_deltas.append(h_self - h_other)

        if not diff_deltas:
            continue

        diff_deltas = torch.stack(diff_deltas)
        direction = diff_deltas.mean(dim=0)
        direction = direction / direction.norm()

        # 2. Evaluate on Test Pairs
        scores = []
        y_true = []

        for p in test_pairs:
            h_self = cache[p.question_id]["self"][tuple_idx]
            h_other = cache[p.question_id]["other"][tuple_idx]
            delta = h_self - h_other

            score = float(torch.dot(direction, delta))
            scores.append(score)
            y_true.append(1 if p.label == "different" else 0)

        # 3. Compute AUC
        try:
            auc = roc_auc_score(y_true, scores)
            # Flip if anti-correlated
            if auc < 0.5:
                auc = 1.0 - auc
        except:
            auc = 0.5

        results.append(
            {
                "layer": layer_idx,
                "auc": auc,
                "num_train_diff": len(diff_deltas),
                "num_test": len(y_true),
            }
        )

    return results


def plot_results(results: list[dict], output_path: Path):
    """Generate AUC vs Layer plot with NeurIPS-ready aesthetics."""
    layers = [r["layer"] for r in results]
    aucs = [r["auc"] for r in results]

    # Set style
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    plt.figure(figsize=(8, 5), dpi=300)

    # Plot main line
    plt.plot(layers, aucs, color="#2E86C1", linewidth=2.5, label="Self-Other Separation")

    # Random chance line
    plt.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.8, label="Random Chance")

    # Find peak
    max_auc = max(aucs)
    best_layer = layers[aucs.index(max_auc)]
    plt.plot(best_layer, max_auc, marker="*", color="#E74C3C", markersize=12,
             linestyle="None", label=f"Peak: L{best_layer} ({max_auc:.3f})")

    # Aesthetics
    plt.title(f"Self-Other Separation by Layer ({MODEL_NAME})", fontsize=12, pad=10)
    plt.xlabel("Layer Index", fontsize=10)
    plt.ylabel("AUC (Same vs Different)", fontsize=10)
    plt.ylim(0, 1.05)
    plt.xlim(0, 80)

    # Legend
    plt.legend(frameon=True, fancybox=True, framealpha=0.9, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Plot saved to {output_path}")


def main():
    print("\n" + "=" * 60)
    print("Layer Sweep: Self-Other Direction")
    print("=" * 60 + "\n")

    # Load data
    question_data = load_question_data(MODEL_NAME)
    pairs_train = load_pairs_from_csv(CSV_SAME_TRAIN, CSV_DIFF_TRAIN, question_data)
    pairs_test = load_pairs_from_csv(CSV_SAME_TEST, CSV_DIFF_TEST, question_data)

    all_pairs = pairs_train + pairs_test

    # Load model
    tokenizer, model = load_model_and_tokenizer()

    # Cache activations (The Heavy Lifting)
    cache = cache_all_activations(model, tokenizer, all_pairs)

    # Run Sweep (Fast)
    results = run_sweep_with_split(cache, pairs_train, pairs_test, LAYERS_TO_SCAN)

    # Save Results
    json_path = OUTPUT_DIR / "layer_sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to {json_path}")

    # Plot
    plot_path = OUTPUT_DIR / "layer_sweep_plot.png"
    plot_results(results, plot_path)

    # Report Peak
    best = max(results, key=lambda x: x["auc"])
    print("\n" + "=" * 60)
    print(f"🏆 PEAK PERFORMANCE: Layer {best['layer']}")
    print(f"📈 AUC: {best['auc']:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
