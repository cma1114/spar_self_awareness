#!/usr/bin/env python3
"""
Save Self-Other Directions for Specific Layers

Computes and saves the Self-Other direction vector for specified layers.
Useful for generating input files for logit_lens.py.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
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
CSV_DIFF_TRAIN = DATA_DIR / f"{MODEL_NAME}_different_perspective_train.csv"
COMPILED_RESULTS = Path("compiled_results_smc") / f"{MODEL_NAME}_phase1_compiled.json"

# Output
OUTPUT_DIR = Path("interp/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Pair:
    question_id: str
    self_prompt: str
    other_prompt: str
    label: Literal["same", "different"]


def load_question_data(model_name: str) -> dict:
    baseline_path = Path("compiled_results_smc") / f"{model_name}_phase1_compiled.json"
    with open(baseline_path) as f:
        data = json.load(f)
    return data["results"]





def load_diff_pairs(diff_csv: Path, question_data: dict) -> list[Pair]:
    print(f"Loading DIFFERENT pairs from {diff_csv}")
    df = pd.read_csv(diff_csv)
    pairs = []
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
                label="different",
            )
        )
    print(f"Loaded {len(pairs)} pairs")
    return pairs


def load_model_and_tokenizer():
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
    return tokenizer, model


@torch.no_grad()
def compute_and_save_direction(model, tokenizer, pairs: list[Pair], layer_idx: int):
    """Compute mean(Self - Other) at specific layer and save to disk."""
    print(f"\nComputing direction at Layer {layer_idx}...")
    deltas = []

    # Use all pairs for high quality direction
    for i, pair in enumerate(pairs):
        if (i + 1) % 50 == 0:
            print(f"  Processing {i+1}/{len(pairs)}")

        # Self
        messages_self = [{"role": "user", "content": pair.self_prompt}]
        formatted_self = tokenizer.apply_chat_template(
            messages_self, tokenize=False, add_generation_prompt=True
        )
        inputs_self = tokenizer(formatted_self, return_tensors="pt").to(model.device)
        out_self = model(**inputs_self, output_hidden_states=True)
        h_self = out_self.hidden_states[layer_idx + 1][:, -1, :].squeeze(0)

        # Other
        messages_other = [{"role": "user", "content": pair.other_prompt}]
        formatted_other = tokenizer.apply_chat_template(
            messages_other, tokenize=False, add_generation_prompt=True
        )
        inputs_other = tokenizer(formatted_other, return_tensors="pt").to(model.device)
        out_other = model(**inputs_other, output_hidden_states=True)
        h_other = out_other.hidden_states[layer_idx + 1][:, -1, :].squeeze(0)

        deltas.append(h_self - h_other)

    direction = torch.stack(deltas).mean(dim=0)
    direction = direction / direction.norm()

    # Save
    output_path = OUTPUT_DIR / f"self_other_direction_layer{layer_idx}.pt"
    print(f"Saving to {output_path}...")
    torch.save(
        {
            "direction": direction.cpu(),
            "layer_index": layer_idx,
            "auc": 0.0,  # Placeholder, we just want the vector
            "model_id": MODEL_ID,
        },
        output_path,
    )
    print("✅ Saved!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[35, 50], help="Layers to save directions for"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(f"Saving Directions for Layers: {args.layers}")
    print("=" * 60 + "\n")

    # Load data
    question_data = load_question_data(MODEL_NAME)
    pairs = load_diff_pairs(CSV_DIFF_TRAIN, question_data)

    # Load model
    tokenizer, model = load_model_and_tokenizer()

    for layer_idx in args.layers:
        compute_and_save_direction(model, tokenizer, pairs, layer_idx)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
