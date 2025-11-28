#!/usr/bin/env python3
"""
Logit Lens for Introspective Confidence Direction

This script:
1. Computes the "Introspective Confidence" direction (Group A - Group B) for specific layers.
2. Projects this direction onto the vocabulary (Logit Lens).
3. Prints the top/bottom tokens to understand the semantic meaning of "Confidence" in this model.

Layers analyzed: 35 (Peak), 50 (SAE), 79 (Final).
"""

import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompt_utils import build_self_prompt


# --------- CONFIG ---------
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = "llama-3.3-70b-instruct"
LAYERS_TO_ANALYZE = [35, 50, 79]
TOP_K = 20

# Paths
DATA_DIR = Path("contrastive_pairs") / MODEL_NAME
INTRO_TRAIN = DATA_DIR / f"{MODEL_NAME}_introspective_extremes_AB_train.csv"
COMPILED_RESULTS = Path("compiled_results_smc") / f"{MODEL_NAME}_phase1_compiled.json"
OUTPUT_DIR = Path("interp/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_question_text(model_name: str) -> dict[str, str]:
    path = Path("compiled_results_smc") / f"{model_name}_phase1_compiled.json"
    with open(path) as f:
        data = json.load(f)
    return {k: v["question"] for k, v in data["results"].items()}





def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (4-bit)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


@torch.no_grad()
def get_layer_direction(
    model, tokenizer, df: pd.DataFrame, q_text_map: dict, layer_idx: int
) -> torch.Tensor:
    """Compute Mean(A) - Mean(B) for a specific layer."""
    print(f"Computing direction for Layer {layer_idx}...")

    a_vecs = []
    b_vecs = []

    # We'll batch this slightly or just run sequentially (dataset is small, ~20-50 pairs)
    # Running sequentially is fine for <100 items.

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Layer {layer_idx}"):
        if row["A_qid"] not in q_text_map or row["B_qid"] not in q_text_map:
            continue

        p_a = build_self_prompt(q_text_map[row["A_qid"]])
        p_b = build_self_prompt(q_text_map[row["B_qid"]])

        # Run A
        messages_a = [{"role": "user", "content": p_a}]
        formatted_a = tokenizer.apply_chat_template(
            messages_a, tokenize=False, add_generation_prompt=True
        )
        inputs_a = tokenizer(formatted_a, return_tensors="pt").to(model.device)
        out_a = model(**inputs_a, output_hidden_states=True)
        h_a = out_a.hidden_states[layer_idx + 1][:, -1, :].squeeze(0)
        a_vecs.append(h_a)

        # Run B
        messages_b = [{"role": "user", "content": p_b}]
        formatted_b = tokenizer.apply_chat_template(
            messages_b, tokenize=False, add_generation_prompt=True
        )
        inputs_b = tokenizer(formatted_b, return_tensors="pt").to(model.device)
        out_b = model(**inputs_b, output_hidden_states=True)
        h_b = out_b.hidden_states[layer_idx + 1][:, -1, :].squeeze(0)
        b_vecs.append(h_b)

    mu_a = torch.stack(a_vecs).mean(dim=0)
    mu_b = torch.stack(b_vecs).mean(dim=0)
    direction = mu_a - mu_b
    direction = direction / direction.norm()

    return direction


def run_logit_lens(model, tokenizer, direction: torch.Tensor, layer_idx: int):
    """Project direction onto vocab and print top/bottom tokens."""
    print(f"\n--- Logit Lens: Layer {layer_idx} ---")

    # Project onto vocabulary
    # direction: (hidden_dim,)
    # lm_head: (vocab_size, hidden_dim)
    # logits = direction @ lm_head.T
    logits = model.lm_head(direction.unsqueeze(0)).squeeze(0)

    # Top tokens (Aligned with A / High Confidence)
    top_vals, top_indices = torch.topk(logits, TOP_K)
    print(f"\nTop {TOP_K} tokens (Aligned with 'High Confidence'):")
    for val, idx in zip(top_vals, top_indices, strict=False):
        token = tokenizer.decode(idx.item()).strip()
        print(f"  {token:<15} ({val:.4f})")

    # Bottom tokens (Aligned with B / Low Confidence)
    bot_vals, bot_indices = torch.topk(logits, TOP_K, largest=False)
    print(f"\nBottom {TOP_K} tokens (Aligned with 'Low Confidence'):")
    for val, idx in zip(bot_vals, bot_indices, strict=False):
        token = tokenizer.decode(idx.item()).strip()
        print(f"  {token:<15} ({val:.4f})")

    # Save to file
    out_file = OUTPUT_DIR / f"introspective_logit_lens_layer{layer_idx}.txt"
    with open(out_file, "w") as f:
        f.write(f"Logit Lens for Introspective Confidence (Layer {layer_idx})\n")
        f.write("=" * 50 + "\n\n")
        f.write("Top Tokens (High Confidence):\n")
        for val, idx in zip(top_vals, top_indices, strict=False):
            token = tokenizer.decode(idx.item()).strip()
            f.write(f"{token:<20} {val:.4f}\n")

        f.write("\nBottom Tokens (Low Confidence):\n")
        for val, idx in zip(bot_vals, bot_indices, strict=False):
            token = tokenizer.decode(idx.item()).strip()
            f.write(f"{token:<20} {val:.4f}\n")
    print(f"\nSaved to {out_file}")


def main():
    print("=== Introspective Confidence Logit Lens ===")

    # Load Data
    q_text_map = load_question_text(MODEL_NAME)
    df_train = pd.read_csv(INTRO_TRAIN)
    print(f"Loaded {len(df_train)} training pairs")

    # Load Model
    tokenizer, model = load_model()

    # Analyze Layers
    for layer in LAYERS_TO_ANALYZE:
        direction = get_layer_direction(model, tokenizer, df_train, q_text_map, layer)
        run_logit_lens(model, tokenizer, direction, layer)

    print("\nDone!")


if __name__ == "__main__":
    main()
