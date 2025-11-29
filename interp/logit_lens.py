#!/usr/bin/env python3
"""
Logit-lens style analysis of the Self–Other direction.

Given:
  - A saved direction file from self_other_direction.py
  - The Llama-3.3-70B-Instruct model

This script:
  1. Loads the direction vector v (shape [hidden_size])
  2. Loads the model and tokenizer in 4-bit
  3. Takes the unembedding matrix W_U (vocab_size x hidden_size)
  4. Computes scores = W_U @ v  (one score per token)
  5. Prints the top-k tokens on the positive side (SELF-ish)
     and the top-k on the negative side (OTHER-ish)

You can point it at any direction file, e.g.:
  - layer -1 file  : interp/outputs/self_other_direction_layer-1.pt
  - layer 35 file  : interp/outputs/self_other_direction_layer35.pt  (if you create it)
  - layer 50 file  : interp/outputs/self_other_direction_layer50.pt  (for SAE layer)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ---------------- CONFIG DEFAULTS ----------------

DEFAULT_MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_DIRECTION_PATH = "interp/outputs/self_other_direction_layer35.pt"
TOP_K = 40  # how many tokens to show on each side


# ---------------- HELPER FUNCTIONS ----------------


def load_direction(path: Path) -> dict:
    """Load the saved direction dict from disk."""
    print(f"Loading direction file from: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    required_keys = {"direction", "layer_index", "auc", "model_id"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Direction file is missing keys: {missing}")

    v = data["direction"]
    if v.ndim != 1:
        raise ValueError(f"Expected 1D direction vector, got shape {tuple(v.shape)}")

    print(f"  Layer index: {data['layer_index']}")
    print(f"  Stored model_id: {data['model_id']}")
    print(f"  AUC: {data['auc']:.4f}")
    print(f"  Direction shape: {tuple(v.shape)}")
    return data


def load_model_and_tokenizer(model_id: str):
    """Load Llama-3.3-70B in 4-bit quantization (same as other scripts)."""
    print(f"\nLoading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Setting up 4-bit quantization config...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    print("Loading model (weights should already be cached)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.eval()
    print("✅ Model loaded successfully!")
    return tokenizer, model


def get_unembedding_matrix(model) -> torch.Tensor:
    """
    Get the output (unembedding) matrix W_U: vocab_size x hidden_size.

    For Llama-family models in transformers, this is usually model.lm_head.weight.
    """
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        W_U = model.lm_head.weight  # (vocab_size, hidden_size)
        print("Using model.lm_head.weight as unembedding matrix.")
    else:
        raise RuntimeError("Could not find unembedding matrix (lm_head.weight) on model.")
    return W_U


def analyze_direction_with_logit_lens(
    direction: torch.Tensor,
    tokenizer,
    model,
    top_k: int = TOP_K,
    output_path: Path | None = None,
):
    """
    Project the direction through the unembedding matrix and print top tokens.
    """
    W_U = get_unembedding_matrix(model)  # (V, H)

    # Move direction to same device & dtype as W_U
    direction = direction.to(W_U.device, dtype=W_U.dtype)

    print("\nComputing token scores: scores = W_U @ direction ...")
    # scores: (vocab_size,)
    scores = torch.matmul(W_U, direction)  # (V,)

    scores_cpu = scores.detach().float().cpu()
    scores_np = scores_cpu.numpy()
    vocab_size = scores_np.shape[0]
    print(f"  Got scores for vocab size: {vocab_size}")

    # Top-k positive (SELF-ish) and negative (OTHER-ish) tokens
    top_pos_indices = scores_np.argsort()[-top_k:][::-1]  # highest first
    top_neg_indices = scores_np.argsort()[:top_k]  # most negative

    def pretty_print(indices, title: str, f):
        header = "\n" + "=" * 60 + "\n" + title + "\n" + "=" * 60 + "\n"
        print(header, end="")
        if f:
            f.write(header)

        for rank, idx in enumerate(indices, start=1):
            token = tokenizer.convert_ids_to_tokens(int(idx))
            score = float(scores_np[int(idx)])
            # Make whitespace tokens more readable
            printable = repr(token)
            line = f"{rank:2d}. id={idx:6d}  token={printable:20s}  score={score: .4f}\n"
            print(line, end="")
            if f:
                f.write(line)

    if output_path:
        with open(output_path, "w") as f:
            pretty_print(top_pos_indices, "Top tokens on POSITIVE side (SELF-aligned)", f)
            pretty_print(top_neg_indices, "Top tokens on NEGATIVE side (OTHER-aligned)", f)
    else:
        pretty_print(top_pos_indices, "Top tokens on POSITIVE side (SELF-aligned)", None)
        pretty_print(top_neg_indices, "Top tokens on NEGATIVE side (OTHER-aligned)", None)


# ---------------- MAIN ----------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Logit-lens analysis of Self–Other direction.")
    parser.add_argument(
        "--direction-file",
        type=str,
        default=DEFAULT_DIRECTION_PATH,
        help=("Path to .pt file saved by self_other_direction.py " "(default: %(default)s)"),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Number of tokens to show on each side (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    direction_path = Path(args.direction_file)
    if not direction_path.exists():
        raise FileNotFoundError(f"Direction file not found: {direction_path}")

    # 1) Load direction
    data = load_direction(direction_path)
    direction = data["direction"]

    # 2) Load model + tokenizer
    tokenizer, model = load_model_and_tokenizer(args.model_id)

    # 3) Analyze via logit lens
    output_file = direction_path.with_name(f"logit_lens_{direction_path.stem}.txt")
    print(f"Saving results to {output_file}")

    with open(output_file, "w") as f:
        # Redirect stdout to file temporarily or just pass file handle
        # Let's modify analyze_direction_with_logit_lens to accept a file handle
        pass

    analyze_direction_with_logit_lens(
        direction, tokenizer, model, top_k=args.top_k, output_path=output_file
    )


if __name__ == "__main__":
    main()
