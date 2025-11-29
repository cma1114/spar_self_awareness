#!/usr/bin/env python3
"""
Logit Lens Heatmap Visualization

Creates a heatmap showing top-K tokens at each layer when projecting
a direction vector through the unembedding matrix.

Inspired by: https://github.com/cma1114/activation_steering/blob/main/code/steering-honesty.ipynb
"""

import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompt_utils import build_self_prompt, build_other_prompt


# --------- CONFIG ---------
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = "llama-3.3-70b-instruct"
OUTPUT_DIR = Path("interp/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Visualization params
TOP_K = 10
LAYERS_TO_SHOW = list(range(0, 80, 2))  # Every 2nd layer to fit on screen, or use range(80) for all


def load_model():
    """Load model and tokenizer."""
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


def load_question_text(model_name: str) -> dict[str, str]:
    """Load question texts from compiled results."""
    path = Path("compiled_results_smc") / f"{model_name}_phase1_compiled.json"
    with open(path) as f:
        data = json.load(f)
    return {k: v["question"] for k, v in data["results"].items()}


@torch.no_grad()
def get_hidden_states_all_layers(model, tokenizer, prompt: str) -> torch.Tensor:
    """
    Get hidden states at the last token for all layers.

    Returns:
        Tensor of shape (num_layers, hidden_dim) - excludes embedding layer
    """
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    outputs = model(**inputs, output_hidden_states=True)

    # Stack layers 1-80 (skip embeddings at index 0)
    h_all = []
    for layer_idx in range(len(outputs.hidden_states) - 1):
        h = outputs.hidden_states[layer_idx + 1][:, -1, :].squeeze(0)
        h_all.append(h)

    return torch.stack(h_all)  # (80, hidden_dim)


@torch.no_grad()
def compute_direction_all_layers(
    model, tokenizer, prompts_a: list[str], prompts_b: list[str]
) -> torch.Tensor:
    """
    Compute direction = mean(A) - mean(B) at each layer.

    Returns:
        Tensor of shape (num_layers, hidden_dim)
    """
    print(f"Computing directions from {len(prompts_a)} pairs...")

    a_states = []
    b_states = []

    for i, (p_a, p_b) in enumerate(zip(prompts_a, prompts_b)):
        if (i + 1) % 20 == 0:
            print(f"  Processing {i+1}/{len(prompts_a)}")

        h_a = get_hidden_states_all_layers(model, tokenizer, p_a)
        h_b = get_hidden_states_all_layers(model, tokenizer, p_b)
        a_states.append(h_a)
        b_states.append(h_b)

    # Stack and compute mean: (num_samples, num_layers, hidden_dim) -> (num_layers, hidden_dim)
    a_mean = torch.stack(a_states).mean(dim=0)
    b_mean = torch.stack(b_states).mean(dim=0)

    direction = a_mean - b_mean
    # Normalize each layer's direction
    direction = direction / direction.norm(dim=-1, keepdim=True)

    return direction


@torch.no_grad()
def create_logit_lens_heatmap(
    model,
    tokenizer,
    directions: torch.Tensor,  # (num_layers, hidden_dim)
    title: str,
    output_path: Path,
    top_k: int = TOP_K,
    layers_to_show: list[int] = None,
    multiplier: float = 1.0,
    highlight_layers: list[list[int]] = None,
    apply_layer_norm: bool = True,
):
    """
    Create a heatmap visualization of logit lens results.

    Args:
        model: The language model
        tokenizer: The tokenizer
        directions: Direction vectors for each layer, shape (num_layers, hidden_dim)
        title: Plot title
        output_path: Where to save the plot
        top_k: Number of top tokens to show per layer
        layers_to_show: Which layers to include (None = all)
        multiplier: Scale factor for the direction vector
        highlight_layers: List of [start, end] layer ranges to highlight with red boxes
        apply_layer_norm: Whether to apply RMSNorm before unembedding (recommended for Llama)
    """
    if layers_to_show is None:
        layers_to_show = list(range(directions.shape[0]))

    token_data = []
    probs_data = []

    print(f"Computing logit lens for {len(layers_to_show)} layers...")

    for layer_idx in layers_to_show:
        vec = (multiplier * directions[layer_idx]).to(model.device).to(model.dtype)

        # Apply layer norm if requested (important for Llama models)
        if apply_layer_norm:
            # For Llama: model.model.norm is the final RMSNorm
            vec_normed = model.model.norm(vec)
        else:
            vec_normed = vec

        # Project through unembedding
        logits = model.lm_head(vec_normed.unsqueeze(0)).squeeze(0)

        # Softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Get top-k
        values, indices = torch.topk(probs, top_k)

        probs_percent = values.cpu().tolist()
        tokens = [tokenizer.decode([idx.item()]).replace('\n', '\\n') for idx in indices]

        token_data.append(tokens)
        probs_data.append(probs_percent)

    # Convert to numpy for plotting
    probs_array = np.array(probs_data)
    token_labels = np.array(token_data)

    # Create plot
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig_height = max(8, len(layers_to_show) * 0.3)
    plt.figure(figsize=(14, fig_height), dpi=150)

    # Choose colormap based on multiplier sign
    colorscale = "Reds" if multiplier >= 0 else "Blues"

    ax = sns.heatmap(
        probs_array,
        annot=token_labels,
        fmt='',
        cmap=colorscale,
        xticklabels=[f"#{i+1}" for i in range(top_k)],
        yticklabels=[f"L{l}" for l in layers_to_show],
        cbar_kws={'label': 'Probability'},
        annot_kws={'size': 8},
    )

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel(f"Top {top_k} Tokens", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)

    # Add highlight boxes for specific layers
    if highlight_layers:
        for layer_range in highlight_layers:
            start_idx = layers_to_show.index(layer_range[0]) if layer_range[0] in layers_to_show else None
            if start_idx is not None:
                num_layers = len(layer_range)
                rect = patches.Rectangle(
                    (0, start_idx), top_k, num_layers,
                    linewidth=3, edgecolor='lime', facecolor='none'
                )
                ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()


def create_comparison_heatmap(
    model,
    tokenizer,
    directions_a: torch.Tensor,  # e.g., Self-Other direction
    directions_b: torch.Tensor,  # e.g., Confidence direction
    title_a: str,
    title_b: str,
    output_path: Path,
    top_k: int = 8,
    layers_to_show: list[int] = None,
):
    """
    Create side-by-side heatmaps comparing two different directions.
    """
    if layers_to_show is None:
        layers_to_show = list(range(0, 80, 4))  # Every 4th layer

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, max(8, len(layers_to_show) * 0.3)), dpi=150)

    for ax, directions, title in [(ax1, directions_a, title_a), (ax2, directions_b, title_b)]:
        token_data = []
        probs_data = []

        for layer_idx in layers_to_show:
            vec = directions[layer_idx].to(model.device).to(model.dtype)
            vec_normed = model.model.norm(vec)
            logits = model.lm_head(vec_normed.unsqueeze(0)).squeeze(0)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            values, indices = torch.topk(probs, top_k)

            tokens = [tokenizer.decode([idx.item()]).replace('\n', '\\n') for idx in indices]
            token_data.append(tokens)
            probs_data.append(values.cpu().tolist())

        probs_array = np.array(probs_data)
        token_labels = np.array(token_data)

        sns.heatmap(
            probs_array,
            annot=token_labels,
            fmt='',
            cmap='Reds',
            xticklabels=False,
            yticklabels=[f"L{l}" for l in layers_to_show],
            cbar_kws={'label': 'Prob'},
            annot_kws={'size': 7},
            ax=ax,
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(f"Top {top_k} Tokens")

    ax1.set_ylabel("Layer")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()


# --------- MAIN ---------

def main():
    """
    Generate logit lens heatmaps for Self-Other and Confidence directions.
    """
    import pandas as pd

    print("=== Logit Lens Heatmap Visualization ===\n")

    # Load model
    tokenizer, model = load_model()

    # Load data
    q_text_map = load_question_text(MODEL_NAME)

    # --- Option 1: Load pre-computed directions ---
    # If you have saved direction .pt files:
    # directions = torch.load("interp/outputs/self_other_direction_layer35.pt")["direction"]
    # But this only gives you one layer. For the heatmap, we need all layers.

    # --- Option 2: Compute directions for all layers ---
    # Load your contrastive data
    DATA_DIR = Path("contrastive_pairs") / MODEL_NAME

    # For Self-Other direction
    diff_train = pd.read_csv(DATA_DIR / f"{MODEL_NAME}_different_perspective_train.csv")

    prompts_self = []
    prompts_other = []
    for _, row in diff_train.iterrows():
        qid = row["question_id"]
        if qid not in q_text_map:
            continue
        text = q_text_map[qid]
        prompts_self.append(build_self_prompt(text))
        prompts_other.append(build_other_prompt(text))

    print("\nComputing Self-Other directions...")
    directions_so = compute_direction_all_layers(model, tokenizer, prompts_self, prompts_other)

    # Create Self-Other heatmap
    create_logit_lens_heatmap(
        model, tokenizer, directions_so,
        title="Self-Other Direction: Logit Lens (Self - Other)",
        output_path=OUTPUT_DIR / "logit_lens_heatmap_self_other.png",
        top_k=10,
        layers_to_show=list(range(0, 80, 2)),  # Every 2nd layer
        highlight_layers=[[34, 35, 36]],  # Highlight peak layers
    )

    # For Confidence direction (if you have introspective extremes data)
    intro_train_path = DATA_DIR / f"{MODEL_NAME}_introspective_extremes_AB_train.csv"
    if intro_train_path.exists():
        intro_train = pd.read_csv(intro_train_path)

        prompts_a = []  # High confidence
        prompts_b = []  # Low confidence
        for _, row in intro_train.iterrows():
            if row["A_qid"] not in q_text_map or row["B_qid"] not in q_text_map:
                continue
            prompts_a.append(build_self_prompt(q_text_map[row["A_qid"]]))
            prompts_b.append(build_self_prompt(q_text_map[row["B_qid"]]))

        print("\nComputing Confidence directions...")
        directions_conf = compute_direction_all_layers(model, tokenizer, prompts_a, prompts_b)

        # Create Confidence heatmap
        create_logit_lens_heatmap(
            model, tokenizer, directions_conf,
            title="Confidence Direction: Logit Lens (High - Low)",
            output_path=OUTPUT_DIR / "logit_lens_heatmap_confidence.png",
            top_k=10,
            layers_to_show=list(range(0, 80, 2)),
            highlight_layers=[[30, 31, 32, 33, 34, 35]],  # Where AUC hits 1.0
        )

        # Create side-by-side comparison
        create_comparison_heatmap(
            model, tokenizer,
            directions_so, directions_conf,
            title_a="Self-Other Direction",
            title_b="Confidence Direction",
            output_path=OUTPUT_DIR / "logit_lens_heatmap_comparison.png",
            top_k=8,
            layers_to_show=list(range(0, 80, 4)),  # Every 4th layer for readability
        )

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
