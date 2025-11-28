#!/usr/bin/env python3
"""
Analyze Introspective Extremes vs Self-Other Direction (Layer Sweep)

This script:
1. Loads "Introspective Extremes" pairs (Group A vs Group B).
2. Loads "Self-Other" pairs (SAME vs DIFFERENT).
3. Caches activations for all prompts across all layers (0-79).
4. Computes the "Introspective Confidence" direction (d_conf) and "Self-Other" direction (d_SO) per layer.
5. Validates both directions using AUC on test sets.
6. Compares d_conf and d_SO using Cosine Similarity per layer.
7. Saves results and plots the comparison.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompt_utils import build_other_prompt, build_self_prompt


# --------- CONFIG ---------
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = "llama-3.3-70b-instruct"

# Paths
DATA_DIR = Path("contrastive_pairs") / MODEL_NAME
INTRO_TRAIN = DATA_DIR / f"{MODEL_NAME}_introspective_extremes_AB_train.csv"
INTRO_TEST = DATA_DIR / f"{MODEL_NAME}_introspective_extremes_AB_test.csv"

DIFF_TRAIN = DATA_DIR / f"{MODEL_NAME}_different_perspective_train.csv"
SAME_TEST = DATA_DIR / f"{MODEL_NAME}_same_perspective_test.csv"
DIFF_TEST = DATA_DIR / f"{MODEL_NAME}_different_perspective_test.csv"

COMPILED_RESULTS = Path("compiled_results_smc") / f"{MODEL_NAME}_phase1_compiled.json"
OUTPUT_DIR = Path("interp/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PromptItem:
    id: str
    prompt_text: str
    group: str  # "A", "B", "Self", "Other"


def load_question_text(model_name: str) -> dict[str, str]:
    path = Path("compiled_results_smc") / f"{model_name}_phase1_compiled.json"
    with open(path) as f:
        data = json.load(f)
    # Map qid -> question text
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
def cache_activations(model, tokenizer, unique_prompts: list[str]) -> dict[str, torch.Tensor]:
    """
    Run model on unique prompts and cache hidden states for all layers.
    Returns: Dict[prompt_str -> tensor(num_layers, hidden_dim)]
    """
    cache = {}
    print(f"Caching activations for {len(unique_prompts)} unique prompts...")

    for i, prompt in enumerate(tqdm(unique_prompts)):
        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        outputs = model(**inputs, output_hidden_states=True)

        # Stack hidden states: (num_layers+1, batch=1, seq_len, hidden_dim)
        # We take the last token: (num_layers+1, hidden_dim)
        # Layer 0 is embeddings, Layer 1 is first transformer block output...
        # We usually want layers 1..N (indices 1 to 80 for Llama-3-70B)
        # Let's store all of them to be safe, indexed 0..80

        # Store outputs from all layers (skipping embeddings at index 0)
        # HF convention: hidden_states[i+1] is the output of layer i.
        h_all = []

        for layer_idx in range(len(outputs.hidden_states) - 1):
            # layer_idx 0 -> hidden_states[1]
            h = outputs.hidden_states[layer_idx + 1][:, -1, :].cpu().squeeze(0)
            h_all.append(h)

        cache[prompt] = torch.stack(h_all)  # (80, hidden_dim)

    return cache


def main():
    print("=== Analyzing Introspective Extremes vs Self-Other ===")

    # 1. Load Data
    q_text_map = load_question_text(MODEL_NAME)

    # Introspective Data
    intro_train = pd.read_csv(INTRO_TRAIN)
    intro_test = pd.read_csv(INTRO_TEST)
    print(f"Loaded Intro Extremes: Train={len(intro_train)}, Test={len(intro_test)}")

    # Self-Other Data
    diff_train = pd.read_csv(DIFF_TRAIN)
    same_test = pd.read_csv(SAME_TEST)
    diff_test = pd.read_csv(DIFF_TEST)
    print(
        f"Loaded Self-Other: DiffTrain={len(diff_train)}, SameTest={len(same_test)}, DiffTest={len(diff_test)}"
    )

    # 2. Collect Unique Prompts
    prompts_to_cache = set()

    # Helper to add prompts
    def add_prompts(df, qid_col, type="self"):
        for qid in df[qid_col]:
            if qid not in q_text_map:
                continue
            text = q_text_map[qid]
            if type == "self":
                prompts_to_cache.add(build_self_prompt(text))
            elif type == "other":
                prompts_to_cache.add(build_other_prompt(text))

    # Intro (A/B are both Self prompts)
    add_prompts(intro_train, "A_qid", "self")
    add_prompts(intro_train, "B_qid", "self")
    add_prompts(intro_test, "A_qid", "self")
    add_prompts(intro_test, "B_qid", "self")

    # Self-Other (DiffTrain uses Self & Other)
    add_prompts(diff_train, "question_id", "self")
    add_prompts(diff_train, "question_id", "other")

    # Self-Other Validation (SameTest/DiffTest use Self & Other)
    add_prompts(same_test, "question_id", "self")
    add_prompts(same_test, "question_id", "other")
    add_prompts(diff_test, "question_id", "self")
    add_prompts(diff_test, "question_id", "other")

    sorted_prompts = sorted(list(prompts_to_cache))
    print(f"Total unique prompts to cache: {len(sorted_prompts)}")

    # 3. Cache Activations
    tokenizer, model = load_model()
    cache = cache_activations(model, tokenizer, sorted_prompts)

    # 4. Layer Sweep Analysis
    results = []
    num_layers = 80

    print("\nRunning layer sweep...")
    for layer in tqdm(range(num_layers)):
        # --- A. Introspective Confidence (d_conf) ---
        # Train d_conf
        a_vecs, b_vecs = [], []
        for _, row in intro_train.iterrows():
            if row["A_qid"] not in q_text_map or row["B_qid"] not in q_text_map:
                continue
            p_a = build_self_prompt(q_text_map[row["A_qid"]])
            p_b = build_self_prompt(q_text_map[row["B_qid"]])
            a_vecs.append(cache[p_a][layer])
            b_vecs.append(cache[p_b][layer])

        if not a_vecs:
            print(f"Warning: No valid intro pairs for layer {layer}")
            continue

        mu_a = torch.stack(a_vecs).mean(dim=0)
        mu_b = torch.stack(b_vecs).mean(dim=0)
        d_conf = mu_a - mu_b
        d_conf = d_conf / d_conf.norm()

        # Test AUC_conf
        y_true, y_scores = [], []
        for _, row in intro_test.iterrows():
            if row["A_qid"] not in q_text_map or row["B_qid"] not in q_text_map:
                continue
            p_a = build_self_prompt(q_text_map[row["A_qid"]])
            p_b = build_self_prompt(q_text_map[row["B_qid"]])

            s_a = torch.dot(d_conf, cache[p_a][layer]).item()
            s_b = torch.dot(d_conf, cache[p_b][layer]).item()

            y_true.extend([1, 0])
            y_scores.extend([s_a, s_b])

        auc_conf = roc_auc_score(y_true, y_scores) if y_true else 0.5
        if auc_conf < 0.5:
            auc_conf = 1.0 - auc_conf

        # --- B. Self-Other (d_SO) ---
        # Train d_SO (using DIFFERENT pairs)
        s_vecs, o_vecs = [], []
        for _, row in diff_train.iterrows():
            qid = row["question_id"]
            if qid not in q_text_map:
                continue
            p_s = build_self_prompt(q_text_map[qid])
            p_o = build_other_prompt(q_text_map[qid])
            s_vecs.append(cache[p_s][layer])
            o_vecs.append(cache[p_o][layer])

        mu_s = torch.stack(s_vecs).mean(dim=0)
        mu_o = torch.stack(o_vecs).mean(dim=0)
        d_so = mu_s - mu_o
        d_so = d_so / d_so.norm()

        # Test AUC_SO (SAME vs DIFFERENT)
        y_true_so, y_scores_so = [], []

        # Add DIFFERENT test samples (Label 1)
        for _, row in diff_test.iterrows():
            qid = row["question_id"]
            if qid not in q_text_map:
                continue
            p_s = build_self_prompt(q_text_map[qid])
            p_o = build_other_prompt(q_text_map[qid])
            delta = cache[p_s][layer] - cache[p_o][layer]
            score = float(torch.dot(d_so, delta).item())
            y_true_so.append(1)
            y_scores_so.append(score)

        # Add SAME test samples (Label 0)
        for _, row in same_test.iterrows():
            qid = row["question_id"]
            if qid not in q_text_map:
                continue
            p_s = build_self_prompt(q_text_map[qid])
            p_o = build_other_prompt(q_text_map[qid])
            delta = cache[p_s][layer] - cache[p_o][layer]
            score = float(torch.dot(d_so, delta).item())
            y_true_so.append(0)
            y_scores_so.append(score)

        auc_so = roc_auc_score(y_true_so, y_scores_so) if y_true_so else 0.5
        if auc_so < 0.5:
            auc_so = 1.0 - auc_so

        # --- C. Comparison ---
        cosine_sim = torch.nn.functional.cosine_similarity(d_conf, d_so, dim=0).item()

        results.append(
            {
                "layer": layer,
                "auc_conf": auc_conf,
                "auc_so": auc_so,
                "cosine_sim": cosine_sim,
                "n_test_conf": len(y_true),
                "n_test_so": len(y_true_so),
            }
        )

    # 5. Save Results
    df_res = pd.DataFrame(results)
    json_path = OUTPUT_DIR / "introspective_vs_self_other_sweep.json"
    df_res.to_json(json_path, orient="records", indent=2)
    print(f"Saved results to {json_path}")

    # 6. Plot
    # Set style
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot AUCs
    ax1.plot(
        df_res["layer"],
        df_res["auc_conf"],
        label="AUC: Introspective (A vs B)",
        color="#2E86C1",  # Strong Blue
        linewidth=2.5,
    )
    ax1.plot(
        df_res["layer"],
        df_res["auc_so"],
        label="AUC: Self-Other (Diff vs Same)",
        color="#27AE60",  # Strong Green
        linewidth=2.5,
        linestyle="--",
    )

    # Plot Cosine (on secondary axis)
    ax2 = ax1.twinx()
    ax2.plot(
        df_res["layer"],
        df_res["cosine_sim"],
        label="Cosine Sim (Conf vs SO)",
        color="#E74C3C",  # Strong Red
        linewidth=2,
        alpha=0.8,
    )

    # Axes limits and labels
    ax1.set_xlabel("Layer Index", fontsize=12)
    ax1.set_ylabel("AUC", fontsize=12)
    ax2.set_ylabel("Cosine Similarity", fontsize=12)

    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, 80)
    ax2.set_ylim(-1.05, 1.05)

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="lower right",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
    )

    plt.title(
        f"Introspective Confidence vs Self-Other Direction ({MODEL_NAME})", fontsize=14, pad=15
    )
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "introspective_vs_self_other_plot.png"
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    print("Done!")


if __name__ == "__main__":
    main()
