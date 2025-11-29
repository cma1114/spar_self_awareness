#!/usr/bin/env python3
"""
Generate NeurIPS-ready plots from local JSON results.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Config
OUTPUT_DIR = Path("interp/outputs")
MODEL_NAME = "llama-3.3-70b-instruct"


def setup_style():
    """Set up professional plotting style."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["figure.dpi"] = 300


def plot_layer_sweep():
    """Plot Self-Other AUC vs Layer."""
    json_path = OUTPUT_DIR / "layer_sweep_results.json"
    if not json_path.exists():
        print(f"Skipping {json_path} (not found)")
        return

    with open(json_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 5))

    # Main line
    plt.plot(df["layer"], df["auc"], color="#2E86C1", linewidth=2.5, label="Self-Other Separation")

    # Random chance
    plt.axhline(
        y=0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.8, label="Random Chance"
    )

    # Peak
    max_auc = df["auc"].max()
    best_layer = df.loc[df["auc"].idxmax(), "layer"]
    plt.plot(
        best_layer,
        max_auc,
        marker="*",
        color="#E74C3C",
        markersize=12,
        linestyle="None",
        label=f"Peak: L{best_layer} ({max_auc:.3f})",
    )

    plt.title(f"Self-Other Separation by Layer ({MODEL_NAME})", fontsize=12, pad=10)
    plt.xlabel("Layer Index", fontsize=10)
    plt.ylabel("AUC (Same vs Different)", fontsize=10)
    plt.ylim(0, 1.05)
    plt.xlim(0, 80)
    plt.legend(frameon=True, fancybox=True, framealpha=0.9, loc="lower right")
    plt.tight_layout()

    out_path = OUTPUT_DIR / "layer_sweep_plot.png"
    plt.savefig(out_path)
    print(f"Saved {out_path}")


def plot_introspective_comparison():
    """Plot Introspective vs Self-Other Comparison."""
    json_path = OUTPUT_DIR / "introspective_vs_self_other_sweep.json"
    if not json_path.exists():
        print(f"Skipping {json_path} (not found)")
        return

    with open(json_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # AUCs on Left Axis
    l1 = ax1.plot(
        df["layer"],
        df["auc_conf"],
        label="AUC: Introspective (A vs B)",
        color="#2E86C1",
        linewidth=2.5,
    )
    l2 = ax1.plot(
        df["layer"],
        df["auc_so"],
        label="AUC: Self-Other (Diff vs Same)",
        color="#27AE60",
        linewidth=2.5,
        linestyle="--",
    )

    ax1.set_xlabel("Layer Index", fontsize=12)
    ax1.set_ylabel("AUC", fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, 80)

    # Cosine on Right Axis
    ax2 = ax1.twinx()
    l3 = ax2.plot(
        df["layer"],
        df["cosine_sim"],
        label="Cosine Sim (Conf vs SO)",
        color="#E74C3C",
        linewidth=2,
        alpha=0.8,
    )

    ax2.set_ylabel("Cosine Similarity", fontsize=12)
    ax2.set_ylim(-1.05, 1.05)

    # Combined Legend
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", frameon=True, fancybox=True, framealpha=0.9)

    plt.title(
        f"Introspective Confidence vs Self-Other Direction ({MODEL_NAME})", fontsize=14, pad=15
    )
    plt.tight_layout()

    out_path = OUTPUT_DIR / "introspective_vs_self_other_plot.png"
    plt.savefig(out_path)
    print(f"Saved {out_path}")


def main():
    setup_style()
    plot_layer_sweep()
    plot_introspective_comparison()


if __name__ == "__main__":
    main()
