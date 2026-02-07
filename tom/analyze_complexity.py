#!/usr/bin/env python3
"""
Analyze scenario complexity from ToM game JSON files.

Produces 4 charts:
1. Average event count per extra condition (aggregate)
2. Average event count per extra condition by scenario
3. Average epistemic changes per extra condition (aggregate)
4. Average epistemic changes per extra condition by scenario

Usage:
    python analyze_complexity.py <directory>

Where <directory> contains *_game_data.json files. Charts are saved to the same directory.
"""

import json
import glob
import math
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CONDITIONS = ['0A', '0B', '1A', '1B']
COLORS = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']


def load_records(directory):
    """Load all player-A records from game_data.json files in directory."""
    pattern = os.path.join(directory, '*_game_data.json')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No *_game_data.json files found in {directory}")
        sys.exit(1)

    records = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        for r in data.get('turn_records', data if isinstance(data, list) else []):
            if r.get('character') == 'A' and r.get('extra') in CONDITIONS:
                records.append(r)

    print(f"Loaded {len(records)} player-A records from {len(files)} files")
    return records


def aggregate_bar(values_by_cond, ylabel, title, outpath):
    """Bar chart of mean values per condition with 95% CI error bars."""
    means, cis, ns = [], [], []
    for c in CONDITIONS:
        vals = values_by_cond[c]
        m = np.mean(vals) if vals else 0
        ci = 1.96 * np.std(vals) / math.sqrt(len(vals)) if vals else 0
        means.append(m)
        cis.append(ci)
        ns.append(len(vals))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(CONDITIONS, means, yerr=cis, capsize=6,
                  color=COLORS, edgecolor='white', linewidth=1.5)

    for bar, m, ci in zip(bars, means, cis):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci + 0.05,
                f'{m:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    for i, n in enumerate(ns):
        ax.text(i, -0.12, f'n={n}', ha='center', va='top', fontsize=10,
                color='gray', transform=ax.get_xaxis_transform())

    ax.set_xlabel('Extra Condition', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(means) * 1.3 if max(means) > 0 else 1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outpath}")


def per_scenario_bar(values_by_sid_cond, ylabel, title, outpath):
    """Grouped bar chart of mean values per scenario and condition."""
    sids = sorted(values_by_sid_cond.keys())

    means = {c: [] for c in CONDITIONS}
    cis = {c: [] for c in CONDITIONS}
    for sid in sids:
        for c in CONDITIONS:
            vals = values_by_sid_cond[sid].get(c, [])
            m = np.mean(vals) if vals else 0
            ci = 1.96 * np.std(vals) / math.sqrt(len(vals)) if vals else 0
            means[c].append(m)
            cis[c].append(ci)

    fig, ax = plt.subplots(figsize=(18, 7))
    x = np.arange(len(sids))
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for i, c in enumerate(CONDITIONS):
        ax.bar(x + offsets[i] * width, means[c], width,
               yerr=cis[c], capsize=2, label=c, color=COLORS[i],
               edgecolor='white', linewidth=0.5, alpha=0.85)

    ax.set_xlabel('Scenario ID', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sids, fontsize=8)
    ax.legend(title='Condition', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outpath}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    records = load_records(directory)

    # Collect values
    events_by_cond = defaultdict(list)
    ect_by_cond = defaultdict(list)
    events_by_sid_cond = defaultdict(lambda: defaultdict(list))
    ect_by_sid_cond = defaultdict(lambda: defaultdict(list))

    for r in records:
        cond = r['extra']
        sid = int(r['scenario_id'])
        sec = r.get('situation_event_count')
        ect = r.get('ect_total')

        if sec is not None:
            events_by_cond[cond].append(sec)
            events_by_sid_cond[sid][cond].append(sec)
        if ect is not None:
            ect_by_cond[cond].append(ect)
            ect_by_sid_cond[sid][cond].append(ect)

    # Generate charts
    print("\nGenerating charts...")

    aggregate_bar(
        events_by_cond,
        'Mean Event Count',
        'Average Number of Events by Extra Condition',
        os.path.join(directory, 'events_by_condition.png')
    )

    per_scenario_bar(
        events_by_sid_cond,
        'Mean Event Count',
        'Average Number of Events by Scenario and Extra Condition',
        os.path.join(directory, 'events_by_scenario_condition.png')
    )

    aggregate_bar(
        ect_by_cond,
        'Mean ECT Total',
        'Average Epistemic Changes by Extra Condition',
        os.path.join(directory, 'ect_by_condition.png')
    )

    per_scenario_bar(
        ect_by_sid_cond,
        'Mean ECT Total',
        'Epistemic Changes by Scenario and Extra Condition',
        os.path.join(directory, 'ect_by_scenario_condition.png')
    )

    print("\nDone.")


if __name__ == '__main__':
    main()
