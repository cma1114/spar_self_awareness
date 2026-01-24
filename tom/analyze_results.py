#!/usr/bin/env python3
"""
Analyze ToM test results across all models.

Reads game_data.json files from tom_llm_logs/, aggregates by model,
and produces a comparison table and bar chart.
"""

import json
import glob
import os
import re
import math
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Return (lower, upper) bounds of the 95% Wilson CI."""
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    half_width = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    lower = (centre - half_width) / denom
    upper = (centre + half_width) / denom
    return lower, upper


def extract_model_name(filepath: str) -> str:
    """Extract model name from filename, stripping timestamp."""
    basename = os.path.basename(filepath)
    # Pattern: {model}_{timestamp}_game_data.json
    match = re.match(r'(.+?)_\d+_game_data\.json$', basename)
    if match:
        return match.group(1)
    return basename


def load_game_data(filepath: str) -> List[dict]:
    """Load and return records from a game_data.json file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def filter_player_a(records: List[dict]) -> List[dict]:
    """Filter to only player A's records."""
    return [r for r in records if r.get('character') == 'A']


def compute_stats(records: List[dict]) -> dict:
    """Compute optimal rates and CIs for a set of records."""
    n_total = len(records)
    n_optimal = sum(1 for r in records if r.get('was_optimal'))

    # Split by extra
    extra0 = [r for r in records if r.get('extra') is None or r.get('extra') == 0]
    extra1 = [r for r in records if r.get('extra') == 1]

    n_extra0 = len(extra0)
    n_extra1 = len(extra1)
    n_opt_extra0 = sum(1 for r in extra0 if r.get('was_optimal'))
    n_opt_extra1 = sum(1 for r in extra1 if r.get('was_optimal'))

    # Compute rates and CIs
    overall_rate = n_optimal / n_total if n_total > 0 else 0
    overall_ci = wilson_ci(n_optimal, n_total)

    extra0_rate = n_opt_extra0 / n_extra0 if n_extra0 > 0 else 0
    extra0_ci = wilson_ci(n_opt_extra0, n_extra0)

    extra1_rate = n_opt_extra1 / n_extra1 if n_extra1 > 0 else 0
    extra1_ci = wilson_ci(n_opt_extra1, n_extra1)

    return {
        'n': n_total,
        'overall': {'rate': overall_rate, 'ci': overall_ci, 'n': n_total, 'k': n_optimal},
        'extra0': {'rate': extra0_rate, 'ci': extra0_ci, 'n': n_extra0, 'k': n_opt_extra0},
        'extra1': {'rate': extra1_rate, 'ci': extra1_ci, 'n': n_extra1, 'k': n_opt_extra1},
    }


# ToM Mastery Categories
# Each category tests a specific aspect of Theory of Mind understanding
# Format: (scenario_ids, weight) where weight is used for chance-level normalization

TOM_MASTERY_CATEGORIES = {
    # 1. Self Knowledge vs Belief: Knowing the distinction between one's own knowledge and belief
    # Pass in 7-9 (Self knows, so pass), Ask teammate in 12-13 (Self only believes, teammate knows)
    # 7-9 worth 2/3 as much as 12-13 (weight 2 vs weight 3, so each group contributes equally)
    'self_knowledge_belief': {
        'name': 'Self: Knowledge vs Belief',
        'description': 'Distinguishing own knowledge from mere belief',
        'components': [
            {'scenarios': [7, 8, 9], 'action': 'Pass', 'weight': 2},   # 3 scenarios × weight 2 = 6
            {'scenarios': [12, 13], 'action': 'Ask', 'weight': 3},    # 2 scenarios × weight 3 = 6
        ],
    },

    # 2. Teammate Knowledge vs Belief: Knowing the distinction between teammate's knowledge and belief
    # Pass in 20-22 (teammate knows), Tell teammate in 17-19 (teammate believes false)
    'teammate_knowledge_belief': {
        'name': 'Teammate: Knowledge vs Belief',
        'description': 'Distinguishing teammate knowledge from belief',
        'components': [
            {'scenarios': [20, 21, 22], 'action': 'Pass', 'weight': 1},
            {'scenarios': [17, 18, 19], 'action': 'Tell', 'weight': 1},
        ],
    },

    # 3. Self + Teammate Uncertainty: Combined uncertainty handling
    # Pass in 10, 11, 23, 24 (all involve uncertainty - self believes or teammate unknown)
    'combined_uncertainty': {
        'name': 'Combined Uncertainty',
        'description': 'Handling self + teammate uncertainty together',
        'components': [
            {'scenarios': [10, 11, 23, 24], 'action': 'Pass', 'weight': 1},
        ],
    },

    # 4. True vs False Belief: Knowing distinction between true and false belief
    # Pass in 14-16 (teammate believes truth), Tell in 17-19 (teammate believes false)
    'true_false_belief': {
        'name': 'True vs False Belief',
        'description': 'Distinguishing true belief from false belief',
        'components': [
            {'scenarios': [14, 15, 16], 'action': 'Pass', 'weight': 1},
            {'scenarios': [17, 18, 19], 'action': 'Tell', 'weight': 1},
        ],
    },

    # 5. Teammate vs Opponent: Distinguishing teammate from opponent
    # Ask teammate in 12, 13 (teammate knows, should ask them not opponent)
    # Pass in 37, 39 (opponent knows but shouldn't tell them)
    # Pass in 30-32 (opponent scenarios, should pass)
    # Tell teammate in 17-19 (help teammate, not opponent)
    'teammate_opponent': {
        'name': 'Teammate vs Opponent',
        'description': 'Treating teammate differently from opponent',
        'components': [
            {'scenarios': [12, 13], 'action': 'Ask', 'weight': 1},
            {'scenarios': [37, 39], 'action': 'Pass', 'weight': 1},
            {'scenarios': [30, 31, 32], 'action': 'Pass', 'weight': 1},
            {'scenarios': [17, 18, 19], 'action': 'Tell', 'weight': 1},
        ],
    },
}


def compute_mastery_score(records: List[dict], category: dict, extra_filter: int = None) -> dict:
    """
    Compute mastery score for a single ToM category.

    Args:
        records: List of game records
        category: Category definition with components
        extra_filter: If specified, only include records with this Extra value (0 or 1)

    Returns dict with:
    - score: weighted accuracy (0-1)
    - n: total weighted trials
    - k: weighted correct
    - by_component: breakdown by component
    """
    # Apply extra filter if specified
    if extra_filter is not None:
        records = [r for r in records if (r.get('extra') or 0) == extra_filter]

    total_weighted = 0
    correct_weighted = 0
    by_component = []

    for component in category['components']:
        scenarios = component['scenarios']
        required_action = component['action']
        weight = component['weight']

        # Filter records for these scenarios
        component_records = [r for r in records
                           if r.get('scenario_id') in [str(s) for s in scenarios]]

        n_total = len(component_records)
        n_correct = sum(1 for r in component_records if r.get('was_optimal'))

        weighted_total = n_total * weight
        weighted_correct = n_correct * weight

        total_weighted += weighted_total
        correct_weighted += weighted_correct

        rate = n_correct / n_total if n_total > 0 else 0
        by_component.append({
            'scenarios': scenarios,
            'action': required_action,
            'n': n_total,
            'k': n_correct,
            'rate': rate,
            'weight': weight,
        })

    score = correct_weighted / total_weighted if total_weighted > 0 else 0

    return {
        'score': score,
        'n': total_weighted,
        'k': correct_weighted,
        'by_component': by_component,
    }


def compute_all_mastery_scores(records: List[dict], extra_filter: int = None) -> dict:
    """Compute all ToM mastery scores for a set of records."""
    results = {}
    for key, category in TOM_MASTERY_CATEGORIES.items():
        results[key] = compute_mastery_score(records, category, extra_filter)
        results[key]['name'] = category['name']
        results[key]['description'] = category['description']
    return results


def compute_mastery_with_extra_breakout(records: List[dict]) -> dict:
    """Compute mastery scores with overall, Extra=0, and Extra=1 breakouts."""
    return {
        'overall': compute_all_mastery_scores(records),
        'extra0': compute_all_mastery_scores(records, extra_filter=0),
        'extra1': compute_all_mastery_scores(records, extra_filter=1),
    }


def format_rate_ci(rate: float, ci: Tuple[float, float]) -> str:
    """Format rate with CI as string like '85.3% (±2.5)'."""
    half_width = (ci[1] - ci[0]) / 2 * 100
    return f"{rate*100:5.1f}% (±{half_width:4.1f})"


def find_game_data_files(base_dir: str) -> List[str]:
    """Find all game_data.json files recursively."""
    pattern = os.path.join(base_dir, '**', '*_game_data.json')
    return glob.glob(pattern, recursive=True)


def main():
    # Find the tom_llm_logs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, 'tom_llm_logs')

    if not os.path.exists(logs_dir):
        print(f"Error: {logs_dir} not found")
        return

    # Find all game data files
    files = find_game_data_files(logs_dir)
    print(f"Found {len(files)} game_data.json files")

    # Aggregate records by model
    model_records: Dict[str, List[dict]] = defaultdict(list)

    for filepath in files:
        records = load_game_data(filepath)
        a_records = filter_player_a(records)

        # Only include files where record count is multiple of 78
        if len(a_records) == 0 or len(a_records) % 78 != 0:
            continue

        model_name = extract_model_name(filepath)
        model_records[model_name].extend(a_records)

    if not model_records:
        print("No valid game data files found (need multiples of 78 records)")
        return

    # Compute stats for each model
    model_stats = {}
    for model_name, records in model_records.items():
        model_stats[model_name] = compute_stats(records)

    # Sort by overall rate (descending)
    sorted_models = sorted(model_stats.keys(),
                          key=lambda m: model_stats[m]['overall']['rate'],
                          reverse=True)

    # Print table
    print("\n" + "=" * 90)
    print(f"{'Model':<35} | {'N':>6} | {'Overall':>14} | {'Extra=0':>14} | {'Extra=1':>14}")
    print("-" * 90)

    for model in sorted_models:
        stats = model_stats[model]
        overall_str = format_rate_ci(stats['overall']['rate'], stats['overall']['ci'])
        extra0_str = format_rate_ci(stats['extra0']['rate'], stats['extra0']['ci'])
        extra1_str = format_rate_ci(stats['extra1']['rate'], stats['extra1']['ci'])

        print(f"{model:<35} | {stats['n']:>6} | {overall_str:>14} | {extra0_str:>14} | {extra1_str:>14}")

    print("=" * 90)

    # Aggregate all records for cross-model analysis
    all_records = []
    for records in model_records.values():
        all_records.extend(records)

    # Compute ToM Mastery scores for each model with Extra breakouts
    print("\n" + "=" * 140)
    print("ToM MASTERY SCORES BY CATEGORY (Overall)")
    print("=" * 140)

    # Build category short names
    cat_keys = list(TOM_MASTERY_CATEGORIES.keys())
    cat_short_names = []
    for k in cat_keys:
        name = TOM_MASTERY_CATEGORIES[k]['name']
        short_name = name[:15] if len(name) > 15 else name
        cat_short_names.append(short_name)

    # Header
    header = f"{'Model':<35}"
    for short_name in cat_short_names:
        header += f" | {short_name:>15}"
    print(header)
    print("-" * 140)

    # Compute and store mastery scores for each model
    model_mastery = {}
    mastery_table_lines = [header, "-" * 140]

    for model in sorted_models:
        records = model_records[model]
        mastery = compute_mastery_with_extra_breakout(records)
        model_mastery[model] = mastery

        row = f"{model:<35}"
        for key in cat_keys:
            score = mastery['overall'][key]['score'] * 100
            row += f" | {score:>14.1f}%"
        print(row)
        mastery_table_lines.append(row)

    print("-" * 140)
    mastery_table_lines.append("-" * 140)

    # Compute aggregate mastery scores across all models
    all_mastery = compute_mastery_with_extra_breakout(all_records)
    row = f"{'ALL MODELS':<35}"
    for key in cat_keys:
        score = all_mastery['overall'][key]['score'] * 100
        row += f" | {score:>14.1f}%"
    print(row)
    mastery_table_lines.append(row)
    print("=" * 140)
    mastery_table_lines.append("=" * 140)

    # Print Extra=0 breakdown
    print("\n" + "=" * 140)
    print("ToM MASTERY SCORES BY CATEGORY (Extra=0 - Base Scenarios)")
    print("=" * 140)
    print(header)
    print("-" * 140)

    extra0_table_lines = [header, "-" * 140]
    for model in sorted_models:
        mastery = model_mastery[model]
        row = f"{model:<35}"
        for key in cat_keys:
            score = mastery['extra0'][key]['score'] * 100
            row += f" | {score:>14.1f}%"
        print(row)
        extra0_table_lines.append(row)

    print("-" * 140)
    extra0_table_lines.append("-" * 140)
    row = f"{'ALL MODELS':<35}"
    for key in cat_keys:
        score = all_mastery['extra0'][key]['score'] * 100
        row += f" | {score:>14.1f}%"
    print(row)
    extra0_table_lines.append(row)
    print("=" * 140)
    extra0_table_lines.append("=" * 140)

    # Print Extra=1 breakdown
    print("\n" + "=" * 140)
    print("ToM MASTERY SCORES BY CATEGORY (Extra=1 - Complex Scenarios)")
    print("=" * 140)
    print(header)
    print("-" * 140)

    extra1_table_lines = [header, "-" * 140]
    for model in sorted_models:
        mastery = model_mastery[model]
        row = f"{model:<35}"
        for key in cat_keys:
            score = mastery['extra1'][key]['score'] * 100
            row += f" | {score:>14.1f}%"
        print(row)
        extra1_table_lines.append(row)

    print("-" * 140)
    extra1_table_lines.append("-" * 140)
    row = f"{'ALL MODELS':<35}"
    for key in cat_keys:
        score = all_mastery['extra1'][key]['score'] * 100
        row += f" | {score:>14.1f}%"
    print(row)
    extra1_table_lines.append(row)
    print("=" * 140)
    extra1_table_lines.append("=" * 140)

    # Save mastery tables to files
    mastery_overall_path = os.path.join(logs_dir, 'mastery_overall.txt')
    with open(mastery_overall_path, 'w') as f:
        f.write("ToM MASTERY SCORES BY CATEGORY (Overall)\n")
        f.write("\n".join(mastery_table_lines))
    print(f"\nMastery table (overall) saved to: {mastery_overall_path}")

    mastery_extra0_path = os.path.join(logs_dir, 'mastery_extra0.txt')
    with open(mastery_extra0_path, 'w') as f:
        f.write("ToM MASTERY SCORES BY CATEGORY (Extra=0 - Base Scenarios)\n")
        f.write("\n".join(extra0_table_lines))
    print(f"Mastery table (Extra=0) saved to: {mastery_extra0_path}")

    mastery_extra1_path = os.path.join(logs_dir, 'mastery_extra1.txt')
    with open(mastery_extra1_path, 'w') as f:
        f.write("ToM MASTERY SCORES BY CATEGORY (Extra=1 - Complex Scenarios)\n")
        f.write("\n".join(extra1_table_lines))
    print(f"Mastery table (Extra=1) saved to: {mastery_extra1_path}")

    # Save combined CSV for easy analysis
    mastery_csv_path = os.path.join(logs_dir, 'mastery_scores.csv')
    with open(mastery_csv_path, 'w') as f:
        # Header
        csv_header = "Model,Extra"
        for key in cat_keys:
            csv_header += f",{TOM_MASTERY_CATEGORIES[key]['name']}"
        f.write(csv_header + "\n")

        # Data rows
        for model in sorted_models:
            mastery = model_mastery[model]
            for extra_label, extra_key in [('Overall', 'overall'), ('0', 'extra0'), ('1', 'extra1')]:
                row = f"{model},{extra_label}"
                for key in cat_keys:
                    score = mastery[extra_key][key]['score'] * 100
                    row += f",{score:.1f}"
                f.write(row + "\n")

        # All models aggregate
        for extra_label, extra_key in [('Overall', 'overall'), ('0', 'extra0'), ('1', 'extra1')]:
            row = f"ALL MODELS,{extra_label}"
            for key in cat_keys:
                score = all_mastery[extra_key][key]['score'] * 100
                row += f",{score:.1f}"
            f.write(row + "\n")

    print(f"Mastery CSV saved to: {mastery_csv_path}")

    # Print detailed breakdown for each category (using overall)
    print("\nDetailed Mastery Category Breakdown (All Models, Overall):")
    print("-" * 80)
    for key, category in TOM_MASTERY_CATEGORIES.items():
        mastery = all_mastery['overall'][key]
        print(f"\n{category['name']}: {mastery['score']*100:.1f}%")
        print(f"  {category['description']}")
        for comp in mastery['by_component']:
            scenarios_str = ', '.join(str(s) for s in comp['scenarios'])
            print(f"  - Scenarios [{scenarios_str}] → {comp['action']}: "
                  f"{comp['k']}/{comp['n']} = {comp['rate']*100:.1f}% (weight={comp['weight']})")

    # Compute per-scenario stats across all models
    scenario_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in all_records:
        key = (r.get('scenario_id'), r.get('extra', 0) or 0)
        scenario_stats[key]['total'] += 1
        if r.get('was_optimal'):
            scenario_stats[key]['correct'] += 1

    # Sort by accuracy (ascending = hardest first)
    sorted_scenarios = sorted(
        scenario_stats.items(),
        key=lambda x: x[1]['correct'] / x[1]['total'] if x[1]['total'] > 0 else 0
    )

    print(f"\n{'Scenario':<10} | {'Extra':>5} | {'N':>6} | {'Accuracy':>10}")
    print("-" * 40)
    for (scenario_id, extra), stats in sorted_scenarios:
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{scenario_id:<10} | {extra:>5} | {stats['total']:>6} | {acc*100:>9.1f}%")
    print("-" * 40)

    # Generate scenario difficulty chart
    scenario_labels = []
    scenario_accs = []
    scenario_colors = []
    for (scenario_id, extra), stats in sorted_scenarios:
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        scenario_labels.append(f"{scenario_id}" + ("*" if extra else ""))
        scenario_accs.append(acc * 100)
        scenario_colors.append('#e74c3c' if extra else '#3498db')

    fig2, ax2 = plt.subplots(figsize=(16, 8))
    x2 = np.arange(len(scenario_labels))
    ax2.bar(x2, scenario_accs, color=scenario_colors, edgecolor='white', linewidth=0.5)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    ax2.axhline(y=np.mean(scenario_accs), color='green', linestyle='-', alpha=0.7)
    ax2.set_xlabel('Scenario ID (* = Extra=1)', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Per-Scenario Accuracy Across All Models (sorted by difficulty)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(scenario_labels, rotation=90, fontsize=8)
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3)
    from matplotlib.patches import Patch
    ax2.legend(handles=[
        Patch(facecolor='#3498db', label='Extra=0 (Base)'),
        Patch(facecolor='#e74c3c', label='Extra=1 (Complex)'),
    ], loc='lower right')
    plt.tight_layout()
    scenario_chart_path = os.path.join(logs_dir, 'scenario_difficulty.png')
    plt.savefig(scenario_chart_path, dpi=150, bbox_inches='tight')
    print(f"\nScenario chart saved to: {scenario_chart_path}")
    plt.close(fig2)

    # Generate paired Extra=0 vs Extra=1 comparison chart
    scenario_ids = sorted(set(k[0] for k in scenario_stats.keys()), key=lambda x: int(x))
    extra0_accs = []
    extra1_accs = []
    for sid in scenario_ids:
        acc0 = scenario_stats[(sid, 0)]['correct'] / scenario_stats[(sid, 0)]['total'] * 100 if scenario_stats[(sid, 0)]['total'] > 0 else 0
        acc1 = scenario_stats[(sid, 1)]['correct'] / scenario_stats[(sid, 1)]['total'] * 100 if scenario_stats[(sid, 1)]['total'] > 0 else 0
        extra0_accs.append(acc0)
        extra1_accs.append(acc1)

    fig3, ax3 = plt.subplots(figsize=(16, 8))
    x3 = np.arange(len(scenario_ids))
    width = 0.35
    bars_e0 = ax3.bar(x3 - width/2, extra0_accs, width, label='Extra=0 (Base)', color='#3498db')
    bars_e1 = ax3.bar(x3 + width/2, extra1_accs, width, label='Extra=1 (Complex)', color='#e74c3c')
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Scenario ID', fontsize=12)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_title('Extra=0 vs Extra=1 Accuracy by Scenario (All Models)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(scenario_ids, fontsize=8)
    ax3.set_ylim(0, 105)
    ax3.legend(loc='lower right')
    ax3.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    paired_chart_path = os.path.join(logs_dir, 'extra_comparison.png')
    plt.savefig(paired_chart_path, dpi=150, bbox_inches='tight')
    print(f"Paired comparison chart saved to: {paired_chart_path}")
    plt.close(fig3)

    # Generate model performance bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(sorted_models))
    width = 0.25

    # Extract data for plotting
    overall_rates = [model_stats[m]['overall']['rate'] * 100 for m in sorted_models]
    extra0_rates = [model_stats[m]['extra0']['rate'] * 100 for m in sorted_models]
    extra1_rates = [model_stats[m]['extra1']['rate'] * 100 for m in sorted_models]

    # Error bars (half-width of CI)
    overall_errs = [(model_stats[m]['overall']['ci'][1] - model_stats[m]['overall']['ci'][0]) / 2 * 100
                    for m in sorted_models]
    extra0_errs = [(model_stats[m]['extra0']['ci'][1] - model_stats[m]['extra0']['ci'][0]) / 2 * 100
                   for m in sorted_models]
    extra1_errs = [(model_stats[m]['extra1']['ci'][1] - model_stats[m]['extra1']['ci'][0]) / 2 * 100
                   for m in sorted_models]

    # Create bars
    bars1 = ax.bar(x - width, overall_rates, width, label='Overall',
                   yerr=overall_errs, capsize=3, color='#2ecc71')
    bars2 = ax.bar(x, extra0_rates, width, label='Extra=0 (Base)',
                   yerr=extra0_errs, capsize=3, color='#3498db')
    bars3 = ax.bar(x + width, extra1_rates, width, label='Extra=1 (Complex)',
                   yerr=extra1_errs, capsize=3, color='#e74c3c')

    ax.set_ylabel('Optimal Action Rate (%)', fontsize=12)
    ax.set_title('ToM Test Performance by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save chart
    chart_path = os.path.join(logs_dir, 'performance_comparison.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\nChart saved to: {chart_path}")

    plt.show()


if __name__ == '__main__':
    main()
