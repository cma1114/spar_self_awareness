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
from matplotlib.patches import Patch
import numpy as np


# Extra category metadata - see EXTRA_MAPPING.md
EXTRA_CATEGORIES = {
    '0A': {'name': 'Minimal Events', 'short': 'Min Events', 'color': '#9b59b6'},
    '0B': {'name': 'Event Load', 'short': 'Event Load', 'color': '#f39c12'},
    '1A': {'name': 'Minimal ECT', 'short': 'Min ECT', 'color': '#3498db'},
    '1B': {'name': 'ECT Load', 'short': 'ECT Load', 'color': '#e74c3c'},
}


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
        data = json.load(f)
    # Handle both old (flat list) and new (wrapper object) formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'turn_records' in data:
        return data['turn_records']
    return data


def filter_player_a(records: List[dict]) -> List[dict]:
    """Filter to only player A's records."""
    return [r for r in records if r.get('character') == 'A']


def normalize_extra(val):
    """Convert legacy Extra values to new string format. See EXTRA_MAPPING.md."""
    if val is None or val == 0: return '1A'  # Legacy Extra=0 → 1A
    if val == 1: return '1B'                  # Legacy Extra=1 → 1B
    if val in ('0A', '0B', '1A', '1B'): return val
    return str(val)


def is_success(record: dict, lies_okay: bool = False) -> bool:
    """Check if a record counts as a success.

    Args:
        record: A turn record dict
        lies_okay: If True, lying to opponent answerer also counts as success

    Returns:
        True if was_optimal, or (lies_okay and lied_to_opponent_answerer=="TRUE")
    """
    if record.get('was_optimal'):
        return True
    if lies_okay and record.get('lied_to_opponent_answerer') == "TRUE":
        return True
    return False


def compute_stats(records: List[dict], lies_okay: bool = False) -> dict:
    """Compute optimal rates and CIs for a set of records."""
    n_total = len(records)
    n_optimal = sum(1 for r in records if is_success(r, lies_okay))

    # Split by extra (0A, 0B, 1A, 1B) - see EXTRA_MAPPING.md
    extra0a = [r for r in records if normalize_extra(r.get('extra')) == '0A']
    extra0b = [r for r in records if normalize_extra(r.get('extra')) == '0B']
    extra1a = [r for r in records if normalize_extra(r.get('extra')) == '1A']
    extra1b = [r for r in records if normalize_extra(r.get('extra')) == '1B']

    def compute_extra_stats(recs):
        n = len(recs)
        k = sum(1 for r in recs if is_success(r, lies_okay))
        rate = k / n if n > 0 else 0
        ci = wilson_ci(k, n)
        return {'rate': rate, 'ci': ci, 'n': n, 'k': k}

    # Compute rates and CIs
    overall_rate = n_optimal / n_total if n_total > 0 else 0
    overall_ci = wilson_ci(n_optimal, n_total)

    return {
        'n': n_total,
        'overall': {'rate': overall_rate, 'ci': overall_ci, 'n': n_total, 'k': n_optimal},
        'extra0a': compute_extra_stats(extra0a),
        'extra0b': compute_extra_stats(extra0b),
        'extra1a': compute_extra_stats(extra1a),
        'extra1b': compute_extra_stats(extra1b),
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
    # Pass in 10, 11 (self believes, teammate believes), 23, 24 (self knows, teammate unknown)
    'combined_uncertainty': {
        'name': 'Combined Uncertainty',
        'description': 'Handling self + teammate uncertainty together',
        'components': [
            {'scenarios': [10, 11], 'action': 'Pass', 'weight': 1},  # Self believes, teammate believes
            {'scenarios': [23, 24], 'action': 'Pass', 'weight': 1},  # Self knows, teammate unknown
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


def compute_mastery_score(records: List[dict], category: dict, extra_filter = None, lies_okay: bool = False) -> dict:
    """
    Compute mastery score for a single ToM category.

    Args:
        records: List of game records
        category: Category definition with components
        extra_filter: If specified, only include records with this Extra value.
                     Can be a single string ('0A', '0B', '1A', '1B') or tuple of strings.
                     See EXTRA_MAPPING.md for details.
        lies_okay: If True, lying to opponent answerer also counts as success

    Returns dict with:
    - score: weighted accuracy (0-1)
    - n: total weighted trials
    - k: weighted correct
    - by_component: breakdown by component
    """
    # Apply extra filter if specified
    if extra_filter is not None:
        if isinstance(extra_filter, tuple):
            records = [r for r in records if normalize_extra(r.get('extra')) in extra_filter]
        else:
            records = [r for r in records if normalize_extra(r.get('extra')) == extra_filter]

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
        n_correct = sum(1 for r in component_records if is_success(r, lies_okay))

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


def compute_all_mastery_scores(records: List[dict], extra_filter = None, lies_okay: bool = False) -> dict:
    """Compute all ToM mastery scores for a set of records.

    Args:
        extra_filter: Single string ('0A', '0B', '1A', '1B') or tuple of strings to filter by.
    """
    results = {}
    for key, category in TOM_MASTERY_CATEGORIES.items():
        results[key] = compute_mastery_score(records, category, extra_filter, lies_okay)
        results[key]['name'] = category['name']
        results[key]['description'] = category['description']
    return results


def compute_mastery_with_extra_breakout(records: List[dict], lies_okay: bool = False) -> dict:
    """Compute mastery scores with overall and all Extra category breakouts.

    Returns dict with keys:
        overall: All records
        extra0a: Minimal Events (0A)
        extra0b: Event Load (0B)
        extra1a: Minimal ECT (1A)
        extra1b: ECT Load (1B)
    """
    return {
        'overall': compute_all_mastery_scores(records, lies_okay=lies_okay),
        # All 4 individual categories
        'extra0a': compute_all_mastery_scores(records, extra_filter='0A', lies_okay=lies_okay),
        'extra0b': compute_all_mastery_scores(records, extra_filter='0B', lies_okay=lies_okay),
        'extra1a': compute_all_mastery_scores(records, extra_filter='1A', lies_okay=lies_okay),
        'extra1b': compute_all_mastery_scores(records, extra_filter='1B', lies_okay=lies_okay),
    }


def build_mastery_detail_lines(all_records: List[dict], model_records: dict, sorted_models: List[str], title: str, lies_okay: bool = False) -> List[str]:
    """Build detailed mastery breakdown lines for a set of records.

    Args:
        all_records: All records to analyze (for overall stats)
        model_records: Dict mapping model name to its records
        sorted_models: List of model names in display order
        title: Title for this section
        lies_okay: Whether to count lies to opponent as success

    Returns:
        List of formatted lines
    """
    lines = [title, "=" * 80]

    # Overall stats
    all_mastery = compute_all_mastery_scores(all_records, lies_okay=lies_okay)
    lines.append("\nOVERALL (All Models)")
    lines.append("-" * 40)
    for key, category in TOM_MASTERY_CATEGORIES.items():
        mastery = all_mastery[key]
        lines.append(f"\n{category['name']}: {mastery['score']*100:.1f}%")
        lines.append(f"  {category['description']}")
        for comp in mastery['by_component']:
            scenarios_str = ', '.join(str(s) for s in comp['scenarios'])
            lines.append(f"  - Scenarios [{scenarios_str}] → {comp['action']}: "
                  f"{comp['k']}/{comp['n']} = {comp['rate']*100:.1f}% (weight={comp['weight']})")

    # Per-model stats
    lines.append("\n" + "=" * 80)
    lines.append("PER-MODEL BREAKDOWN")
    lines.append("=" * 80)

    for model in sorted_models:
        if model not in model_records or not model_records[model]:
            continue
        model_mastery = compute_all_mastery_scores(model_records[model], lies_okay=lies_okay)
        n_records = len(model_records[model])
        lines.append(f"\n=== {model} ({n_records} records) ===")
        for key, category in TOM_MASTERY_CATEGORIES.items():
            mastery = model_mastery[key]
            lines.append(f"\n{category['name']}: {mastery['score']*100:.1f}%")
            for comp in mastery['by_component']:
                scenarios_str = ', '.join(str(s) for s in comp['scenarios'])
                lines.append(f"  - [{scenarios_str}] → {comp['action']}: "
                      f"{comp['k']}/{comp['n']} = {comp['rate']*100:.1f}%")

    return lines


def format_rate_ci(rate: float, ci: Tuple[float, float]) -> str:
    """Format rate with CI as string like '85.3% (±2.5)'."""
    half_width = (ci[1] - ci[0]) / 2 * 100
    return f"{rate*100:5.1f}% (±{half_width:4.1f})"


def find_game_data_files(base_dir: str) -> List[str]:
    """Find all game_data.json files recursively."""
    pattern = os.path.join(base_dir, '**', '*_game_data.json')
    return glob.glob(pattern, recursive=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze ToM test results')
    parser.add_argument('--lies_okay', action='store_true',
                        help='Count lying to opponent answerer as success')
    args = parser.parse_args()

    # Find the tom_llm_logs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, 'tom_llm_logs')

    if not os.path.exists(logs_dir):
        print(f"Error: {logs_dir} not found")
        return

    if args.lies_okay:
        print("*** lies_okay mode: counting lies to opponent answerer as success ***")

    # Find all game data files
    files = find_game_data_files(logs_dir)

    # Aggregate records by model
    model_records: Dict[str, List[dict]] = defaultdict(list)

    for filepath in files:
        records = load_game_data(filepath)
        a_records = filter_player_a(records)

        if not a_records:
            continue

        model_name = extract_model_name(filepath)
        model_records[model_name].extend(a_records)

    if not model_records:
        print("No valid game data files found")
        return

    # Compute stats for each model
    model_stats = {}
    for model_name, records in model_records.items():
        model_stats[model_name] = compute_stats(records, lies_okay=args.lies_okay)

    # Sort by overall rate (descending)
    sorted_models = sorted(model_stats.keys(),
                          key=lambda m: model_stats[m]['overall']['rate'],
                          reverse=True)

    # Aggregate all records for cross-model analysis
    all_records = []
    for records in model_records.values():
        all_records.extend(records)

    # Group records by (model, free_response)
    fr_model_records = {}
    for fr_mode in [True, False]:
        fr_model_records[fr_mode] = {}
        for model, recs in model_records.items():
            filtered = [r for r in recs if r.get('free_response') == fr_mode]
            if filtered:
                fr_model_records[fr_mode][model] = filtered

    # Build detailed output for file
    output_lines = []

    # Main table
    output_lines.append("=" * 120)
    output_lines.append(f"{'Model':<30} | {'N':>5} | {'Overall':>12} | {'Min Events':>12} | {'Event Load':>12} | {'Min ECT':>12} | {'ECT Load':>12}")
    output_lines.append("-" * 120)

    for model in sorted_models:
        stats = model_stats[model]
        overall_str = format_rate_ci(stats['overall']['rate'], stats['overall']['ci'])
        extra0a_str = format_rate_ci(stats['extra0a']['rate'], stats['extra0a']['ci'])
        extra0b_str = format_rate_ci(stats['extra0b']['rate'], stats['extra0b']['ci'])
        extra1a_str = format_rate_ci(stats['extra1a']['rate'], stats['extra1a']['ci'])
        extra1b_str = format_rate_ci(stats['extra1b']['rate'], stats['extra1b']['ci'])
        output_lines.append(f"{model:<30} | {stats['n']:>5} | {overall_str:>12} | {extra0a_str:>12} | {extra0b_str:>12} | {extra1a_str:>12} | {extra1b_str:>12}")

    output_lines.append("=" * 120)

    # Free response breakdown
    output_lines.append("\nBREAKDOWN BY FREE_RESPONSE MODE")
    output_lines.append("=" * 120)

    for fr_mode in [True, False]:
        if not fr_model_records[fr_mode]:
            continue

        mode_label = "Free Response" if fr_mode else "Multiple Choice"
        output_lines.append(f"\n{mode_label} (free_response={fr_mode})")
        output_lines.append(f"{'Model':<30} | {'N':>5} | {'Overall':>12} | {'Min Events':>12} | {'Event Load':>12} | {'Min ECT':>12} | {'ECT Load':>12}")
        output_lines.append("-" * 120)

        for model in sorted_models:
            if model not in fr_model_records[fr_mode]:
                continue
            stats = compute_stats(fr_model_records[fr_mode][model], lies_okay=args.lies_okay)
            overall_str = format_rate_ci(stats['overall']['rate'], stats['overall']['ci'])
            extra0a_str = format_rate_ci(stats['extra0a']['rate'], stats['extra0a']['ci'])
            extra0b_str = format_rate_ci(stats['extra0b']['rate'], stats['extra0b']['ci'])
            extra1a_str = format_rate_ci(stats['extra1a']['rate'], stats['extra1a']['ci'])
            extra1b_str = format_rate_ci(stats['extra1b']['rate'], stats['extra1b']['ci'])
            output_lines.append(f"{model:<30} | {stats['n']:>5} | {overall_str:>12} | {extra0a_str:>12} | {extra0b_str:>12} | {extra1a_str:>12} | {extra1b_str:>12}")

    output_lines.append("=" * 120)

    # Compute ToM Mastery scores for each model with Extra breakouts
    cat_keys = list(TOM_MASTERY_CATEGORIES.keys())
    cat_short_names = []
    for k in cat_keys:
        name = TOM_MASTERY_CATEGORIES[k]['name']
        short_name = name[:15] if len(name) > 15 else name
        cat_short_names.append(short_name)

    header = f"{'Model':<35}"
    for short_name in cat_short_names:
        header += f" | {short_name:>15}"

    # Compute and store mastery scores for each model
    model_mastery = {}
    mastery_table_lines = ["ToM MASTERY SCORES BY CATEGORY (Overall)", "=" * 140, header, "-" * 140]

    for model in sorted_models:
        records = model_records[model]
        mastery = compute_mastery_with_extra_breakout(records, lies_okay=args.lies_okay)
        model_mastery[model] = mastery

        row = f"{model:<35}"
        for key in cat_keys:
            score = mastery['overall'][key]['score'] * 100
            row += f" | {score:>14.1f}%"
        mastery_table_lines.append(row)

    mastery_table_lines.append("-" * 140)

    # Compute aggregate mastery scores across all models
    all_mastery = compute_mastery_with_extra_breakout(all_records, lies_okay=args.lies_okay)
    row = f"{'ALL MODELS':<35}"
    for key in cat_keys:
        score = all_mastery['overall'][key]['score'] * 100
        row += f" | {score:>14.1f}%"
    mastery_table_lines.append(row)
    mastery_table_lines.append("=" * 140)

    # Build breakdown for each Extra category (no screen output)
    extra_table_data = {}
    for extra_key, extra_info in EXTRA_CATEGORIES.items():
        mastery_key = f'extra{extra_key.lower()}'
        title = f"ToM MASTERY SCORES BY CATEGORY ({extra_info['name']} - {extra_key})"

        table_lines = [header, "-" * 140]
        for model in sorted_models:
            mastery = model_mastery[model]
            row = f"{model:<35}"
            for key in cat_keys:
                score = mastery[mastery_key][key]['score'] * 100
                row += f" | {score:>14.1f}%"
            table_lines.append(row)

        table_lines.append("-" * 140)
        row = f"{'ALL MODELS':<35}"
        for key in cat_keys:
            score = all_mastery[mastery_key][key]['score'] * 100
            row += f" | {score:>14.1f}%"
        table_lines.append(row)
        table_lines.append("=" * 140)

        extra_table_data[extra_key] = {'title': title, 'lines': table_lines}

    # Save mastery tables to files
    mastery_overall_path = os.path.join(logs_dir, 'mastery_overall.txt')
    with open(mastery_overall_path, 'w') as f:
        f.write("\n".join(mastery_table_lines))

    # Save each extra category table
    extra_file_names = {
        '0A': 'mastery_minimal_events.txt',
        '0B': 'mastery_event_load.txt',
        '1A': 'mastery_minimal_ect.txt',
        '1B': 'mastery_ect_load.txt',
    }
    for extra_key, filename in extra_file_names.items():
        filepath = os.path.join(logs_dir, filename)
        with open(filepath, 'w') as f:
            f.write(extra_table_data[extra_key]['title'] + "\n")
            f.write("\n".join(extra_table_data[extra_key]['lines']))

    # Save combined CSV for easy analysis
    mastery_csv_path = os.path.join(logs_dir, 'mastery_scores.csv')
    with open(mastery_csv_path, 'w') as f:
        # Header
        csv_header = "Model,Extra"
        for key in cat_keys:
            csv_header += f",{TOM_MASTERY_CATEGORIES[key]['name']}"
        f.write(csv_header + "\n")

        # Define extra categories for CSV
        csv_extra_mapping = [
            ('Overall', 'overall'),
            ('Minimal Events', 'extra0a'),
            ('Event Load', 'extra0b'),
            ('Minimal ECT', 'extra1a'),
            ('ECT Load', 'extra1b'),
        ]

        # Data rows
        for model in sorted_models:
            mastery = model_mastery[model]
            for extra_label, extra_key in csv_extra_mapping:
                row = f"{model},{extra_label}"
                for key in cat_keys:
                    score = mastery[extra_key][key]['score'] * 100
                    row += f",{score:.1f}"
                f.write(row + "\n")

        # All models aggregate
        for extra_label, extra_key in csv_extra_mapping:
            row = f"ALL MODELS,{extra_label}"
            for key in cat_keys:
                score = all_mastery[extra_key][key]['score'] * 100
                row += f",{score:.1f}"
            f.write(row + "\n")

    # Build detailed breakdown files for each free_response mode
    # 1. All data combined
    detail_lines = build_mastery_detail_lines(
        all_records, model_records, sorted_models,
        "Detailed Mastery Category Breakdown (All Data)", lies_okay=args.lies_okay
    )
    mastery_detail_path = os.path.join(logs_dir, 'mastery_detail.txt')
    with open(mastery_detail_path, 'w') as f:
        f.write("\n".join(detail_lines))

    # 2. Free Response only (fr_true)
    fr_true_records = [r for r in all_records if r.get('free_response') == True]
    if fr_true_records:
        fr_true_model_records = {m: [r for r in recs if r.get('free_response') == True]
                                  for m, recs in model_records.items()}
        fr_true_model_records = {m: recs for m, recs in fr_true_model_records.items() if recs}
        detail_lines_fr_true = build_mastery_detail_lines(
            fr_true_records, fr_true_model_records, sorted_models,
            "Detailed Mastery Category Breakdown (Free Response)", lies_okay=args.lies_okay
        )
        mastery_detail_fr_true_path = os.path.join(logs_dir, 'mastery_detail_fr_true.txt')
        with open(mastery_detail_fr_true_path, 'w') as f:
            f.write("\n".join(detail_lines_fr_true))

    # 3. Multiple Choice only (fr_false)
    fr_false_records = [r for r in all_records if r.get('free_response') == False]
    if fr_false_records:
        fr_false_model_records = {m: [r for r in recs if r.get('free_response') == False]
                                   for m, recs in model_records.items()}
        fr_false_model_records = {m: recs for m, recs in fr_false_model_records.items() if recs}
        detail_lines_fr_false = build_mastery_detail_lines(
            fr_false_records, fr_false_model_records, sorted_models,
            "Detailed Mastery Category Breakdown (Multiple Choice)", lies_okay=args.lies_okay
        )
        mastery_detail_fr_false_path = os.path.join(logs_dir, 'mastery_detail_fr_false.txt')
        with open(mastery_detail_fr_false_path, 'w') as f:
            f.write("\n".join(detail_lines_fr_false))

    # Save main summary table to file
    summary_path = os.path.join(logs_dir, 'summary_table.txt')
    with open(summary_path, 'w') as f:
        f.write("\n".join(output_lines))

    # Track saved files for summary
    saved_files = [
        summary_path,
        mastery_overall_path,
        mastery_detail_path,
        mastery_csv_path,
    ]
    if fr_true_records:
        saved_files.append(mastery_detail_fr_true_path)
    if fr_false_records:
        saved_files.append(mastery_detail_fr_false_path)
    saved_files.extend([os.path.join(logs_dir, fn) for fn in extra_file_names.values()])

    # Generate charts for each free_response mode
    # Define modes: None = all data, True = free response only, False = multiple choice only
    chart_modes = [(None, '', 'All Data')]
    for fr_mode in [True, False]:
        if fr_model_records.get(fr_mode):
            suffix = '_fr_true' if fr_mode else '_fr_false'
            label = 'Free Response' if fr_mode else 'Multiple Choice'
            chart_modes.append((fr_mode, suffix, label))

    for fr_mode, file_suffix, mode_label in chart_modes:
        # Filter records for this mode
        if fr_mode is None:
            chart_records = all_records
            chart_model_stats = model_stats
        else:
            chart_records = [r for r in all_records if r.get('free_response') == fr_mode]
            if not chart_records:
                continue
            # Recompute model stats for filtered records
            chart_model_stats = {}
            for model in sorted_models:
                model_recs = [r for r in chart_records if r in model_records.get(model, [])]
                if model_recs:
                    chart_model_stats[model] = compute_stats(model_recs, lies_okay=args.lies_okay)

        # Compute per-scenario stats
        scenario_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in chart_records:
            extra_str = normalize_extra(r.get('extra'))
            key = (r.get('scenario_id'), extra_str)
            scenario_stats[key]['total'] += 1
            if is_success(r, lies_okay=args.lies_okay):
                scenario_stats[key]['correct'] += 1

        # Sort by accuracy (ascending = hardest first)
        sorted_scenarios = sorted(
            scenario_stats.items(),
            key=lambda x: x[1]['correct'] / x[1]['total'] if x[1]['total'] > 0 else 0
        )

        # Save per-scenario stats to file (not screen)
        if fr_mode is None:
            scenario_lines = [f"{'Scenario':<10} | {'Extra':>12} | {'N':>6} | {'Accuracy':>10}", "-" * 50]
            for (scenario_id, extra), stats in sorted_scenarios:
                acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                extra_name = EXTRA_CATEGORIES.get(extra, {}).get('short', extra)
                scenario_lines.append(f"{scenario_id:<10} | {extra_name:>12} | {stats['total']:>6} | {acc*100:>9.1f}%")
            scenario_lines.append("-" * 50)
            scenario_stats_path = os.path.join(logs_dir, 'per_scenario_stats.txt')
            with open(scenario_stats_path, 'w') as f:
                f.write("\n".join(scenario_lines))
            saved_files.append(scenario_stats_path)

        # Generate scenario difficulty chart
        scenario_labels = []
        scenario_accs = []
        scenario_colors = []
        for (scenario_id, extra), stats in sorted_scenarios:
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            scenario_labels.append(f"{scenario_id}-{extra}")
            scenario_accs.append(acc * 100)
            scenario_colors.append(EXTRA_CATEGORIES.get(extra, {}).get('color', '#999999'))

        if scenario_accs:
            fig2, ax2 = plt.subplots(figsize=(16, 8))
            x2 = np.arange(len(scenario_labels))
            ax2.bar(x2, scenario_accs, color=scenario_colors, edgecolor='white', linewidth=0.5)
            ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
            ax2.axhline(y=np.mean(scenario_accs), color='green', linestyle='-', alpha=0.7)
            ax2.set_xlabel('Scenario ID - Extra Category', fontsize=12)
            ax2.set_ylabel('Accuracy (%)', fontsize=12)
            ax2.set_title(f'Per-Scenario Accuracy ({mode_label})', fontsize=14, fontweight='bold')
            ax2.set_xticks(x2)
            ax2.set_xticklabels(scenario_labels, rotation=90, fontsize=8)
            ax2.set_ylim(0, 105)
            ax2.grid(axis='y', alpha=0.3)
            ax2.legend(handles=[
                Patch(facecolor=EXTRA_CATEGORIES['0A']['color'], label=f"0A: {EXTRA_CATEGORIES['0A']['name']}"),
                Patch(facecolor=EXTRA_CATEGORIES['0B']['color'], label=f"0B: {EXTRA_CATEGORIES['0B']['name']}"),
                Patch(facecolor=EXTRA_CATEGORIES['1A']['color'], label=f"1A: {EXTRA_CATEGORIES['1A']['name']}"),
                Patch(facecolor=EXTRA_CATEGORIES['1B']['color'], label=f"1B: {EXTRA_CATEGORIES['1B']['name']}"),
            ], loc='lower right')
            plt.tight_layout()
            scenario_chart_path = os.path.join(logs_dir, f'scenario_difficulty{file_suffix}.png')
            plt.savefig(scenario_chart_path, dpi=150, bbox_inches='tight')
            saved_files.append(scenario_chart_path)
            plt.close(fig2)

        # Generate Extra category comparison chart
        scenario_ids = sorted(set(k[0] for k in scenario_stats.keys()), key=lambda x: int(x))
        if scenario_ids:
            extra_accs = {cat: [] for cat in EXTRA_CATEGORIES.keys()}
            for sid in scenario_ids:
                for cat in EXTRA_CATEGORIES.keys():
                    stats = scenario_stats.get((sid, cat), {'correct': 0, 'total': 0})
                    acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
                    extra_accs[cat].append(acc)

            fig3, ax3 = plt.subplots(figsize=(18, 8))
            x3 = np.arange(len(scenario_ids))
            width = 0.2
            offsets = {'0A': -1.5, '0B': -0.5, '1A': 0.5, '1B': 1.5}

            for cat, offset in offsets.items():
                ax3.bar(x3 + offset * width, extra_accs[cat], width,
                        label=f"{cat}: {EXTRA_CATEGORIES[cat]['name']}",
                        color=EXTRA_CATEGORIES[cat]['color'])

            ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
            ax3.set_xlabel('Scenario ID', fontsize=12)
            ax3.set_ylabel('Accuracy (%)', fontsize=12)
            ax3.set_title(f'Accuracy by Scenario and Extra Category ({mode_label})', fontsize=14, fontweight='bold')
            ax3.set_xticks(x3)
            ax3.set_xticklabels(scenario_ids, fontsize=8)
            ax3.set_ylim(0, 105)
            ax3.legend(loc='lower right')
            ax3.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            paired_chart_path = os.path.join(logs_dir, f'extra_comparison{file_suffix}.png')
            plt.savefig(paired_chart_path, dpi=150, bbox_inches='tight')
            saved_files.append(paired_chart_path)
            plt.close(fig3)

        # Generate model performance bar chart
        chart_models = [m for m in sorted_models if m in chart_model_stats]
        if chart_models:
            fig, ax = plt.subplots(figsize=(16, 8))
            x = np.arange(len(chart_models))
            width = 0.15

            overall_rates = [chart_model_stats[m]['overall']['rate'] * 100 for m in chart_models]
            extra0a_rates = [chart_model_stats[m]['extra0a']['rate'] * 100 for m in chart_models]
            extra0b_rates = [chart_model_stats[m]['extra0b']['rate'] * 100 for m in chart_models]
            extra1a_rates = [chart_model_stats[m]['extra1a']['rate'] * 100 for m in chart_models]
            extra1b_rates = [chart_model_stats[m]['extra1b']['rate'] * 100 for m in chart_models]

            overall_errs = [(chart_model_stats[m]['overall']['ci'][1] - chart_model_stats[m]['overall']['ci'][0]) / 2 * 100 for m in chart_models]
            extra0a_errs = [(chart_model_stats[m]['extra0a']['ci'][1] - chart_model_stats[m]['extra0a']['ci'][0]) / 2 * 100 for m in chart_models]
            extra0b_errs = [(chart_model_stats[m]['extra0b']['ci'][1] - chart_model_stats[m]['extra0b']['ci'][0]) / 2 * 100 for m in chart_models]
            extra1a_errs = [(chart_model_stats[m]['extra1a']['ci'][1] - chart_model_stats[m]['extra1a']['ci'][0]) / 2 * 100 for m in chart_models]
            extra1b_errs = [(chart_model_stats[m]['extra1b']['ci'][1] - chart_model_stats[m]['extra1b']['ci'][0]) / 2 * 100 for m in chart_models]

            ax.bar(x - 2*width, overall_rates, width, label='Overall', yerr=overall_errs, capsize=2, color='#2ecc71')
            ax.bar(x - width, extra0a_rates, width, label=f"0A: {EXTRA_CATEGORIES['0A']['name']}", yerr=extra0a_errs, capsize=2, color=EXTRA_CATEGORIES['0A']['color'])
            ax.bar(x, extra0b_rates, width, label=f"0B: {EXTRA_CATEGORIES['0B']['name']}", yerr=extra0b_errs, capsize=2, color=EXTRA_CATEGORIES['0B']['color'])
            ax.bar(x + width, extra1a_rates, width, label=f"1A: {EXTRA_CATEGORIES['1A']['name']}", yerr=extra1a_errs, capsize=2, color=EXTRA_CATEGORIES['1A']['color'])
            ax.bar(x + 2*width, extra1b_rates, width, label=f"1B: {EXTRA_CATEGORIES['1B']['name']}", yerr=extra1b_errs, capsize=2, color=EXTRA_CATEGORIES['1B']['color'])

            ax.set_ylabel('Optimal Action Rate (%)', fontsize=12)
            ax.set_title(f'ToM Test Performance by Model ({mode_label})', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(chart_models, rotation=45, ha='right', fontsize=9)
            ax.legend(loc='lower right', fontsize=8)
            ax.set_ylim(0, 105)
            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            chart_path = os.path.join(logs_dir, f'performance_comparison{file_suffix}.png')
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            saved_files.append(chart_path)
            plt.close(fig)

    # Print high-level summary
    print("\n" + "=" * 60)
    print("ToM ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Models analyzed: {len(sorted_models)}")
    print(f"Total records: {len(all_records)}")
    print()

    # Show top-level results per model
    print(f"{'Model':<30} | {'Records':>7} | {'Accuracy':>10}")
    print("-" * 55)
    for model in sorted_models:
        stats = model_stats[model]
        print(f"{model:<30} | {stats['n']:>7} | {stats['overall']['rate']*100:>9.1f}%")
    print("-" * 55)

    # Show files saved
    print(f"\nSaved {len(saved_files)} files to {logs_dir}/")
    print("  Tables: summary_table.txt, per_scenario_stats.txt")
    print("  Mastery: mastery_*.txt, mastery_scores.csv")
    print("  Charts: performance_comparison*.png, scenario_difficulty*.png, extra_comparison*.png")

    plt.show()


if __name__ == '__main__':
    main()
