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


def format_model_name(model: str, free_response: bool) -> str:
    """Format model name with COT suffix if free_response is True."""
    return f"{model} (with COT)" if free_response else model


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


def is_component_success(record: dict, component: dict, lies_okay: bool = False) -> bool:
    """Check if a record counts as a success for a specific component.

    Args:
        record: A turn record dict
        component: Component definition dict (may have 'success_is_lie' flag)
        lies_okay: If True, lying to opponent answerer also counts as success (for non-lie components)

    Returns:
        True if the record meets the component's success criteria
    """
    if component.get('success_is_lie'):
        # For strategic lies category: success = lying to opponent answerer
        return record.get('lied_to_opponent_answerer') == "TRUE"
    else:
        # Default behavior: use is_success()
        return is_success(record, lies_okay)


def has_valid_action(action: str) -> bool:
    """Check if action string contains a valid action keyword.

    Used to filter out records where the model hit token limit before
    outputting its action (Pass/Ask/Tell).
    """
    if not action:
        return False
    # Check for Pass (case insensitive, word boundary)
    if re.search(r'\bpass\b', action, re.IGNORECASE):
        return True
    # Check for Ask() or Tell()
    if re.search(r'\b(ask|tell)\s*\(', action, re.IGNORECASE):
        return True
    return False


def filter_valid_records(records: List[dict]) -> Tuple[List[dict], int]:
    """Filter out records where model hit token limit before giving action.

    Returns: (filtered_records, excluded_count)
    """
    filtered = [r for r in records if has_valid_action(r.get('action', ''))]
    return filtered, len(records) - len(filtered)


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

    # 6. Strategic Lies: Knowing when lying is effective vs. unnecessary
    # 27-29: Opponent believes truth - lying could deceive them (success = lie)
    # 30-32: Opponent believes false - no intervention helps (success = pass)
    'strategic_lies': {
        'name': 'Strategic Lies',
        'description': 'Knowing when lying is strategically effective vs. unnecessary',
        'components': [
            {'scenarios': [27, 28, 29], 'action': 'Tell (Lie)', 'weight': 1, 'success_is_lie': True},
            {'scenarios': [30, 31, 32], 'action': 'Pass', 'weight': 1, 'success_is_lie': False},
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

        # For success_is_lie components, exclude records missing lie data
        if component.get('success_is_lie'):
            component_records = [r for r in component_records
                                if 'lied_to_opponent_answerer' in r]

        n_total = len(component_records)
        n_correct = sum(1 for r in component_records if is_component_success(r, component, lies_okay))

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


def build_mastery_detail_lines(all_records: List[dict], model_records: dict, sorted_models: List[str], title: str, extra_filter: str = None) -> List[str]:
    """Build detailed mastery breakdown lines for a set of records.

    Args:
        all_records: All records to analyze (for overall stats)
        model_records: Dict mapping model name to its records
        sorted_models: List of model names in display order
        title: Title for this section
        extra_filter: If specified, only include records with this Extra value ('0A', '0B', '1A', '1B')

    Returns:
        List of formatted lines
    """
    # Scenarios that should show both without and with lies_okay
    # Note: 30-32 removed - only Pass is correct (opponent already believes false)
    # Note: 27-29 not needed - strategic_lies category already counts lies as success
    LIES_SCENARIOS = [[37, 39]]
    # Scenarios to show gratuitous lie rate (info only, not scored)
    GRATUITOUS_LIE_SCENARIOS = [30, 31, 32]

    def scenarios_need_lies_line(scenarios):
        return scenarios in LIES_SCENARIOS

    def compute_gratuitous_lie_rate(records, scenarios, extra_filter=None):
        """Compute lie rate for scenarios where lying is gratuitous (opponent already wrong)."""
        filtered = [r for r in records if r.get('scenario_id') in [str(s) for s in scenarios]]
        if extra_filter:
            if isinstance(extra_filter, tuple):
                filtered = [r for r in filtered if normalize_extra(r.get('extra')) in extra_filter]
            else:
                filtered = [r for r in filtered if normalize_extra(r.get('extra')) == extra_filter]
        # Only count records with lie data
        filtered = [r for r in filtered if 'lied_to_opponent_answerer' in r]
        if not filtered:
            return 0, 0, 0.0
        n_lies = sum(1 for r in filtered if r.get('lied_to_opponent_answerer') == 'TRUE')
        return n_lies, len(filtered), n_lies / len(filtered) if filtered else 0.0

    lines = [title, "=" * 80]

    # Overall stats
    all_mastery = compute_all_mastery_scores(all_records, extra_filter=extra_filter, lies_okay=False)
    all_mastery_lies = compute_all_mastery_scores(all_records, extra_filter=extra_filter, lies_okay=True)
    lines.append("\nOVERALL (All Models)")
    lines.append("-" * 40)
    for key, category in TOM_MASTERY_CATEGORIES.items():
        mastery = all_mastery[key]
        lines.append(f"\n{category['name']}: {mastery['score']*100:.1f}%")
        lines.append(f"  {category['description']}")
        for i, comp in enumerate(mastery['by_component']):
            scenarios_str = ', '.join(str(s) for s in comp['scenarios'])
            lines.append(f"  - Scenarios [{scenarios_str}] → {comp['action']}: "
                  f"{comp['k']}/{comp['n']} = {comp['rate']*100:.1f}% (weight={comp['weight']})")
            # Add lies_okay line for specific scenarios
            if scenarios_need_lies_line(comp['scenarios']):
                comp_lies = all_mastery_lies[key]['by_component'][i]
                lines.append(f"  - Scenarios [{scenarios_str}] → {comp['action']} (lies ok): "
                      f"{comp_lies['k']}/{comp_lies['n']} = {comp_lies['rate']*100:.1f}% (weight={comp_lies['weight']})")
            # Add gratuitous lie info for 30-32 (only in strategic_lies category)
            if key == 'strategic_lies' and comp['scenarios'] == GRATUITOUS_LIE_SCENARIOS:
                n_lies, n_total, rate = compute_gratuitous_lie_rate(all_records, GRATUITOUS_LIE_SCENARIOS, extra_filter)
                if n_total > 0:
                    lines.append(f"  - Scenarios [{scenarios_str}] → Gratuitous Lies: "
                          f"{n_lies}/{n_total} = {rate*100:.1f}%  [info only]")

    # Per-model stats
    lines.append("\n" + "=" * 80)
    lines.append("PER-MODEL BREAKDOWN")
    lines.append("=" * 80)

    for model in sorted_models:
        if model not in model_records or not model_records[model]:
            continue
        model_mastery = compute_all_mastery_scores(model_records[model], extra_filter=extra_filter, lies_okay=False)
        model_mastery_lies = compute_all_mastery_scores(model_records[model], extra_filter=extra_filter, lies_okay=True)
        # Count records with extra_filter applied
        if extra_filter:
            n_records = len([r for r in model_records[model] if normalize_extra(r.get('extra')) == extra_filter])
        else:
            n_records = len(model_records[model])
        lines.append(f"\n=== {model} ({n_records} records) ===")
        for key, category in TOM_MASTERY_CATEGORIES.items():
            mastery = model_mastery[key]
            lines.append(f"\n{category['name']}: {mastery['score']*100:.1f}%")
            for i, comp in enumerate(mastery['by_component']):
                scenarios_str = ', '.join(str(s) for s in comp['scenarios'])
                lines.append(f"  - [{scenarios_str}] → {comp['action']}: "
                      f"{comp['k']}/{comp['n']} = {comp['rate']*100:.1f}%")
                # Add lies_okay line for specific scenarios
                if scenarios_need_lies_line(comp['scenarios']):
                    comp_lies = model_mastery_lies[key]['by_component'][i]
                    lines.append(f"  - [{scenarios_str}] → {comp['action']} (lies ok): "
                          f"{comp_lies['k']}/{comp_lies['n']} = {comp_lies['rate']*100:.1f}%")
                # Add gratuitous lie info for 30-32 (only in strategic_lies category)
                if key == 'strategic_lies' and comp['scenarios'] == GRATUITOUS_LIE_SCENARIOS:
                    n_lies, n_total, rate = compute_gratuitous_lie_rate(model_records[model], GRATUITOUS_LIE_SCENARIOS, extra_filter)
                    if n_total > 0:
                        lines.append(f"  - [{scenarios_str}] → Gratuitous Lies: "
                              f"{n_lies}/{n_total} = {rate*100:.1f}%  [info only]")

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
    # Find the tom_llm_logs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, 'tom_llm_logs')

    if not os.path.exists(logs_dir):
        print(f"Error: {logs_dir} not found")
        return

    # Find all game data files
    files = find_game_data_files(logs_dir)

    # Aggregate records by model
    model_records: Dict[str, List[dict]] = defaultdict(list)

    total_excluded = 0
    for filepath in files:
        records = load_game_data(filepath)
        a_records = filter_player_a(records)
        # Filter out records where model hit token limit before giving action
        a_records, excluded = filter_valid_records(a_records)
        total_excluded += excluded

        if not a_records:
            continue

        model_name = extract_model_name(filepath)
        model_records[model_name].extend(a_records)

    if not model_records:
        print("No valid game data files found")
        return

    if total_excluded > 0:
        print(f"Excluded {total_excluded} records with no action (token limit)")

    # Compute stats for each model
    model_stats = {}
    for model_name, records in model_records.items():
        model_stats[model_name] = compute_stats(records, lies_okay=False)

    # Sort alphabetically
    sorted_models = sorted(model_stats.keys())

    # Aggregate all records for cross-model analysis
    all_records = []
    for records in model_records.values():
        all_records.extend(records)

    # Build combined model records with "(with COT)" suffix for free_response=True
    # This creates adjacent ordering: model, model (with COT), next_model, next_model (with COT)
    combined_model_records = {}
    combined_model_order = []  # Maintains adjacent ordering

    for base_model in sorted_models:
        for fr_mode in [False, True]:  # Non-COT first, then COT
            filtered = [r for r in model_records[base_model] if r.get('free_response') == fr_mode]
            if filtered:
                display_name = format_model_name(base_model, fr_mode)
                combined_model_records[display_name] = filtered
                combined_model_order.append(display_name)

    # Build detailed output for file (single combined table)
    output_lines = []
    output_lines.append(f"\n{'Model':<40} | {'N':>5} | {'Overall':>12} | {'Min Events':>12} | {'Event Load':>12} | {'Min ECT':>12} | {'ECT Load':>12}")
    output_lines.append("-" * 130)

    for display_name in combined_model_order:
        recs = combined_model_records[display_name]
        # Without lies_okay
        stats = compute_stats(recs, lies_okay=False)
        overall_str = format_rate_ci(stats['overall']['rate'], stats['overall']['ci'])
        extra0a_str = format_rate_ci(stats['extra0a']['rate'], stats['extra0a']['ci'])
        extra0b_str = format_rate_ci(stats['extra0b']['rate'], stats['extra0b']['ci'])
        extra1a_str = format_rate_ci(stats['extra1a']['rate'], stats['extra1a']['ci'])
        extra1b_str = format_rate_ci(stats['extra1b']['rate'], stats['extra1b']['ci'])
        output_lines.append(f"{display_name:<40} | {stats['n']:>5} | {overall_str:>12} | {extra0a_str:>12} | {extra0b_str:>12} | {extra1a_str:>12} | {extra1b_str:>12}")
        # With lies_okay
        stats_lies = compute_stats(recs, lies_okay=True)
        overall_str = format_rate_ci(stats_lies['overall']['rate'], stats_lies['overall']['ci'])
        extra0a_str = format_rate_ci(stats_lies['extra0a']['rate'], stats_lies['extra0a']['ci'])
        extra0b_str = format_rate_ci(stats_lies['extra0b']['rate'], stats_lies['extra0b']['ci'])
        extra1a_str = format_rate_ci(stats_lies['extra1a']['rate'], stats_lies['extra1a']['ci'])
        extra1b_str = format_rate_ci(stats_lies['extra1b']['rate'], stats_lies['extra1b']['ci'])
        output_lines.append(f"{display_name + ' (lies ok)':<40} | {stats_lies['n']:>5} | {overall_str:>12} | {extra0a_str:>12} | {extra0b_str:>12} | {extra1a_str:>12} | {extra1b_str:>12}")

    output_lines.append("=" * 130)

    # Compute ToM Mastery scores for each model with Extra breakouts
    cat_keys = list(TOM_MASTERY_CATEGORIES.keys())
    cat_short_names = []
    for k in cat_keys:
        name = TOM_MASTERY_CATEGORIES[k]['name']
        short_name = name[:15] if len(name) > 15 else name
        cat_short_names.append(short_name)

    # Header now includes N column and wider category columns for "XX.X% (N)"
    # Add "TM vs Opp (lies)" column after teammate_opponent
    header = f"{'Model':<45} | {'N':>6}"
    for short_name in cat_short_names:
        header += f" | {short_name:>18}"
    header += f" | {'TM vs Opp (lies)':>18}"

    # Compute and store mastery scores for each combined model (with COT suffix)
    combined_model_mastery = {}
    combined_model_mastery_lies = {}  # For lies_okay=True version
    mastery_table_lines = ["ToM MASTERY SCORES BY CATEGORY (Overall)", "=" * 190, header, "-" * 190]

    for display_name in combined_model_order:
        records = combined_model_records[display_name]
        mastery = compute_mastery_with_extra_breakout(records, lies_okay=False)
        mastery_lies = compute_mastery_with_extra_breakout(records, lies_okay=True)
        combined_model_mastery[display_name] = mastery
        combined_model_mastery_lies[display_name] = mastery_lies

        n_records = len(records)
        row = f"{display_name:<45} | {n_records:>6}"
        for key in cat_keys:
            score = mastery['overall'][key]['score'] * 100
            n_cat = int(mastery['overall'][key]['n'])
            row += f" | {score:>6.1f}% ({n_cat:>4})"
        # Add lies_okay version of teammate_opponent
        score_lies = mastery_lies['overall']['teammate_opponent']['score'] * 100
        n_lies = int(mastery_lies['overall']['teammate_opponent']['n'])
        row += f" | {score_lies:>6.1f}% ({n_lies:>4})"
        mastery_table_lines.append(row)

    mastery_table_lines.append("-" * 190)

    # Compute aggregate mastery scores across all models
    all_mastery = compute_mastery_with_extra_breakout(all_records, lies_okay=False)
    all_mastery_lies = compute_mastery_with_extra_breakout(all_records, lies_okay=True)
    n_all = len(all_records)
    row = f"{'ALL MODELS':<45} | {n_all:>6}"
    for key in cat_keys:
        score = all_mastery['overall'][key]['score'] * 100
        n_cat = int(all_mastery['overall'][key]['n'])
        row += f" | {score:>6.1f}% ({n_cat:>4})"
    # Add lies_okay version of teammate_opponent for ALL MODELS
    score_lies = all_mastery_lies['overall']['teammate_opponent']['score'] * 100
    n_lies = int(all_mastery_lies['overall']['teammate_opponent']['n'])
    row += f" | {score_lies:>6.1f}% ({n_lies:>4})"
    mastery_table_lines.append(row)
    mastery_table_lines.append("=" * 190)

    # Build breakdown for each Extra category (no screen output)
    extra_table_data = {}
    for extra_key, extra_info in EXTRA_CATEGORIES.items():
        mastery_key = f'extra{extra_key.lower()}'
        title = f"ToM MASTERY SCORES BY CATEGORY ({extra_info['name']} - {extra_key})"

        table_lines = [header, "-" * 170]
        for display_name in combined_model_order:
            mastery = combined_model_mastery[display_name]
            # Count records for this extra type
            n_extra = len([r for r in combined_model_records[display_name]
                          if normalize_extra(r.get('extra')) == extra_key])
            row = f"{display_name:<45} | {n_extra:>6}"
            for key in cat_keys:
                score = mastery[mastery_key][key]['score'] * 100
                n_cat = int(mastery[mastery_key][key]['n'])
                row += f" | {score:>6.1f}% ({n_cat:>4})"
            table_lines.append(row)

        table_lines.append("-" * 170)
        n_all_extra = len([r for r in all_records if normalize_extra(r.get('extra')) == extra_key])
        row = f"{'ALL MODELS':<45} | {n_all_extra:>6}"
        for key in cat_keys:
            score = all_mastery[mastery_key][key]['score'] * 100
            n_cat = int(all_mastery[mastery_key][key]['n'])
            row += f" | {score:>6.1f}% ({n_cat:>4})"
        table_lines.append(row)
        table_lines.append("=" * 170)

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

        # Data rows (using combined model names with COT suffix)
        for display_name in combined_model_order:
            mastery = combined_model_mastery[display_name]
            for extra_label, extra_key in csv_extra_mapping:
                row = f"{display_name},{extra_label}"
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

    # Build detailed breakdown file with combined model names (COT suffix)
    detail_lines = build_mastery_detail_lines(
        all_records, combined_model_records, combined_model_order,
        "Detailed Mastery Category Breakdown"
    )
    mastery_detail_path = os.path.join(logs_dir, 'mastery_detail.txt')
    with open(mastery_detail_path, 'w') as f:
        f.write("\n".join(detail_lines))

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
    saved_files.extend([os.path.join(logs_dir, fn) for fn in extra_file_names.values()])

    # Generate mastery_detail files broken out by extra type only (consolidated with COT suffix)
    extra_types = ['0A', '0B', '1A', '1B']

    for extra_type in extra_types:
        # Check if there are records for this extra type
        extra_recs = [r for r in all_records if normalize_extra(r.get('extra')) == extra_type]
        if not extra_recs:
            continue

        filename = f'mastery_detail_{extra_type}.txt'
        extra_name = EXTRA_CATEGORIES[extra_type]['name']
        title = f"Detailed Mastery Category Breakdown ({extra_name} - {extra_type})"

        detail_lines = build_mastery_detail_lines(
            all_records, combined_model_records, combined_model_order,
            title, extra_filter=extra_type
        )
        filepath = os.path.join(logs_dir, filename)
        with open(filepath, 'w') as f:
            f.write("\n".join(detail_lines))
        saved_files.append(filepath)

    # Generate per-category mastery files (load conditions as columns)
    # Format: one file per mastery category showing models × load conditions
    for cat_key, cat_info in TOM_MASTERY_CATEGORIES.items():
        cat_name = cat_info['name']
        cat_filename = f"mastery_cat_{cat_key}.txt"

        # Build header with load conditions as columns
        cat_header = f"{'Model':<45} | {'N':>6} | {'0A':>14} | {'0B':>14} | {'1A':>14} | {'1B':>14} | {'Overall':>14}"
        cat_lines = [
            f"ToM MASTERY: {cat_name}",
            cat_info['description'],
            "=" * 140,
            cat_header,
            "-" * 140
        ]

        for display_name in combined_model_order:
            mastery = combined_model_mastery[display_name]
            n_records = len(combined_model_records[display_name])

            # Get scores and N for each extra category
            s_0a = mastery['extra0a'][cat_key]['score'] * 100
            n_0a = int(mastery['extra0a'][cat_key]['n'])
            s_0b = mastery['extra0b'][cat_key]['score'] * 100
            n_0b = int(mastery['extra0b'][cat_key]['n'])
            s_1a = mastery['extra1a'][cat_key]['score'] * 100
            n_1a = int(mastery['extra1a'][cat_key]['n'])
            s_1b = mastery['extra1b'][cat_key]['score'] * 100
            n_1b = int(mastery['extra1b'][cat_key]['n'])
            s_ov = mastery['overall'][cat_key]['score'] * 100
            n_ov = int(mastery['overall'][cat_key]['n'])

            row = f"{display_name:<45} | {n_records:>6}"
            row += f" | {s_0a:>5.1f}% ({n_0a:>3})"
            row += f" | {s_0b:>5.1f}% ({n_0b:>3})"
            row += f" | {s_1a:>5.1f}% ({n_1a:>3})"
            row += f" | {s_1b:>5.1f}% ({n_1b:>3})"
            row += f" | {s_ov:>5.1f}% ({n_ov:>3})"
            cat_lines.append(row)

        cat_lines.append("-" * 140)

        # All models row
        n_all = len(all_records)
        s_0a = all_mastery['extra0a'][cat_key]['score'] * 100
        n_0a = int(all_mastery['extra0a'][cat_key]['n'])
        s_0b = all_mastery['extra0b'][cat_key]['score'] * 100
        n_0b = int(all_mastery['extra0b'][cat_key]['n'])
        s_1a = all_mastery['extra1a'][cat_key]['score'] * 100
        n_1a = int(all_mastery['extra1a'][cat_key]['n'])
        s_1b = all_mastery['extra1b'][cat_key]['score'] * 100
        n_1b = int(all_mastery['extra1b'][cat_key]['n'])
        s_ov = all_mastery['overall'][cat_key]['score'] * 100
        n_ov = int(all_mastery['overall'][cat_key]['n'])

        row = f"{'ALL MODELS':<45} | {n_all:>6}"
        row += f" | {s_0a:>5.1f}% ({n_0a:>3})"
        row += f" | {s_0b:>5.1f}% ({n_0b:>3})"
        row += f" | {s_1a:>5.1f}% ({n_1a:>3})"
        row += f" | {s_1b:>5.1f}% ({n_1b:>3})"
        row += f" | {s_ov:>5.1f}% ({n_ov:>3})"
        cat_lines.append(row)
        cat_lines.append("=" * 140)

        cat_filepath = os.path.join(logs_dir, cat_filename)
        with open(cat_filepath, 'w') as f:
            f.write("\n".join(cat_lines))
        saved_files.append(cat_filepath)

    # Generate mastery category comparison chart
    # Select representative models: top performers, mid-tier, and aggregate
    # Sort models by overall accuracy to pick representatives
    model_overall_acc = []
    for display_name in combined_model_order:
        stats = compute_stats(combined_model_records[display_name], lies_okay=False)
        model_overall_acc.append((display_name, stats['overall']['rate']))
    model_overall_acc.sort(key=lambda x: -x[1])  # Sort by accuracy descending

    # Select models for chart: top 3, middle 2, bottom 1, plus ALL MODELS
    n_models = len(model_overall_acc)
    selected_models = []
    if n_models >= 6:
        selected_models = [
            model_overall_acc[0][0],  # Top 1
            model_overall_acc[1][0],  # Top 2
            model_overall_acc[2][0],  # Top 3
            model_overall_acc[n_models // 2][0],  # Middle
            model_overall_acc[-2][0],  # Near bottom
        ]
    else:
        selected_models = [m[0] for m in model_overall_acc[:min(5, n_models)]]

    # Create multi-panel figure (2 rows × 4 cols, last panel for legend/summary)
    fig_mastery, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    cat_names_short = {
        'self_knowledge_belief': 'Self Knowledge',
        'teammate_knowledge_belief': 'Teammate Knowledge',
        'combined_uncertainty': 'Combined Uncertainty',
        'true_false_belief': 'True vs False Belief',
        'teammate_opponent': 'Teammate vs Opponent',
        'strategic_lies': 'Strategic Lies',
    }

    load_colors = {
        '0A': EXTRA_CATEGORIES['0A']['color'],
        '0B': EXTRA_CATEGORIES['0B']['color'],
        '1A': EXTRA_CATEGORIES['1A']['color'],
        '1B': EXTRA_CATEGORIES['1B']['color'],
    }

    for i, (cat_key, cat_info) in enumerate(TOM_MASTERY_CATEGORIES.items()):
        ax = axes[i]

        # Data for this category
        model_labels = []
        scores_0a, scores_0b, scores_1a, scores_1b = [], [], [], []

        for model in selected_models:
            mastery = combined_model_mastery[model]
            # Shorten model name for display
            short_name = model.replace('-instruct', '').replace('anthropic-', '').replace('-24b', '')
            short_name = short_name[:25] + '...' if len(short_name) > 28 else short_name
            model_labels.append(short_name)
            scores_0a.append(mastery['extra0a'][cat_key]['score'] * 100)
            scores_0b.append(mastery['extra0b'][cat_key]['score'] * 100)
            scores_1a.append(mastery['extra1a'][cat_key]['score'] * 100)
            scores_1b.append(mastery['extra1b'][cat_key]['score'] * 100)

        # Add ALL MODELS aggregate
        model_labels.append('ALL MODELS')
        scores_0a.append(all_mastery['extra0a'][cat_key]['score'] * 100)
        scores_0b.append(all_mastery['extra0b'][cat_key]['score'] * 100)
        scores_1a.append(all_mastery['extra1a'][cat_key]['score'] * 100)
        scores_1b.append(all_mastery['extra1b'][cat_key]['score'] * 100)

        x = np.arange(len(model_labels))
        width = 0.2

        ax.bar(x - 1.5*width, scores_0a, width, label='0A: Min Events', color=load_colors['0A'])
        ax.bar(x - 0.5*width, scores_0b, width, label='0B: Event Load', color=load_colors['0B'])
        ax.bar(x + 0.5*width, scores_1a, width, label='1A: Min ECT', color=load_colors['1A'])
        ax.bar(x + 1.5*width, scores_1b, width, label='1B: ECT Load', color=load_colors['1B'])

        ax.set_ylabel('Mastery Score (%)', fontsize=10)
        ax.set_title(cat_names_short.get(cat_key, cat_info['name']), fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 105)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.grid(axis='y', alpha=0.3)

    # Use last panels for legend and hide empty panels
    for idx in range(len(TOM_MASTERY_CATEGORIES), len(axes)):
        axes[idx].axis('off')
    handles = [
        Patch(facecolor=load_colors['0A'], label='0A: Minimal Events'),
        Patch(facecolor=load_colors['0B'], label='0B: Event Load'),
        Patch(facecolor=load_colors['1A'], label='1A: Minimal ECT'),
        Patch(facecolor=load_colors['1B'], label='1B: ECT Load'),
    ]
    # Put legend in panel after all categories
    legend_idx = len(TOM_MASTERY_CATEGORIES)
    axes[legend_idx].legend(handles=handles, loc='center', fontsize=12, title='Load Conditions', title_fontsize=13)

    # Add note about model selection
    axes[legend_idx].text(0.5, 0.3, f'Showing {len(selected_models)} representative models\n(top, middle, bottom performers)\nplus ALL MODELS aggregate',
                 ha='center', va='center', fontsize=10, style='italic', transform=axes[legend_idx].transAxes)

    fig_mastery.suptitle('ToM Mastery by Category and Load Condition', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    mastery_chart_path = os.path.join(logs_dir, 'mastery_by_category.png')
    plt.savefig(mastery_chart_path, dpi=150, bbox_inches='tight')
    saved_files.append(mastery_chart_path)
    plt.close(fig_mastery)

    # Generate mastery overall bar chart (all models, all categories)
    # This is a direct visualization of mastery_overall.txt
    # Now includes 7 panels: 6 categories + teammate_opponent with lies_okay
    cat_colors = {
        'self_knowledge_belief': '#1f77b4',       # blue
        'teammate_knowledge_belief': '#ff7f0e',   # orange
        'combined_uncertainty': '#2ca02c',        # green
        'true_false_belief': '#d62728',           # red
        'teammate_opponent': '#9467bd',           # purple
        'strategic_lies': '#17becf',              # cyan
        'teammate_opponent_lies': '#8c564b',      # brown (lies okay version)
    }

    # Build list of panels to show: 6 categories + lies_okay version of teammate_opponent
    chart_panels = list(TOM_MASTERY_CATEGORIES.keys()) + ['teammate_opponent_lies']
    chart_names = {**cat_names_short, 'teammate_opponent_lies': 'TM vs Opp (lies ok)', 'strategic_lies': 'Strategic Lies'}

    # Build data: all models + ALL MODELS row
    all_model_names = list(combined_model_order) + ['ALL MODELS']

    # Create figure with 7 subplots (one per category) - 2x4 grid (one empty)
    fig_overall, axes_overall = plt.subplots(2, 4, figsize=(20, 14))
    axes_overall = axes_overall.flatten()

    for i, cat_key in enumerate(chart_panels):
        ax = axes_overall[i]
        cat_name = chart_names.get(cat_key, cat_key)
        color = cat_colors[cat_key]

        # Get scores for all models with error bar data
        scores = []
        labels = []
        err_lower = []
        err_upper = []
        for model in combined_model_order:
            if cat_key == 'teammate_opponent_lies':
                # Use lies_okay version for this special panel
                mastery = combined_model_mastery_lies[model]
                cat_data = mastery['overall']['teammate_opponent']
            else:
                mastery = combined_model_mastery[model]
                cat_data = mastery['overall'][cat_key]
            score = cat_data['score'] * 100
            k = int(cat_data['k'])
            n = int(cat_data['n'])
            # Compute Wilson CI (clamp to non-negative for edge cases)
            ci_low, ci_high = wilson_ci(k, n)
            err_lower.append(max(0, score - ci_low * 100))
            err_upper.append(max(0, ci_high * 100 - score))
            scores.append(score)
            # Shorten model name for display
            short_name = model.replace('-instruct', '').replace('anthropic-', '').replace('-24b-', '-')
            short_name = short_name.replace('mistral-small', 'mist-sm').replace('mistral-large', 'mist-lg')
            short_name = short_name.replace('qwen3-next-80b-a3b', 'qwen3-80b')
            labels.append(short_name)

        # Add ALL MODELS
        if cat_key == 'teammate_opponent_lies':
            cat_data = all_mastery_lies['overall']['teammate_opponent']
        else:
            cat_data = all_mastery['overall'][cat_key]
        scores.append(cat_data['score'] * 100)
        k = int(cat_data['k'])
        n = int(cat_data['n'])
        ci_low, ci_high = wilson_ci(k, n)
        err_lower.append(max(0, scores[-1] - ci_low * 100))
        err_upper.append(max(0, ci_high * 100 - scores[-1]))
        labels.append('ALL MODELS')

        # Create horizontal bar chart with error bars
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, scores, xerr=[err_lower, err_upper], color=color, alpha=0.8, height=0.7,
                       capsize=2, error_kw={'elinewidth': 0.8, 'capthick': 0.8, 'alpha': 0.6})

        # Highlight ALL MODELS bar
        bars[-1].set_alpha(1.0)
        bars[-1].set_edgecolor('black')
        bars[-1].set_linewidth(2)

        # Add value labels on bars
        for j, (score, bar) in enumerate(zip(scores, bars)):
            ax.text(score + 1, bar.get_y() + bar.get_height()/2,
                   f'{score:.0f}%', va='center', fontsize=7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Mastery Score (%)', fontsize=9)
        ax.set_xlim(0, 110)
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(cat_name, fontsize=11, fontweight='bold', color=color)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()  # Top model at top

    # Hide empty subplot(s) in the 2x4 grid
    for idx in range(len(chart_panels), len(axes_overall)):
        axes_overall[idx].axis('off')

    fig_overall.suptitle('ToM Mastery Scores by Category (Overall)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    mastery_overall_chart_path = os.path.join(logs_dir, 'mastery_overall_chart.png')
    plt.savefig(mastery_overall_chart_path, dpi=150, bbox_inches='tight')
    saved_files.append(mastery_overall_chart_path)
    plt.close(fig_overall)

    # Generate single combined charts (models with COT suffix shown together)

    # Compute per-scenario stats
    scenario_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in all_records:
        extra_str = normalize_extra(r.get('extra'))
        key = (r.get('scenario_id'), extra_str)
        scenario_stats[key]['total'] += 1
        if is_success(r, lies_okay=False):
            scenario_stats[key]['correct'] += 1

    # Sort by accuracy (ascending = hardest first)
    sorted_scenarios = sorted(
        scenario_stats.items(),
        key=lambda x: x[1]['correct'] / x[1]['total'] if x[1]['total'] > 0 else 0
    )

    # Save per-scenario stats to file
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

    # Generate per-model per-scenario stats
    model_scenario_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    for model_name in combined_model_order:
        for r in combined_model_records[model_name]:
            extra_str = normalize_extra(r.get('extra'))
            key = (r.get('scenario_id'), extra_str)
            model_scenario_stats[model_name][key]['total'] += 1
            if is_success(r, lies_okay=False):
                model_scenario_stats[model_name][key]['correct'] += 1

    # Build per-model per-scenario output lines
    model_scenario_lines = [
        f"{'Model':<45} | {'Scenario':>8} | {'Extra':>12} | {'N':>4} | {'Accuracy':>10}",
        "-" * 90
    ]

    # Sort models, then within each model sort by (scenario_id, extra)
    for model_name in combined_model_order:
        model_stats = model_scenario_stats[model_name]
        sorted_keys = sorted(model_stats.keys(), key=lambda x: (int(x[0]) if str(x[0]).isdigit() else 0, x[1]))
        for (scenario_id, extra) in sorted_keys:
            stats = model_stats[(scenario_id, extra)]
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                extra_name = EXTRA_CATEGORIES.get(extra, {}).get('short', extra)
                model_scenario_lines.append(
                    f"{model_name:<45} | {scenario_id:>8} | {extra_name:>12} | {stats['total']:>4} | {acc*100:>9.1f}%"
                )
        model_scenario_lines.append("")  # Blank line between models

    model_scenario_stats_path = os.path.join(logs_dir, 'per_model_scenario_stats.txt')
    with open(model_scenario_stats_path, 'w') as f:
        f.write("\n".join(model_scenario_lines))
    saved_files.append(model_scenario_stats_path)

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
        ax2.set_title('Per-Scenario Accuracy', fontsize=14, fontweight='bold')
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
        scenario_chart_path = os.path.join(logs_dir, 'scenario_difficulty.png')
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
        ax3.set_title('Accuracy by Scenario and Extra Category', fontsize=14, fontweight='bold')
        ax3.set_xticks(x3)
        ax3.set_xticklabels(scenario_ids, fontsize=8)
        ax3.set_ylim(0, 105)
        ax3.legend(loc='lower right')
        ax3.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        paired_chart_path = os.path.join(logs_dir, 'extra_comparison.png')
        plt.savefig(paired_chart_path, dpi=150, bbox_inches='tight')
        saved_files.append(paired_chart_path)
        plt.close(fig3)

    # Generate model performance bar chart (with COT suffix in model names)
    # Compute stats for each combined model entry
    combined_model_stats = {}
    for display_name in combined_model_order:
        combined_model_stats[display_name] = compute_stats(combined_model_records[display_name], lies_okay=False)

    if combined_model_order:
        fig, ax = plt.subplots(figsize=(18, 8))
        x = np.arange(len(combined_model_order))
        width = 0.15

        overall_rates = [combined_model_stats[m]['overall']['rate'] * 100 for m in combined_model_order]
        extra0a_rates = [combined_model_stats[m]['extra0a']['rate'] * 100 for m in combined_model_order]
        extra0b_rates = [combined_model_stats[m]['extra0b']['rate'] * 100 for m in combined_model_order]
        extra1a_rates = [combined_model_stats[m]['extra1a']['rate'] * 100 for m in combined_model_order]
        extra1b_rates = [combined_model_stats[m]['extra1b']['rate'] * 100 for m in combined_model_order]

        overall_errs = [(combined_model_stats[m]['overall']['ci'][1] - combined_model_stats[m]['overall']['ci'][0]) / 2 * 100 for m in combined_model_order]
        extra0a_errs = [(combined_model_stats[m]['extra0a']['ci'][1] - combined_model_stats[m]['extra0a']['ci'][0]) / 2 * 100 for m in combined_model_order]
        extra0b_errs = [(combined_model_stats[m]['extra0b']['ci'][1] - combined_model_stats[m]['extra0b']['ci'][0]) / 2 * 100 for m in combined_model_order]
        extra1a_errs = [(combined_model_stats[m]['extra1a']['ci'][1] - combined_model_stats[m]['extra1a']['ci'][0]) / 2 * 100 for m in combined_model_order]
        extra1b_errs = [(combined_model_stats[m]['extra1b']['ci'][1] - combined_model_stats[m]['extra1b']['ci'][0]) / 2 * 100 for m in combined_model_order]

        ax.bar(x - 2*width, overall_rates, width, label='Overall', yerr=overall_errs, capsize=2, color='#2ecc71')
        ax.bar(x - width, extra0a_rates, width, label=f"0A: {EXTRA_CATEGORIES['0A']['name']}", yerr=extra0a_errs, capsize=2, color=EXTRA_CATEGORIES['0A']['color'])
        ax.bar(x, extra0b_rates, width, label=f"0B: {EXTRA_CATEGORIES['0B']['name']}", yerr=extra0b_errs, capsize=2, color=EXTRA_CATEGORIES['0B']['color'])
        ax.bar(x + width, extra1a_rates, width, label=f"1A: {EXTRA_CATEGORIES['1A']['name']}", yerr=extra1a_errs, capsize=2, color=EXTRA_CATEGORIES['1A']['color'])
        ax.bar(x + 2*width, extra1b_rates, width, label=f"1B: {EXTRA_CATEGORIES['1B']['name']}", yerr=extra1b_errs, capsize=2, color=EXTRA_CATEGORIES['1B']['color'])

        ax.set_ylabel('Optimal Action Rate (%)', fontsize=12)
        ax.set_title('ToM Test Performance by Model', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(combined_model_order, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim(0, 105)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        chart_path = os.path.join(logs_dir, 'performance_comparison.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        saved_files.append(chart_path)
        plt.close(fig)

    # Print high-level summary
    print("\n" + "=" * 60)
    print("ToM ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Base models: {len(sorted_models)}, Model variants (incl. COT): {len(combined_model_order)}")
    print(f"Total records: {len(all_records)}")
    print()

    # Show top-level results per model (with COT suffix)
    print(f"{'Model':<40} | {'Records':>7} | {'Accuracy':>10}")
    print("-" * 65)
    for display_name in combined_model_order:
        stats = combined_model_stats[display_name]
        print(f"{display_name:<40} | {stats['n']:>7} | {stats['overall']['rate']*100:>9.1f}%")
    print("-" * 65)

    # Show files saved
    print(f"\nSaved {len(saved_files)} files to {logs_dir}/")
    print("  Tables: summary_table.txt, per_scenario_stats.txt, per_model_scenario_stats.txt")
    print("  Mastery: mastery_*.txt, mastery_scores.csv")
    print("  Charts: performance_comparison.png, scenario_difficulty.png, extra_comparison.png")

    plt.show()


if __name__ == '__main__':
    main()
