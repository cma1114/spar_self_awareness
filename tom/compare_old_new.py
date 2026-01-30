#!/usr/bin/env python3
"""
Compare old vs new ToM test results.

Compares accuracy and mastery scores between old (pre-fix) and new (post-fix)
Extra=1B scenario results.
"""

import json
import math
import os
from typing import List, Tuple, Dict, Optional


def normalize_extra(val):
    """Normalize Extra field to string format for backward compatibility.

    Legacy int values are converted to new string format:
    - None or 0 → '1A' (legacy Extra=0 behavior)
    - 1 → '1B' (legacy Extra=1 behavior)
    """
    if val is None or val == 0: return '1A'  # Legacy Extra=0 → 1A
    if val == 1: return '1B'                  # Legacy Extra=1 → 1B
    if val in ('0A', '0B', '1A', '1B'): return val
    return str(val)

# File pairs: (new_file, old_file)
# Format: (new_timestamp, old_timestamp) - higher timestamp = newer
FILE_PAIRS = [
    # anthropic-claude-opus-4.5 (non-thinking)
    ('anthropic-claude-opus-4.5_1769101131_game_data.json', 'anthropic-claude-opus-4.5_1768760292_game_data.json'),
    # anthropic-claude-opus-4.5_think
    ('anthropic-claude-opus-4.5_think_1769101823_game_data.json', 'anthropic-claude-opus-4.5_think_1768768643_game_data.json'),
    # deepseek-chat-v3.1
    ('deepseek-chat-v3.1_1769102919_game_data.json', 'deepseek-chat-v3.1_1768758058_game_data.json'),
    # openai-gpt-5.2 (non-thinking)
    ('openai-gpt-5.2_1769098709_game_data.json', 'openai-gpt-5.2_1768758353_game_data.json'),
    # openai-gpt-5.2_think
    ('openai-gpt-5.2_think_1768795768_game_data.json', 'openai-gpt-5.2_think_1769107960_game_data.json'),
]

# ToM Mastery Categories (copied from analyze_results.py)
TOM_MASTERY_CATEGORIES = {
    'self_knowledge_belief': {
        'name': 'Self: Knowledge vs Belief',
        'components': [
            {'scenarios': [7, 8, 9], 'action': 'Pass', 'weight': 2},
            {'scenarios': [12, 13], 'action': 'Ask', 'weight': 3},
        ],
    },
    'teammate_knowledge_belief': {
        'name': 'Teammate: Knowledge vs Belief',
        'components': [
            {'scenarios': [20, 21, 22], 'action': 'Pass', 'weight': 1},
            {'scenarios': [17, 18, 19], 'action': 'Tell', 'weight': 1},
        ],
    },
    'combined_uncertainty': {
        'name': 'Combined Uncertainty',
        'components': [
            {'scenarios': [10, 11, 23, 24], 'action': 'Pass', 'weight': 1},
        ],
    },
    'true_false_belief': {
        'name': 'True vs False Belief',
        'components': [
            {'scenarios': [14, 15, 16], 'action': 'Pass', 'weight': 1},
            {'scenarios': [17, 18, 19], 'action': 'Tell', 'weight': 1},
        ],
    },
    'teammate_opponent': {
        'name': 'Teammate vs Opponent',
        'components': [
            {'scenarios': [12, 13], 'action': 'Ask', 'weight': 1},
            {'scenarios': [37, 39], 'action': 'Pass', 'weight': 1},
            {'scenarios': [30, 31, 32], 'action': 'Pass', 'weight': 1},
            {'scenarios': [17, 18, 19], 'action': 'Tell', 'weight': 1},
        ],
    },
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


def load_game_data(filepath: str) -> List[dict]:
    """Load and return records from a game_data.json file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def filter_player_a(records: List[dict]) -> List[dict]:
    """Filter to only player A's records."""
    return [r for r in records if r.get('character') == 'A']


def compute_accuracy(records: List[dict], extra_filter: Optional[str] = None) -> Dict:
    """Compute accuracy stats for records."""
    if extra_filter is not None:
        records = [r for r in records if normalize_extra(r.get('extra')) == extra_filter]

    n = len(records)
    k = sum(1 for r in records if r.get('was_optimal'))
    rate = k / n if n > 0 else 0
    ci = wilson_ci(k, n)

    return {'n': n, 'k': k, 'rate': rate, 'ci': ci}


def compute_mastery_score(records: List[dict], category: dict, extra_filter: Optional[str] = None) -> Dict:
    """Compute mastery score for a single ToM category."""
    if extra_filter is not None:
        records = [r for r in records if normalize_extra(r.get('extra')) == extra_filter]

    total_weighted = 0
    correct_weighted = 0

    for component in category['components']:
        scenarios = component['scenarios']
        weight = component['weight']

        component_records = [r for r in records
                           if r.get('scenario_id') in [str(s) for s in scenarios]]

        n_total = len(component_records)
        n_correct = sum(1 for r in component_records if r.get('was_optimal'))

        total_weighted += n_total * weight
        correct_weighted += n_correct * weight

    score = correct_weighted / total_weighted if total_weighted > 0 else 0
    return {'score': score, 'n': total_weighted, 'k': correct_weighted}


def format_diff(old_val: float, new_val: float) -> str:
    """Format a difference with + or - sign."""
    diff = new_val - old_val
    sign = '+' if diff >= 0 else ''
    return f"{sign}{diff:.1f}%"


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, 'tom_llm_logs')

    print("=" * 100)
    print("COMPARISON: Old vs New Extra=1B Scenario Results")
    print("=" * 100)

    for new_file, old_file in FILE_PAIRS:
        new_path = os.path.join(logs_dir, new_file)
        old_path = os.path.join(logs_dir, old_file)

        # Check files exist
        if not os.path.exists(new_path):
            print(f"\nWarning: New file not found: {new_file}")
            continue
        if not os.path.exists(old_path):
            print(f"\nWarning: Old file not found: {old_file}")
            continue

        # Load data
        new_records = filter_player_a(load_game_data(new_path))
        old_records = filter_player_a(load_game_data(old_path))

        # Extract model name from filename
        model_name = new_file.split('_')[0] + '-' + new_file.split('_')[1]
        if 'think' in new_file:
            model_name = new_file.rsplit('_', 2)[0]

        print(f"\n{'=' * 80}")
        print(f"Model: {model_name}")
        print(f"Old file: {old_file} ({len(old_records)} records)")
        print(f"New file: {new_file} ({len(new_records)} records)")
        print(f"{'=' * 80}")

        # Overall accuracy comparison
        print("\n--- ACCURACY COMPARISON ---")
        print(f"{'Metric':<20} | {'Old':>12} | {'New':>12} | {'Diff':>10}")
        print("-" * 60)

        for label, extra_val in [('Overall', None), ('Extra=1A', '1A'), ('Extra=1B', '1B')]:
            old_stats = compute_accuracy(old_records, extra_val)
            new_stats = compute_accuracy(new_records, extra_val)

            old_pct = old_stats['rate'] * 100
            new_pct = new_stats['rate'] * 100
            diff_str = format_diff(old_pct, new_pct)

            print(f"{label:<20} | {old_pct:>11.1f}% | {new_pct:>11.1f}% | {diff_str:>10}")

        # Mastery scores comparison
        print("\n--- MASTERY SCORES COMPARISON (Overall) ---")
        print(f"{'Category':<30} | {'Old':>10} | {'New':>10} | {'Diff':>10}")
        print("-" * 70)

        for key, category in TOM_MASTERY_CATEGORIES.items():
            old_mastery = compute_mastery_score(old_records, category)
            new_mastery = compute_mastery_score(new_records, category)

            old_pct = old_mastery['score'] * 100
            new_pct = new_mastery['score'] * 100
            diff_str = format_diff(old_pct, new_pct)

            print(f"{category['name']:<30} | {old_pct:>9.1f}% | {new_pct:>9.1f}% | {diff_str:>10}")

        # Extra=1B mastery scores comparison (most relevant for the fix)
        print("\n--- MASTERY SCORES COMPARISON (Extra=1B only) ---")
        print(f"{'Category':<30} | {'Old':>10} | {'New':>10} | {'Diff':>10}")
        print("-" * 70)

        for key, category in TOM_MASTERY_CATEGORIES.items():
            old_mastery = compute_mastery_score(old_records, category, extra_filter='1B')
            new_mastery = compute_mastery_score(new_records, category, extra_filter='1B')

            old_pct = old_mastery['score'] * 100
            new_pct = new_mastery['score'] * 100
            diff_str = format_diff(old_pct, new_pct)

            print(f"{category['name']:<30} | {old_pct:>9.1f}% | {new_pct:>9.1f}% | {diff_str:>10}")

        # Per-scenario comparison for Extra=1B
        print("\n--- PER-SCENARIO ACCURACY (Extra=1B only) ---")
        print(f"{'Scenario':<10} | {'Old':>10} | {'New':>10} | {'Diff':>10}")
        print("-" * 50)

        # Get unique scenario IDs
        scenario_ids = sorted(set(r.get('scenario_id') for r in old_records + new_records
                                  if r.get('scenario_id')), key=lambda x: int(x))

        for sid in scenario_ids:
            old_scenario = [r for r in old_records
                          if r.get('scenario_id') == sid and normalize_extra(r.get('extra')) == '1B']
            new_scenario = [r for r in new_records
                          if r.get('scenario_id') == sid and normalize_extra(r.get('extra')) == '1B']

            if not old_scenario or not new_scenario:
                continue

            old_acc = sum(1 for r in old_scenario if r.get('was_optimal')) / len(old_scenario) * 100
            new_acc = sum(1 for r in new_scenario if r.get('was_optimal')) / len(new_scenario) * 100
            diff_str = format_diff(old_acc, new_acc)

            print(f"{sid:<10} | {old_acc:>9.1f}% | {new_acc:>9.1f}% | {diff_str:>10}")

    print("\n" + "=" * 100)
    print("Comparison complete.")


if __name__ == '__main__':
    main()
