#!/usr/bin/env python3
"""
Analyze reasoning traces from ToM test logs.

Parses log files to identify which turns used reasoning, correlates with
game_data.json for accuracy, and computes statistics on reasoning usage
and its impact on performance.
"""

import json
import glob
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def extract_model_name(filepath: str) -> str:
    """Extract model name from filename, stripping timestamp."""
    basename = os.path.basename(filepath)
    # Pattern: {model}_{timestamp}.log or {model}_{timestamp}_game_data.json
    match = re.match(r'(.+?)_\d+(?:_game_data)?\.(?:log|json)$', basename)
    if match:
        return match.group(1)
    return basename


def find_paired_files(logs_dir: str) -> List[Tuple[str, str, str]]:
    """
    Find log files with matching game_data.json files.
    Returns list of (model_name, log_path, json_path) tuples.
    Only includes files where JSON has A records as multiple of 78.
    """
    pairs = []
    log_files = glob.glob(os.path.join(logs_dir, '**', '*_think*.log'), recursive=True)

    for log_path in log_files:
        base = log_path.replace('.log', '')
        json_path = base + '_game_data.json'

        if not os.path.exists(json_path):
            continue

        # Check if JSON has valid A record count
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            a_records = [r for r in data if r.get('character') == 'A']
            if len(a_records) == 0 or len(a_records) % 78 != 0:
                continue
        except (json.JSONDecodeError, IOError):
            continue

        model_name = extract_model_name(log_path)
        pairs.append((model_name, log_path, json_path))

    return pairs


def parse_log_reasoning(log_path: str) -> List[bool]:
    """
    Parse a log file and return a list of bools indicating whether
    each player A turn had a reasoning trace.

    Uses spec headers ("--- Running Spec N/M: {...} ---") to delimit scenarios.
    Each spec corresponds to one A turn. Within each spec section, checks
    for "REASONING TRACE:" after "It is your turn.".
    """
    with open(log_path, 'r') as f:
        content = f.read()

    reasoning_flags = []

    # Split by spec headers
    spec_pattern = r'--- Running Spec \d+/\d+:.*?---'
    parts = re.split(spec_pattern, content)

    # First part is before any spec, skip it. Each subsequent part is one spec's content.
    for part in parts[1:]:
        # Each spec has exactly one A turn (player A is the test subject)
        # Check if this spec section has "It is your turn." followed by "REASONING TRACE:"
        if "It is your turn." in part:
            # Find the A turn section
            a_turn_idx = part.find("It is your turn.")
            # Look for reasoning trace after this
            section_after_turn = part[a_turn_idx:]
            # Check for reasoning before the next turn or end of spec
            next_turn = section_after_turn.find("It is ", 1)  # Skip the one we found
            if next_turn == -1:
                check_section = section_after_turn
            else:
                check_section = section_after_turn[:next_turn]

            has_reasoning = "REASONING TRACE:" in check_section
            reasoning_flags.append(has_reasoning)
        else:
            # No A turn in this spec section (shouldn't happen but handle gracefully)
            reasoning_flags.append(False)

    return reasoning_flags


def load_and_annotate_records(json_path: str, reasoning_flags: List[bool]) -> List[dict]:
    """
    Load game_data.json, filter to A records, and annotate with reasoning flags.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    a_records = [r for r in data if r.get('character') == 'A']

    # Match reasoning flags to A records by order
    if len(reasoning_flags) != len(a_records):
        print(f"Warning: reasoning count ({len(reasoning_flags)}) != A records ({len(a_records)}) in {json_path}")
        # Use what we can
        min_len = min(len(reasoning_flags), len(a_records))
        reasoning_flags = reasoning_flags[:min_len]
        a_records = a_records[:min_len]

    for record, has_reasoning in zip(a_records, reasoning_flags):
        record['has_reasoning'] = has_reasoning

    return a_records


def compute_model_stats(records: List[dict]) -> dict:
    """Compute stats for a set of records from one model."""
    n_total = len(records)
    n_with_reasoning = sum(1 for r in records if r.get('has_reasoning'))
    n_without_reasoning = n_total - n_with_reasoning

    # Accuracy with reasoning
    with_reasoning = [r for r in records if r.get('has_reasoning')]
    n_optimal_with = sum(1 for r in with_reasoning if r.get('was_optimal'))
    acc_with = n_optimal_with / len(with_reasoning) if with_reasoning else None

    # Accuracy without reasoning
    without_reasoning = [r for r in records if not r.get('has_reasoning')]
    n_optimal_without = sum(1 for r in without_reasoning if r.get('was_optimal'))
    acc_without = n_optimal_without / len(without_reasoning) if without_reasoning else None

    return {
        'n_total': n_total,
        'n_with_reasoning': n_with_reasoning,
        'n_without_reasoning': n_without_reasoning,
        'reasoning_rate': n_with_reasoning / n_total if n_total > 0 else 0,
        'acc_with_reasoning': acc_with,
        'acc_without_reasoning': acc_without,
        'n_optimal_with': n_optimal_with,
        'n_optimal_without': n_optimal_without,
    }


def compute_scenario_stats(all_records: List[dict]) -> Dict[str, dict]:
    """
    Compute stats grouped by scenario_id and extra.
    Returns dict keyed by "scenario_id|extra" with reasoning frequency and accuracy.
    """
    grouped = defaultdict(list)
    for r in all_records:
        key = f"{r.get('scenario_id')}|{r.get('extra', 0) or 0}"
        grouped[key].append(r)

    stats = {}
    for key, records in grouped.items():
        n_total = len(records)
        n_with_reasoning = sum(1 for r in records if r.get('has_reasoning'))
        with_reasoning = [r for r in records if r.get('has_reasoning')]
        n_optimal_when_reasoning = sum(1 for r in with_reasoning if r.get('was_optimal'))

        stats[key] = {
            'n_total': n_total,
            'n_with_reasoning': n_with_reasoning,
            'reasoning_rate': n_with_reasoning / n_total if n_total > 0 else 0,
            'acc_when_reasoning': n_optimal_when_reasoning / n_with_reasoning if n_with_reasoning > 0 else None,
        }

    return stats


def fisher_exact_test(a: int, b: int, c: int, d: int) -> Optional[float]:
    """
    Compute Fisher's exact test p-value for a 2x2 contingency table:
           Optimal  Not Optimal
    With     a          b
    Without  c          d

    Returns p-value or None if scipy unavailable.
    """
    try:
        from scipy.stats import fisher_exact
        _, p_value = fisher_exact([[a, b], [c, d]])
        return p_value
    except ImportError:
        return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, 'tom_llm_logs')

    if not os.path.exists(logs_dir):
        print(f"Error: {logs_dir} not found")
        return

    # Find paired files
    pairs = find_paired_files(logs_dir)
    print(f"Found {len(pairs)} valid log/json pairs with thinking models")

    if not pairs:
        print("No valid thinking model data found.")
        return

    # Process each pair and aggregate by model
    model_records: Dict[str, List[dict]] = defaultdict(list)
    all_records: List[dict] = []

    for model_name, log_path, json_path in pairs:
        reasoning_flags = parse_log_reasoning(log_path)
        records = load_and_annotate_records(json_path, reasoning_flags)
        model_records[model_name].extend(records)
        all_records.extend(records)

    # Compute per-model stats
    print("\n" + "=" * 90)
    print("=== Reasoning Trace Analysis ===")
    print("=" * 90)

    print(f"\n{'Model':<35} | {'Turns':>6} | {'Reason%':>8} | {'Acc w/ Rsn':>11} | {'Acc w/o Rsn':>11}")
    print("-" * 90)

    model_stats = {}
    for model_name in sorted(model_records.keys()):
        records = model_records[model_name]
        stats = compute_model_stats(records)
        model_stats[model_name] = stats

        acc_with_str = f"{stats['acc_with_reasoning']*100:6.1f}%" if stats['acc_with_reasoning'] is not None else "   N/A"
        acc_without_str = f"{stats['acc_without_reasoning']*100:6.1f}%" if stats['acc_without_reasoning'] is not None else "   N/A"

        print(f"{model_name:<35} | {stats['n_total']:>6} | {stats['reasoning_rate']*100:>7.1f}% | {acc_with_str:>11} | {acc_without_str:>11}")

    print("=" * 90)

    # Compute scenario stats
    scenario_stats = compute_scenario_stats(all_records)

    # Sort by reasoning rate (descending) and show top scenarios
    sorted_scenarios = sorted(scenario_stats.items(),
                              key=lambda x: x[1]['reasoning_rate'],
                              reverse=True)

    print(f"\n{'Scenario|Extra':<15} | {'N':>5} | {'Reason%':>8} | {'Acc when used':>13}")
    print("-" * 55)

    for key, stats in sorted_scenarios[:20]:  # Show top 20
        scenario_id, extra = key.split('|')
        acc_str = f"{stats['acc_when_reasoning']*100:6.1f}%" if stats['acc_when_reasoning'] is not None else "   N/A"
        print(f"{scenario_id:>7}|{extra:<6} | {stats['n_total']:>5} | {stats['reasoning_rate']*100:>7.1f}% | {acc_str:>13}")

    print("-" * 55)

    # Statistical significance test (pooled across all models)
    # Compare: turns with reasoning vs turns without reasoning
    with_reasoning = [r for r in all_records if r.get('has_reasoning')]
    without_reasoning = [r for r in all_records if not r.get('has_reasoning')]

    n_optimal_with = sum(1 for r in with_reasoning if r.get('was_optimal'))
    n_not_optimal_with = len(with_reasoning) - n_optimal_with
    n_optimal_without = sum(1 for r in without_reasoning if r.get('was_optimal'))
    n_not_optimal_without = len(without_reasoning) - n_optimal_without

    print(f"\n=== Statistical Comparison (pooled across all models) ===")
    print(f"Turns WITH reasoning:    {len(with_reasoning):>4} turns, {n_optimal_with:>4} optimal ({n_optimal_with/len(with_reasoning)*100:.1f}%)" if with_reasoning else "")
    print(f"Turns WITHOUT reasoning: {len(without_reasoning):>4} turns, {n_optimal_without:>4} optimal ({n_optimal_without/len(without_reasoning)*100:.1f}%)" if without_reasoning else "")

    if with_reasoning and without_reasoning:
        acc_with = n_optimal_with / len(with_reasoning)
        acc_without = n_optimal_without / len(without_reasoning)
        diff = acc_with - acc_without

        p_value = fisher_exact_test(n_optimal_with, n_not_optimal_with,
                                    n_optimal_without, n_not_optimal_without)

        print(f"\nDifference: {diff*100:+.1f}%")
        if p_value is not None:
            sig = "*" if p_value < 0.05 else ""
            print(f"Fisher's exact test p-value: {p_value:.4f} {sig}")
        else:
            print("(scipy not available for Fisher's exact test)")

    print("\n" + "=" * 90)


if __name__ == '__main__':
    main()
