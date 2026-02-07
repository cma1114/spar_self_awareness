#!/usr/bin/env python3
"""
Compare two runs (model + free_response pairings) to identify divergences.
Helps understand why COT might hurt or help performance.
"""

import json
import glob
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================
# Runs to compare: (model_name, free_response)
RUN_A = ('mistral-large-2512', True)   # mistral-large with COT
RUN_B = ('mistral-small-24b-instruct-2501', True)  # mistral-small with COT

# =============================================================================
# ToM Mastery Categories (for tagging divergences)
# =============================================================================
TOM_MASTERY_CATEGORIES = {
    'self_knowledge_belief': {
        'name': 'Self: Knowledge vs Belief',
        'scenarios': [7, 8, 9, 12, 13],
    },
    'teammate_knowledge_belief': {
        'name': 'Teammate: Knowledge vs Belief',
        'scenarios': [17, 18, 19, 20, 21, 22],
    },
    'combined_uncertainty': {
        'name': 'Combined Uncertainty',
        'scenarios': [10, 11, 23, 24],
    },
    'true_false_belief': {
        'name': 'True vs False Belief',
        'scenarios': [14, 15, 16, 17, 18, 19],
    },
    'teammate_opponent': {
        'name': 'Teammate vs Opponent',
        'scenarios': [12, 13, 17, 18, 19, 30, 31, 32, 37, 39],
    },
}

def get_mastery_categories(scenario_id: int) -> List[str]:
    """Get all mastery categories that include this scenario."""
    cats = []
    for key, info in TOM_MASTERY_CATEGORIES.items():
        if scenario_id in info['scenarios']:
            cats.append(info['name'])
    return cats if cats else ['Other']


# =============================================================================
# Helper functions (reused from analyze_results.py / analyze_errors.py)
# =============================================================================
def extract_model_name(filepath: str) -> str:
    """Extract model name from filename."""
    basename = os.path.basename(filepath)
    match = re.match(r'(.+?)_\d+_game_data\.json$', basename)
    if match:
        return match.group(1)
    return basename


def load_game_data(filepath: str) -> List[dict]:
    """Load records from a game_data.json file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'turn_records' in data:
        return data['turn_records']
    return data


def normalize_extra(val) -> str:
    """Convert legacy Extra values to new string format."""
    if val is None or val == 0: return '1A'
    if val == 1: return '1B'
    if val in ('0A', '0B', '1A', '1B'): return val
    return str(val)


def normalize_action(action: str) -> str:
    """Normalize action string to Pass/Ask/Tell (for short action strings only)."""
    if action is None:
        return 'Unknown'
    action = action.strip()
    if action == 'Pass' or action.lower() in ('pass', 'pass.'):
        return 'Pass'
    # Only check for Ask/Tell if it's a short action string (not COT reasoning)
    if len(action) < 100:
        if 'Ask' in action or action.lower().startswith('ask'):
            return 'Ask'
        if 'Tell' in action or action.lower().startswith('tell'):
            return 'Tell'
    return 'Pass'  # Direct answer counts as Pass


def get_action_from_record(record: dict) -> str:
    """Get normalized action from a record, using action_cost as ground truth for COT."""
    action_cost = record.get('action_cost', 0)
    action_str = record.get('action', '')

    # For free_response (COT) records, use action_cost to determine action type
    if record.get('free_response'):
        if action_cost == 0:
            return 'Pass'
        # Cost > 0 means Ask or Tell - need to parse from action string
        if 'Ask(' in action_str or 'Ask(B' in action_str or action_str.strip().startswith('Ask'):
            return 'Ask'
        if 'Tell(' in action_str or 'Tell(B' in action_str or action_str.strip().startswith('Tell'):
            return 'Tell'
        # Check for "Your action: Ask" or "Your action: Tell" patterns at end
        if 'Your action: Ask' in action_str or 'action: Ask' in action_str.lower():
            return 'Ask'
        if 'Your action: Tell' in action_str or 'action: Tell' in action_str.lower():
            return 'Tell'
        # Default to Ask if cost > 0 (since we can't tell)
        return 'Ask'
    else:
        return normalize_action(action_str)


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


# =============================================================================
# Core comparison logic
# =============================================================================
def load_run_data(model: str, free_response: bool, logs_dir: str) -> List[dict]:
    """Load all player A records for a specific model + free_response combo."""
    pattern = os.path.join(logs_dir, '*_game_data.json')
    files = glob.glob(pattern)

    all_records = []
    for filepath in files:
        if extract_model_name(filepath) != model:
            continue

        records = load_game_data(filepath)
        if not records:
            continue

        # Check free_response (run-level setting)
        file_fr = records[0].get('free_response')
        if file_fr != free_response:
            continue

        # Filter to player A
        player_a = [r for r in records if r.get('character') == 'A']
        # Filter out records where model hit token limit before giving action
        player_a, _ = filter_valid_records(player_a)
        all_records.extend(player_a)

    return all_records


def aggregate_by_scenario(records: List[dict]) -> Dict[Tuple[str, str], dict]:
    """
    Aggregate records by (scenario_id, extra).
    Returns stats per scenario including success rate and example records.
    """
    grouped = defaultdict(list)
    for r in records:
        key = (str(r.get('scenario_id')), normalize_extra(r.get('extra')))
        grouped[key].append(r)

    result = {}
    for key, recs in grouped.items():
        n_optimal = sum(1 for r in recs if r.get('was_optimal'))
        result[key] = {
            'n': len(recs),
            'n_optimal': n_optimal,
            'rate': n_optimal / len(recs) if recs else 0,
            'records': recs,
            'example': recs[0],  # Keep one example for details
        }
    return result


def build_action_confusion(records: List[dict]) -> Dict[str, Dict[str, int]]:
    """Build action confusion matrix: optimal_action -> {actual_action: count}."""
    confusion = defaultdict(lambda: defaultdict(int))
    for r in records:
        optimal = normalize_action(r.get('optimal_action', ''))
        actual = get_action_from_record(r)
        confusion[optimal][actual] += 1
    return {k: dict(v) for k, v in confusion.items()}


def compare_runs(agg_a: dict, agg_b: dict) -> dict:
    """Compare two aggregated run results."""
    common_keys = set(agg_a.keys()) & set(agg_b.keys())

    comparisons = []
    for key in common_keys:
        a = agg_a[key]
        b = agg_b[key]
        delta = b['rate'] - a['rate']

        comparisons.append({
            'key': key,
            'scenario_id': key[0],
            'extra': key[1],
            'a_rate': a['rate'],
            'b_rate': b['rate'],
            'a_n': a['n'],
            'b_n': b['n'],
            'delta': delta,
            'a_example': a['example'],
            'b_example': b['example'],
            'a_records': a['records'],
            'b_records': b['records'],
        })

    # Sort by delta (most negative first = biggest A advantages)
    comparisons.sort(key=lambda x: x['delta'])

    return {
        'comparisons': comparisons,
        'common_keys': common_keys,
        'a_only': set(agg_a.keys()) - set(agg_b.keys()),
        'b_only': set(agg_b.keys()) - set(agg_a.keys()),
    }


# =============================================================================
# Report generation
# =============================================================================
def generate_report(run_a_records: List[dict], run_b_records: List[dict],
                    run_a_name: str, run_b_name: str) -> str:
    """Generate comprehensive comparison report."""
    lines = []

    # Aggregate by scenario
    agg_a = aggregate_by_scenario(run_a_records)
    agg_b = aggregate_by_scenario(run_b_records)

    # Compare
    comparison = compare_runs(agg_a, agg_b)
    comparisons = comparison['comparisons']

    # Overall stats
    a_total = len(run_a_records)
    b_total = len(run_b_records)
    a_correct = sum(1 for r in run_a_records if r.get('was_optimal'))
    b_correct = sum(1 for r in run_b_records if r.get('was_optimal'))
    a_rate = a_correct / a_total if a_total else 0
    b_rate = b_correct / b_total if b_total else 0

    lines.append("=" * 80)
    lines.append(f"RUN COMPARISON: {run_a_name} vs {run_b_name}")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'':30} {'Run A':>15} {'Run B':>15}")
    lines.append(f"{'Overall accuracy:':<30} {a_rate*100:>14.1f}% {b_rate*100:>14.1f}%")
    lines.append(f"{'Total records:':<30} {a_total:>15} {b_total:>15}")
    lines.append(f"{'Scenarios with data:':<30} {len(agg_a):>15} {len(agg_b):>15}")
    lines.append(f"{'Common scenarios:':<30} {len(comparison['common_keys']):>15}")
    lines.append("")

    # Categorize divergences
    a_advantages = [c for c in comparisons if c['delta'] < -0.1]  # A better by >10%
    b_advantages = [c for c in comparisons if c['delta'] > 0.1]  # B better by >10%
    similar = [c for c in comparisons if abs(c['delta']) <= 0.1]

    lines.append("DIVERGENCE SUMMARY:")
    lines.append(f"  A advantages (A better by >10%): {len(a_advantages)}")
    lines.append(f"  B advantages (B better by >10%): {len(b_advantages)}")
    lines.append(f"  Similar (within 10%):            {len(similar)}")
    lines.append("")

    # Breakdown by Extra category
    lines.append("DIVERGENCE BY EXTRA CATEGORY:")
    lines.append(f"  {'Extra':<6} | {'A rate':>8} | {'B rate':>8} | {'Delta':>8} | {'A wins':>8} | {'B wins':>8}")
    lines.append("  " + "-" * 60)

    for extra in ['0A', '0B', '1A', '1B']:
        extra_comps = [c for c in comparisons if c['extra'] == extra]
        if not extra_comps:
            continue
        avg_a = sum(c['a_rate'] for c in extra_comps) / len(extra_comps)
        avg_b = sum(c['b_rate'] for c in extra_comps) / len(extra_comps)
        delta = avg_b - avg_a
        n_a_wins = len([c for c in extra_comps if c['delta'] < -0.1])
        n_b_wins = len([c for c in extra_comps if c['delta'] > 0.1])
        lines.append(f"  {extra:<6} | {avg_a*100:>7.1f}% | {avg_b*100:>7.1f}% | {delta*100:>+7.1f}% | {n_a_wins:>8} | {n_b_wins:>8}")
    lines.append("")

    # Action distribution analysis
    lines.append("=" * 80)
    lines.append("ACTION DISTRIBUTION ANALYSIS")
    lines.append("=" * 80)

    # Count actual actions taken
    def count_actions(records):
        counts = {'Pass': 0, 'Ask': 0, 'Tell': 0}
        for r in records:
            action = get_action_from_record(r)
            if action in counts:
                counts[action] += 1
        return counts

    # Count optimal actions
    def count_optimal(records):
        counts = {'Pass': 0, 'Ask': 0, 'Tell': 0}
        for r in records:
            optimal = normalize_action(r.get('optimal_action', ''))
            if optimal in counts:
                counts[optimal] += 1
        return counts

    actions_a = count_actions(run_a_records)
    actions_b = count_actions(run_b_records)
    optimal_a = count_optimal(run_a_records)
    optimal_b = count_optimal(run_b_records)

    lines.append("\nACTIONS TAKEN (model behavior):")
    lines.append(f"  {'Action':<10} {'Run A':>12} {'Run A %':>10} {'Run B':>12} {'Run B %':>10}")
    lines.append("  " + "-" * 56)
    for action in ['Pass', 'Ask', 'Tell']:
        a_ct = actions_a.get(action, 0)
        b_ct = actions_b.get(action, 0)
        a_pct = a_ct / a_total * 100 if a_total else 0
        b_pct = b_ct / b_total * 100 if b_total else 0
        lines.append(f"  {action:<10} {a_ct:>12} {a_pct:>9.1f}% {b_ct:>12} {b_pct:>9.1f}%")

    lines.append("\nOPTIMAL ACTIONS (what should happen):")
    lines.append(f"  {'Action':<10} {'Run A':>12} {'Run A %':>10} {'Run B':>12} {'Run B %':>10}")
    lines.append("  " + "-" * 56)
    for action in ['Pass', 'Ask', 'Tell']:
        a_ct = optimal_a.get(action, 0)
        b_ct = optimal_b.get(action, 0)
        a_pct = a_ct / a_total * 100 if a_total else 0
        b_pct = b_ct / b_total * 100 if b_total else 0
        lines.append(f"  {action:<10} {a_ct:>12} {a_pct:>9.1f}% {b_ct:>12} {b_pct:>9.1f}%")

    # Compute per-optimal-action success rates
    conf_a = build_action_confusion(run_a_records)
    conf_b = build_action_confusion(run_b_records)

    lines.append("\nPER-OPTIMAL-ACTION SUCCESS RATES:")
    lines.append(f"  {'Optimal':<10} {'Run A hits':>12} {'Run A %':>10} {'Run B hits':>12} {'Run B %':>10}")
    lines.append("  " + "-" * 56)
    for optimal in ['Pass', 'Ask', 'Tell']:
        a_counts = conf_a.get(optimal, {})
        b_counts = conf_b.get(optimal, {})
        a_opt_total = sum(a_counts.values())
        b_opt_total = sum(b_counts.values())
        a_hits = a_counts.get(optimal, 0)
        b_hits = b_counts.get(optimal, 0)
        a_rate = a_hits / a_opt_total * 100 if a_opt_total else 0
        b_rate = b_hits / b_opt_total * 100 if b_opt_total else 0
        lines.append(f"  {optimal:<10} {a_hits:>8}/{a_opt_total:<4} {a_rate:>9.1f}% {b_hits:>8}/{b_opt_total:<4} {b_rate:>9.1f}%")

    # Math verification
    lines.append("\nMATH VERIFICATION (expected vs actual accuracy):")
    lines.append("  Formula: sum over each optimal action of (proportion × hit_rate)")
    def compute_expected_detailed(conf, total):
        contributions = {}
        expected = 0
        for optimal in ['Pass', 'Ask', 'Tell']:
            counts = conf.get(optimal, {})
            opt_total = sum(counts.values())
            hits = counts.get(optimal, 0)
            if total > 0 and opt_total > 0:
                opt_proportion = opt_total / total
                hit_rate = hits / opt_total
                contribution = opt_proportion * hit_rate
                contributions[optimal] = (opt_proportion, hit_rate, contribution)
                expected += contribution
            else:
                contributions[optimal] = (0, 0, 0)
        return expected, contributions

    expected_a, contrib_a = compute_expected_detailed(conf_a, a_total)
    expected_b, contrib_b = compute_expected_detailed(conf_b, b_total)

    # Use different names to avoid shadowing
    actual_acc_a = a_correct / a_total if a_total else 0
    actual_acc_b = b_correct / b_total if b_total else 0

    lines.append(f"\n  Run A breakdown:")
    for opt in ['Pass', 'Ask', 'Tell']:
        prop, hit_rt, contrib = contrib_a[opt]
        lines.append(f"    {opt}: {prop*100:.1f}% of scenarios × {hit_rt*100:.1f}% hit rate = {contrib*100:.1f}%")
    lines.append(f"    Total expected: {expected_a*100:.1f}% | Actual: {actual_acc_a*100:.1f}%")

    lines.append(f"\n  Run B breakdown:")
    for opt in ['Pass', 'Ask', 'Tell']:
        prop, hit_rt, contrib = contrib_b[opt]
        lines.append(f"    {opt}: {prop*100:.1f}% of scenarios × {hit_rt*100:.1f}% hit rate = {contrib*100:.1f}%")
    lines.append(f"    Total expected: {expected_b*100:.1f}% | Actual: {actual_acc_b*100:.1f}%")

    if abs(expected_b - actual_acc_b) > 0.05:
        lines.append(f"\n  *** DISCREPANCY DETECTED: Expected {expected_b*100:.1f}% but got {actual_acc_b*100:.1f}% ***")
        lines.append(f"  Investigating records where was_optimal=True but action doesn't match optimal_action...")

        # Find cases where was_optimal=True but action != optimal_action
        mismatches = []
        for r in run_b_records:
            if r.get('was_optimal'):
                actual = get_action_from_record(r)
                optimal = normalize_action(r.get('optimal_action', ''))
                if actual != optimal:
                    mismatches.append({
                        'scenario_id': r.get('scenario_id'),
                        'extra': normalize_extra(r.get('extra')),
                        'action': r.get('action'),
                        'actual_parsed': actual,
                        'optimal_action': r.get('optimal_action'),
                        'action_cost': r.get('action_cost'),
                        'was_optimal': r.get('was_optimal'),
                        'ks_self': r.get('ks_self'),
                    })

        lines.append(f"  Found {len(mismatches)} records where was_optimal=True but action type doesn't match optimal type")

        if mismatches:
            # Group by action/optimal pair
            mismatch_pairs = defaultdict(int)
            for m in mismatches:
                pair = (m['actual_parsed'], normalize_action(m['optimal_action']))
                mismatch_pairs[pair] += 1

            lines.append(f"\n  Mismatch patterns (actual → optimal):")
            for (actual, optimal), count in sorted(mismatch_pairs.items(), key=lambda x: -x[1]):
                lines.append(f"    {actual} when optimal={optimal}: {count} cases")

            lines.append(f"\n  Sample mismatches (showing action_cost to verify parsing):")
            for m in mismatches[:5]:
                lines.append(f"    Scenario {m['scenario_id']} ({m['extra']}): parsed={m['actual_parsed']}, cost={m['action_cost']}, optimal={m['optimal_action']}")
    lines.append("")

    # Action confusion matrices
    lines.append("=" * 80)
    lines.append("ACTION CONFUSION MATRICES (detailed)")
    lines.append("=" * 80)

    for optimal in ['Pass', 'Ask', 'Tell']:
        if optimal not in conf_a and optimal not in conf_b:
            continue

        lines.append(f"\nWhen optimal={optimal}:")
        lines.append(f"  {'Action':<10} {'Run A':>12} {'Run B':>12} {'Delta':>10}")
        lines.append("  " + "-" * 45)

        a_counts = conf_a.get(optimal, {})
        b_counts = conf_b.get(optimal, {})
        a_total = sum(a_counts.values()) or 1
        b_total = sum(b_counts.values()) or 1

        for action in ['Pass', 'Ask', 'Tell']:
            a_pct = a_counts.get(action, 0) / a_total * 100
            b_pct = b_counts.get(action, 0) / b_total * 100
            delta = b_pct - a_pct
            marker = " <--" if abs(delta) > 10 else ""
            lines.append(f"  {action:<10} {a_pct:>11.1f}% {b_pct:>11.1f}% {delta:>+9.1f}%{marker}")
    lines.append("")

    # Regression analysis (B worse than A)
    lines.append("=" * 80)
    lines.append("RUN A ADVANTAGES (where Run A outperformed Run B)")
    lines.append("=" * 80)

    if a_advantages:
        # Group by ToM category
        cat_a_advantages = defaultdict(list)
        for r in a_advantages:
            sid = int(r['scenario_id'])
            for cat in get_mastery_categories(sid):
                cat_a_advantages[cat].append(r)

        lines.append("\nBy ToM Mastery Category:")
        for cat, regs in sorted(cat_a_advantages.items(), key=lambda x: -len(x[1])):
            lines.append(f"  {cat}: {len(regs)} scenarios")

        lines.append("\nTop A advantages:")
        for i, r in enumerate(a_advantages[:10], 1):
            sid = r['scenario_id']
            extra = r['extra']
            cats = get_mastery_categories(int(sid))
            ex_a = r['a_example']
            ex_b = r['b_example']

            lines.append(f"\n  {i}. Scenario {sid} (Extra={extra}) - Delta: {r['delta']*100:+.1f}%")
            lines.append(f"     Categories: {', '.join(cats)}")
            lines.append(f"     A rate: {r['a_rate']*100:.0f}% ({r['a_n']} trials) | B rate: {r['b_rate']*100:.0f}% ({r['b_n']} trials)")
            lines.append(f"     Epistemic: self={ex_a.get('ks_self')}, tm={ex_a.get('ks_teammate')}, opp={ex_a.get('ks_opponent')}")
            lines.append(f"     Optimal: {ex_a.get('optimal_action')}")
            lines.append(f"     A action: {ex_a.get('action')} | B action: {ex_b.get('action')}")

            # Check for "correct answer, wrong action"
            b_answer_correct = ex_b.get('answer_correct')
            b_was_optimal = ex_b.get('was_optimal')
            if b_answer_correct and not b_was_optimal:
                lines.append(f"     *** Model knew answer but chose wrong action! ***")
    else:
        lines.append("\n  No scenarios where A outperformed B by >10%.")
    lines.append("")

    # Improvement analysis
    lines.append("=" * 80)
    lines.append("RUN B ADVANTAGES (where Run B outperformed Run A)")
    lines.append("=" * 80)

    if b_advantages:
        cat_b_advantages = defaultdict(list)
        for r in b_advantages:
            sid = int(r['scenario_id'])
            for cat in get_mastery_categories(sid):
                cat_b_advantages[cat].append(r)

        lines.append("\nBy ToM Mastery Category:")
        for cat, imps in sorted(cat_b_advantages.items(), key=lambda x: -len(x[1])):
            lines.append(f"  {cat}: {len(imps)} scenarios")

        lines.append("\nTop B advantages:")
        for i, r in enumerate(sorted(b_advantages, key=lambda x: -x['delta'])[:10], 1):
            sid = r['scenario_id']
            extra = r['extra']
            cats = get_mastery_categories(int(sid))
            ex_a = r['a_example']
            ex_b = r['b_example']

            lines.append(f"\n  {i}. Scenario {sid} (Extra={extra}) - Delta: {r['delta']*100:+.1f}%")
            lines.append(f"     Categories: {', '.join(cats)}")
            lines.append(f"     A rate: {r['a_rate']*100:.0f}% ({r['a_n']} trials) | B rate: {r['b_rate']*100:.0f}% ({r['b_n']} trials)")
            lines.append(f"     Epistemic: self={ex_a.get('ks_self')}, tm={ex_a.get('ks_teammate')}, opp={ex_a.get('ks_opponent')}")
            lines.append(f"     Optimal: {ex_a.get('optimal_action')}")
    else:
        lines.append("\n  No scenarios where B outperformed A by >10%.")
    lines.append("")

    # Pattern summary
    lines.append("=" * 80)
    lines.append("PATTERN SUMMARY")
    lines.append("=" * 80)

    # Action confusion patterns where A outperformed B
    if a_advantages:
        lines.append("\nAction patterns where B failed (A outperformed):")
        reg_confusion = defaultdict(int)
        for r in a_advantages:
            for rec in r['b_records']:
                if not rec.get('was_optimal'):
                    optimal = normalize_action(rec.get('optimal_action', ''))
                    actual = get_action_from_record(rec)
                    if optimal != actual:
                        reg_confusion[f"{optimal}→{actual}"] += 1

        for pattern, count in sorted(reg_confusion.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"  {pattern}: {count} cases")

        # "Correct answer, wrong action" count
        correct_answer_wrong_action = 0
        for r in a_advantages:
            for rec in r['b_records']:
                if rec.get('answer_correct') and not rec.get('was_optimal'):
                    correct_answer_wrong_action += 1

        if correct_answer_wrong_action:
            lines.append(f"\n  'Correct answer, wrong action' (where A beat B): {correct_answer_wrong_action}")
            lines.append("  (Model knew the answer but chose Tell/Ask instead of Pass)")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, 'tom_llm_logs')

    model_a, fr_a = RUN_A
    model_b, fr_b = RUN_B

    run_a_name = f"{model_a}" + (" (with COT)" if fr_a else "")
    run_b_name = f"{model_b}" + (" (with COT)" if fr_b else "")

    print(f"Loading Run A: {run_a_name}...")
    run_a_records = load_run_data(model_a, fr_a, logs_dir)
    print(f"  Found {len(run_a_records)} records")

    print(f"Loading Run B: {run_b_name}...")
    run_b_records = load_run_data(model_b, fr_b, logs_dir)
    print(f"  Found {len(run_b_records)} records")

    if not run_a_records:
        print(f"Error: No records found for Run A ({run_a_name})")
        return
    if not run_b_records:
        print(f"Error: No records found for Run B ({run_b_name})")
        return

    print("\nGenerating comparison report...")
    report = generate_report(run_a_records, run_b_records, run_a_name, run_b_name)

    # Save full report to file
    output_filename = f"comparison_{model_a.replace('-', '_')}_vs_{model_b.replace('-', '_')}_cot.txt"
    output_path = os.path.join(logs_dir, output_filename)
    with open(output_path, 'w') as f:
        f.write(report)

    # Print concise summary to console
    a_correct = sum(1 for r in run_a_records if r.get('was_optimal'))
    b_correct = sum(1 for r in run_b_records if r.get('was_optimal'))
    a_rate = a_correct / len(run_a_records) * 100
    b_rate = b_correct / len(run_b_records) * 100

    conf_a = build_action_confusion(run_a_records)
    conf_b = build_action_confusion(run_b_records)

    print(f"\n{'='*60}")
    print(f"COMPARISON: {run_a_name} vs {run_b_name}")
    print(f"{'='*60}")
    print(f"Overall: {a_rate:.1f}% vs {b_rate:.1f}% ({b_rate - a_rate:+.1f}%)")
    print(f"Records: {len(run_a_records)} vs {len(run_b_records)}")

    print(f"\nAction distributions:")
    for action in ['Pass', 'Ask', 'Tell']:
        a_ct = sum(1 for r in run_a_records if get_action_from_record(r) == action)
        b_ct = sum(1 for r in run_b_records if get_action_from_record(r) == action)
        a_pct = a_ct / len(run_a_records) * 100
        b_pct = b_ct / len(run_b_records) * 100
        print(f"  {action}: {a_pct:.1f}% vs {b_pct:.1f}%")

    print(f"\nPer-optimal success rates:")
    for opt in ['Pass', 'Ask', 'Tell']:
        a_counts = conf_a.get(opt, {})
        b_counts = conf_b.get(opt, {})
        a_total = sum(a_counts.values())
        b_total = sum(b_counts.values())
        a_hits = a_counts.get(opt, 0)
        b_hits = b_counts.get(opt, 0)
        a_pct = a_hits / a_total * 100 if a_total else 0
        b_pct = b_hits / b_total * 100 if b_total else 0
        print(f"  {opt}: {a_pct:.1f}% vs {b_pct:.1f}%")

    print(f"\nFull report saved to: {output_path}")


if __name__ == '__main__':
    main()
