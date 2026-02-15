#!/usr/bin/env python3
"""Analyze response consistency across different reps for the same model.

Compares model responses to the same scenarios across different reps to measure:
- Action consistency (same action type chosen)
- Argument consistency (same arguments when Tell/Ask)
- Accuracy variance across reps

Usage:
    python analyze_rep_consistency.py file1.json file2.json [file3.json ...]
    python analyze_rep_consistency.py --model MODEL_NAME  # finds all files for model
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import List, Dict, Tuple


def load_records(filepath: str) -> Tuple[List[dict], str]:
    """Load turn records from a game data JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    records = data.get('turn_records', [])
    # Extract rep from records (use first non-None rep found)
    rep = None
    for r in records:
        if r.get('rep') is not None:
            rep = r['rep']
            break
    return records, rep


def parse_action(action_str: str) -> dict:
    """Parse action string into components."""
    if not action_str:
        return {'type': None, 'player': None, 'container': None, 'contents': None}

    action_str = action_str.strip()

    if action_str.lower() == 'pass':
        return {'type': 'Pass', 'player': None, 'container': None, 'contents': None}

    # Ask(Player, Container)
    ask_match = re.match(r'Ask\s*\(\s*(\w+)\s*,\s*([^)]+)\s*\)', action_str, re.IGNORECASE)
    if ask_match:
        return {
            'type': 'Ask',
            'player': ask_match.group(1).strip(),
            'container': ask_match.group(2).strip(),
            'contents': None
        }

    # Tell(Player, Container, Contents)
    tell_match = re.match(r'Tell\s*\(\s*(\w+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)', action_str, re.IGNORECASE)
    if tell_match:
        return {
            'type': 'Tell',
            'player': tell_match.group(1).strip(),
            'container': tell_match.group(2).strip(),
            'contents': tell_match.group(3).strip()
        }

    return {'type': action_str, 'player': None, 'container': None, 'contents': None}


def find_model_files(logs_dir: str, model_name: str) -> List[str]:
    """Find all game data files for a given model."""
    files = []
    for f in os.listdir(logs_dir):
        if f.endswith('_game_data.json') and model_name in f:
            files.append(os.path.join(logs_dir, f))
    return sorted(files)


def analyze_consistency(file_records: Dict[str, List[dict]]) -> dict:
    """Analyze consistency across multiple files/reps.

    Args:
        file_records: Dict mapping filename -> list of records

    Returns:
        Dict with consistency analysis results
    """
    # Group records by scenario_id + extra
    scenario_responses = defaultdict(list)  # (scenario_id, extra) -> [(file, record), ...]

    for filename, records in file_records.items():
        for record in records:
            key = (record.get('scenario_id'), record.get('extra'))
            scenario_responses[key].append((filename, record))

    results = {
        'total_scenarios': len(scenario_responses),
        'scenarios_with_multiple_reps': 0,
        'action_type_consistent': 0,
        'action_fully_consistent': 0,  # type + all args match
        'accuracy_consistent': 0,  # was_optimal same across reps
        'inconsistent_scenarios': [],
        'by_extra': defaultdict(lambda: {'total': 0, 'type_consistent': 0, 'fully_consistent': 0}),
        'by_scenario_id': defaultdict(lambda: {'total': 0, 'type_consistent': 0, 'fully_consistent': 0}),
    }

    for (scenario_id, extra), responses in scenario_responses.items():
        if len(responses) < 2:
            continue  # Need at least 2 reps to compare

        results['scenarios_with_multiple_reps'] += 1
        results['by_extra'][extra]['total'] += 1
        results['by_scenario_id'][scenario_id]['total'] += 1

        # Parse all actions
        parsed_actions = []
        for filename, record in responses:
            parsed = parse_action(record.get('action', ''))
            parsed['was_optimal'] = record.get('was_optimal')
            parsed['rep'] = record.get('rep')
            parsed['filename'] = os.path.basename(filename)
            parsed_actions.append(parsed)

        # Check consistency
        action_types = set(p['type'] for p in parsed_actions)
        type_consistent = len(action_types) == 1

        # Full consistency: type, player, container, contents all match
        action_tuples = set((p['type'], p['player'], p['container'], p['contents']) for p in parsed_actions)
        fully_consistent = len(action_tuples) == 1

        # Accuracy consistency
        was_optimal_vals = set(p['was_optimal'] for p in parsed_actions)
        accuracy_consistent = len(was_optimal_vals) == 1

        if type_consistent:
            results['action_type_consistent'] += 1
            results['by_extra'][extra]['type_consistent'] += 1
            results['by_scenario_id'][scenario_id]['type_consistent'] += 1

        if fully_consistent:
            results['action_fully_consistent'] += 1
            results['by_extra'][extra]['fully_consistent'] += 1
            results['by_scenario_id'][scenario_id]['fully_consistent'] += 1

        if accuracy_consistent:
            results['accuracy_consistent'] += 1

        if not type_consistent:
            results['inconsistent_scenarios'].append({
                'scenario_id': scenario_id,
                'extra': extra,
                'actions': parsed_actions,
                'optimal': responses[0][1].get('optimal_action'),
            })

    return results


def format_results(results: dict, files: List[str]) -> str:
    """Format analysis results as text."""
    lines = []
    lines.append("=" * 80)
    lines.append("REP CONSISTENCY ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Files analyzed:")
    for f in files:
        lines.append(f"  - {os.path.basename(f)}")
    lines.append("")

    n = results['scenarios_with_multiple_reps']
    if n == 0:
        lines.append("No scenarios with multiple reps found. Need at least 2 files with overlapping scenarios.")
        return "\n".join(lines)

    lines.append(f"Scenarios compared: {n}")
    lines.append("")

    # Overall consistency
    lines.append("OVERALL CONSISTENCY")
    lines.append("-" * 40)
    type_pct = 100 * results['action_type_consistent'] / n
    full_pct = 100 * results['action_fully_consistent'] / n
    acc_pct = 100 * results['accuracy_consistent'] / n
    lines.append(f"Action type consistent: {results['action_type_consistent']}/{n} ({type_pct:.1f}%)")
    lines.append(f"Action fully consistent: {results['action_fully_consistent']}/{n} ({full_pct:.1f}%)")
    lines.append(f"Accuracy consistent:     {results['accuracy_consistent']}/{n} ({acc_pct:.1f}%)")
    lines.append("")

    # By Extra
    lines.append("BY EXTRA CATEGORY")
    lines.append("-" * 40)
    for extra in ['0A', '0B', '1A', '1B']:
        stats = results['by_extra'].get(extra, {'total': 0, 'type_consistent': 0})
        if stats['total'] > 0:
            pct = 100 * stats['type_consistent'] / stats['total']
            lines.append(f"  {extra}: {stats['type_consistent']}/{stats['total']} type consistent ({pct:.1f}%)")
    lines.append("")

    # By Scenario ID
    lines.append("BY SCENARIO ID (action type consistency)")
    lines.append("-" * 40)
    inconsistent_ids = []
    for sid in sorted(results['by_scenario_id'].keys()):
        stats = results['by_scenario_id'][sid]
        if stats['total'] > 0:
            pct = 100 * stats['type_consistent'] / stats['total']
            if pct < 100:
                inconsistent_ids.append((sid, stats['type_consistent'], stats['total'], pct))

    if inconsistent_ids:
        lines.append("Scenarios with inconsistent action types:")
        for sid, cons, total, pct in inconsistent_ids:
            lines.append(f"  ID {sid}: {cons}/{total} consistent ({pct:.1f}%)")
    else:
        lines.append("All scenarios had consistent action types across reps.")
    lines.append("")

    # Detailed inconsistencies
    if results['inconsistent_scenarios']:
        lines.append("DETAILED INCONSISTENCIES")
        lines.append("-" * 40)
        lines.append("")
        lines.append("ID  Extra  Optimal  |  Rep 1 Action    Rep 2 Action")
        lines.append("--- -----  -------  |  -------------   -------------")
        for item in results['inconsistent_scenarios'][:30]:
            actions = item['actions']
            # Sort by rep (None treated as 1)
            actions_sorted = sorted(actions, key=lambda a: a['rep'] or 1)
            act_strs = []
            for a in actions_sorted:
                opt_mark = "" if a['was_optimal'] else "*"
                act_strs.append(f"{a['type']}{opt_mark}")
            act_display = "   ".join(f"{s:<15}" for s in act_strs)
            lines.append(f"{item['scenario_id']:<3} {item['extra']:<5}  {item['optimal']:<7}  |  {act_display}")
        lines.append("")
        lines.append("* = incorrect action")
        if len(results['inconsistent_scenarios']) > 30:
            lines.append(f"... and {len(results['inconsistent_scenarios']) - 30} more")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Analyze response consistency across reps')
    parser.add_argument('files', nargs='*', help='Game data JSON files to compare')
    parser.add_argument('--model', type=str, help='Model name to find all files for')
    parser.add_argument('--logs-dir', type=str, default='tom_llm_logs', help='Logs directory')
    args = parser.parse_args()

    # Determine files to analyze
    files = args.files
    if args.model:
        files = find_model_files(args.logs_dir, args.model)
        if not files:
            print(f"No files found for model '{args.model}' in {args.logs_dir}")
            return

    if len(files) < 2:
        print("Need at least 2 files to compare. Provide file paths or use --model MODEL_NAME")
        return

    # Load all records
    file_records = {}
    for f in files:
        records, rep = load_records(f)
        file_records[f] = records

    # Analyze
    results = analyze_consistency(file_records)
    output = format_results(results, files)

    # Write to file
    output_file = os.path.join(args.logs_dir, 'rep_consistency.txt')
    with open(output_file, 'w') as f:
        f.write(output)

    # Brief summary to stdout
    n = results['scenarios_with_multiple_reps']
    if n > 0:
        type_pct = 100 * results['action_type_consistent'] / n
        print(f"Compared {len(files)} files, {n} scenarios")
        print(f"Action type consistency: {type_pct:.1f}%")
    print(f"Full results: {output_file}")


if __name__ == '__main__':
    main()
