#!/usr/bin/env python3
"""Analyze self-reference patterns in reasoning traces.

Examines whether models refer to themselves as "I" vs "A" (player character)
and correlates with accuracy.

Usage:
    python analyze_self_reference.py [--logs-dir tom_llm_logs]
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


def load_game_data(log_path: str) -> Dict[Tuple[int, str], bool]:
    """Load corresponding game_data.json for a log file.

    Returns lookup dict: (scenario_id, extra) -> was_optimal
    """
    json_path = log_path.replace('.log', '_game_data.json')
    if not os.path.exists(json_path):
        return {}

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Build lookup: (scenario_id, extra) -> was_optimal
    # Note: scenario_id in JSON may be string, convert to int for consistency
    lookup = {}
    for r in data.get('turn_records', []):
        sid = r.get('scenario_id')
        if sid is not None:
            sid = int(sid) if isinstance(sid, str) else sid
        key = (sid, r.get('extra'))
        lookup[key] = r.get('was_optimal', False)
    return lookup


def parse_log_file(filepath: str, game_data_lookup: Optional[Dict] = None) -> List[dict]:
    """Parse a log file to extract reasoning traces and outcomes.

    Args:
        filepath: Path to log file
        game_data_lookup: Optional dict mapping (scenario_id, extra) -> was_optimal
                          If provided, uses was_optimal; otherwise falls back to answer correctness

    Returns list of dicts with: reasoning_trace, action, was_correct, scenario_id, extra
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    results = []

    # Split by trial markers
    # Handle both formats: "(Rep N, Spec ID M)" and "(Spec ID M)"
    trial_pattern = r'--- Running Trial (\d+)/\d+ \((?:Rep \d+, )?Spec ID (\d+)\): \{([^}]+)\}'
    trials = re.split(r'(?=--- Running Trial)', content)

    for trial in trials:
        if not trial.strip() or 'REASONING TRACE:' not in trial:
            continue

        # Extract trial info
        trial_match = re.search(trial_pattern, trial)
        if not trial_match:
            continue

        # Parse spec dict for scenario info
        spec_str = trial_match.group(3)
        scenario_id = None
        extra = None
        id_match = re.search(r"'Id':\s*'(\d+)'", spec_str)
        if id_match:
            scenario_id = int(id_match.group(1))
        extra_match = re.search(r"'Extra':\s*'(\w+)'", spec_str)
        if extra_match:
            extra = extra_match.group(1)

        # Extract reasoning trace
        trace_match = re.search(r'REASONING TRACE:\s*\n(.*?)(?=\nAction:|$)', trial, re.DOTALL)
        if not trace_match:
            continue
        reasoning = trace_match.group(1).strip()

        # Extract action
        action_match = re.search(r'\nAction:\s*(\w+)', trial)
        action = action_match.group(1) if action_match else None

        # Check if action was optimal (preferred) or fall back to answer correctness
        if game_data_lookup and (scenario_id, extra) in game_data_lookup:
            was_correct = game_data_lookup[(scenario_id, extra)]
        else:
            # Fallback: use answer correctness from log
            was_correct = 'Correct!' in trial

        results.append({
            'reasoning': reasoning,
            'action': action,
            'was_correct': was_correct,
            'scenario_id': scenario_id,
            'extra': extra,
        })

    return results


def analyze_self_reference(reasoning: str) -> dict:
    """Analyze how the model refers to itself in reasoning.

    Returns dict with counts and patterns.
    """
    # Patterns for first-person "I" references
    # Match "I" as standalone word, including contractions
    i_patterns = [
        r"\bI\b(?!')",  # I (not followed by apostrophe to avoid matching contractions separately)
        r"\bI'm\b",
        r"\bI'll\b",
        r"\bI've\b",
        r"\bI'd\b",
        r"\bmy\b",
        r"\bme\b",
        r"\bmyself\b",
    ]

    # Patterns for third-person "A" references (player character)
    a_patterns = [
        r"\bA\s+(?:know|see|saw|leave|enter|put|move|am|is|was|will|can|should|must|need)",
        r"\bA\s*\(",  # A( as in function call context
        r"\bplayer\s+A\b",
        r"\bA\s+answers\b",
        r"A \(me\)",
        r"\bas A\b",
        r"I\s*\(A\)",  # "I (A)" or "I(A)" - explicit self-as-character reference
    ]

    i_count = 0
    a_count = 0

    for pattern in i_patterns:
        i_count += len(re.findall(pattern, reasoning, re.IGNORECASE))

    for pattern in a_patterns:
        a_count += len(re.findall(pattern, reasoning, re.IGNORECASE))

    # Classify dominant style
    if i_count > 0 and a_count == 0:
        style = 'I_only'
    elif a_count > 0 and i_count == 0:
        style = 'A_only'
    elif i_count > 0 and a_count > 0:
        style = 'mixed'
    else:
        style = 'neither'

    return {
        'i_count': i_count,
        'a_count': a_count,
        'style': style,
        'total_refs': i_count + a_count,
    }


def format_results(model_stats: Dict[str, dict]) -> str:
    """Format analysis results."""
    lines = []
    lines.append("=" * 80)
    lines.append("SELF-REFERENCE ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    for model, stats in sorted(model_stats.items()):
        lines.append(f"Model: {model}")
        lines.append("-" * 60)

        total = stats['total']
        if total == 0:
            lines.append("  No reasoning traces found")
            continue

        lines.append(f"  Total traces: {total}")
        lines.append("")

        # Style distribution
        lines.append("  Reference Style Distribution:")
        for style in ['I_only', 'A_only', 'mixed', 'neither']:
            count = stats['by_style'].get(style, {}).get('count', 0)
            pct = 100 * count / total if total > 0 else 0
            lines.append(f"    {style:<10}: {count:>4} ({pct:>5.1f}%)")
        lines.append("")

        # Accuracy by style
        lines.append("  Accuracy by Reference Style:")
        lines.append("    Style       Count   Correct   Accuracy")
        lines.append("    ----------  ------  --------  --------")
        for style in ['I_only', 'A_only', 'mixed', 'neither']:
            style_stats = stats['by_style'].get(style, {})
            count = style_stats.get('count', 0)
            correct = style_stats.get('correct', 0)
            acc = 100 * correct / count if count > 0 else 0
            lines.append(f"    {style:<10}  {count:>6}  {correct:>8}  {acc:>7.1f}%")
        lines.append("")

        # Average reference counts
        lines.append("  Average References per Trace:")
        avg_i = stats['total_i'] / total if total > 0 else 0
        avg_a = stats['total_a'] / total if total > 0 else 0
        lines.append(f"    'I' references: {avg_i:.1f}")
        lines.append(f"    'A' references: {avg_a:.1f}")
        lines.append("")

        # By Extra category
        lines.append("  By Extra Category:")
        lines.append("    Extra   I_only          Mixed           Total")
        lines.append("    -----   -------------   -------------   -----")
        for extra in ['0A', '0B', '1A', '1B']:
            extra_stats = stats['by_extra'].get(extra, {})
            i_only = extra_stats.get('I_only', {'count': 0, 'correct': 0})
            mixed = extra_stats.get('mixed', {'count': 0, 'correct': 0})
            total_extra = sum(extra_stats.get(s, {}).get('count', 0) for s in ['I_only', 'A_only', 'mixed', 'neither'])

            i_str = f"{i_only['correct']}/{i_only['count']}" if i_only['count'] > 0 else "-"
            m_str = f"{mixed['correct']}/{mixed['count']}" if mixed['count'] > 0 else "-"
            lines.append(f"    {extra:<5}   {i_str:<13}   {m_str:<13}   {total_extra}")
        lines.append("")

        # By Scenario ID - show scenarios with mixed references
        lines.append("  Scenarios with Mixed References (I + A):")
        lines.append("    ID    Mixed (corr/tot)   I_only (corr/tot)   Diff")
        lines.append("    ----  -----------------  ------------------  -----")

        scenario_rows = []
        for sid in sorted(stats['by_scenario'].keys()):
            sc_stats = stats['by_scenario'][sid]
            mixed = sc_stats.get('mixed', {'count': 0, 'correct': 0})
            i_only = sc_stats.get('I_only', {'count': 0, 'correct': 0})

            if mixed['count'] > 0:
                m_acc = 100 * mixed['correct'] / mixed['count']
                i_acc = 100 * i_only['correct'] / i_only['count'] if i_only['count'] > 0 else None
                diff = f"{m_acc - i_acc:+.0f}%" if i_acc is not None else "n/a"
                m_str = f"{mixed['correct']}/{mixed['count']} ({m_acc:.0f}%)"
                i_str = f"{i_only['correct']}/{i_only['count']} ({i_acc:.0f}%)" if i_only['count'] > 0 else "-"
                scenario_rows.append((sid, m_str, i_str, diff))

        for sid, m_str, i_str, diff in scenario_rows:
            lines.append(f"    {sid:<4}  {m_str:<17}  {i_str:<18}  {diff}")

        if not scenario_rows:
            lines.append("    (none)")
        lines.append("")

        # By Scenario × Extra - show cells with mixed references
        lines.append("  Scenario × Extra (cells with mixed references):")
        lines.append("    ID    Extra  Mixed           I_only          Diff")
        lines.append("    ----  -----  --------------  --------------  -----")

        scenario_extra_rows = []
        for (sid, extra) in sorted(stats['by_scenario_extra'].keys()):
            se_stats = stats['by_scenario_extra'][(sid, extra)]
            mixed = se_stats.get('mixed', {'count': 0, 'correct': 0})
            i_only = se_stats.get('I_only', {'count': 0, 'correct': 0})

            if mixed['count'] > 0:
                m_acc = 100 * mixed['correct'] / mixed['count']
                i_acc = 100 * i_only['correct'] / i_only['count'] if i_only['count'] > 0 else None
                diff = f"{m_acc - i_acc:+.0f}%" if i_acc is not None else "n/a"
                m_str = f"{mixed['correct']}/{mixed['count']}"
                i_str = f"{i_only['correct']}/{i_only['count']}" if i_only['count'] > 0 else "-"
                scenario_extra_rows.append((sid, extra, m_str, i_str, diff))

        for sid, extra, m_str, i_str, diff in scenario_extra_rows:
            lines.append(f"    {sid:<4}  {extra:<5}  {m_str:<14}  {i_str:<14}  {diff}")

        if not scenario_extra_rows:
            lines.append("    (none)")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Analyze self-reference patterns in reasoning')
    parser.add_argument('--logs-dir', type=str, default='tom_llm_logs', help='Logs directory')
    args = parser.parse_args()

    # Find all log files (not JSON)
    log_files = []
    for f in os.listdir(args.logs_dir):
        if f.endswith('.log'):
            log_files.append(os.path.join(args.logs_dir, f))

    if not log_files:
        print(f"No log files found in {args.logs_dir}")
        return

    # Analyze each model
    def make_stats():
        return {
            'total': 0,
            'total_i': 0,
            'total_a': 0,
            'by_style': defaultdict(lambda: {'count': 0, 'correct': 0}),
            'by_extra': defaultdict(lambda: defaultdict(lambda: {'count': 0, 'correct': 0})),
            'by_scenario': defaultdict(lambda: defaultdict(lambda: {'count': 0, 'correct': 0})),
            'by_scenario_extra': defaultdict(lambda: defaultdict(lambda: {'count': 0, 'correct': 0})),
        }

    model_stats = defaultdict(make_stats)

    for log_file in sorted(log_files):
        # Extract model name from filename
        basename = os.path.basename(log_file)
        # Remove timestamp and .log
        model_name = re.sub(r'_\d+\.log$', '', basename)

        # Load corresponding game_data.json for was_optimal lookup
        game_data_lookup = load_game_data(log_file)

        traces = parse_log_file(log_file, game_data_lookup)

        for trace in traces:
            ref_analysis = analyze_self_reference(trace['reasoning'])

            stats = model_stats[model_name]
            stats['total'] += 1
            stats['total_i'] += ref_analysis['i_count']
            stats['total_a'] += ref_analysis['a_count']

            style = ref_analysis['style']
            stats['by_style'][style]['count'] += 1
            if trace['was_correct']:
                stats['by_style'][style]['correct'] += 1

            # Track by extra
            extra = trace.get('extra')
            if extra:
                stats['by_extra'][extra][style]['count'] += 1
                if trace['was_correct']:
                    stats['by_extra'][extra][style]['correct'] += 1

            # Track by scenario
            scenario_id = trace.get('scenario_id')
            if scenario_id:
                stats['by_scenario'][scenario_id][style]['count'] += 1
                if trace['was_correct']:
                    stats['by_scenario'][scenario_id][style]['correct'] += 1

            # Track by scenario × extra
            if scenario_id and extra:
                key = (scenario_id, extra)
                stats['by_scenario_extra'][key][style]['count'] += 1
                if trace['was_correct']:
                    stats['by_scenario_extra'][key][style]['correct'] += 1

    # Format and output
    output = format_results(dict(model_stats))

    output_file = os.path.join(args.logs_dir, 'self_reference_analysis.txt')
    with open(output_file, 'w') as f:
        f.write(output)

    # Brief summary
    print(f"Analyzed {len(log_files)} log files")
    for model, stats in model_stats.items():
        if stats['total'] > 0:
            i_only = stats['by_style'].get('I_only', {})
            if i_only.get('count', 0) > 0:
                i_acc = 100 * i_only['correct'] / i_only['count']
                print(f"  {model}: I-only accuracy {i_acc:.1f}%")
    print(f"Full results: {output_file}")


if __name__ == '__main__':
    main()
