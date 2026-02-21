#!/usr/bin/env python3
"""
Validate reasoning traces in ToM test JSON files.

Checks that:
1. Files with "_think", "thinking", "r1" in name OR free_response=true have reasoning traces
2. Files without those markers AND free_response=false do NOT have reasoning traces

Reasoning traces are detected by:
- For free_response=true: Check if "action" field contains more than just the action command
- For thinking models with free_response=false: Check LOG files for "REASONING TRACE:" markers
"""

import json
import os
import re
import glob
import argparse
from typing import List, Tuple, Dict, Any


def is_thinking_model_by_name(filename: str) -> bool:
    """
    Check if filename indicates a thinking model.

    Criteria: "_think" OR "thinking" OR "r1" OR "lowthink" in filename
    """
    name_lower = filename.lower()
    return ('_think' in name_lower or 'thinking' in name_lower or
            'r1' in name_lower or 'lowthink' in name_lower)


def is_simple_action(action: str) -> bool:
    """Check if action string is just a simple action (no reasoning)."""
    if not action:
        return True

    action = action.strip()

    # Simple actions: Pass, Ask(Player, Container), Tell(Player, Container, Contents)
    if action.lower() == 'pass':
        return True

    # Check for Ask/Tell with minimal variation
    # Ask(X, Y) or Tell(X, Y, Z)
    if re.match(r'^Ask\s*\([^)]+\)\s*$', action, re.IGNORECASE):
        return True
    if re.match(r'^Tell\s*\([^)]+\)\s*$', action, re.IGNORECASE):
        return True

    # If the action string is short (e.g., under 50 chars), it's likely simple
    # This handles minor variations
    if len(action) < 50:
        # Check if it contains only action-related content
        action_lower = action.lower()
        if 'pass' in action_lower or 'ask(' in action_lower or 'tell(' in action_lower:
            # Simple if it doesn't have reasoning indicators
            reasoning_indicators = ['because', 'since', 'therefore', 'reason', 'let me',
                                   'analysis', 'consider', 'think', 'know', 'believe',
                                   'scenario', 'situation', '\n', '**', '##']
            if not any(ind in action_lower for ind in reasoning_indicators):
                return True

    return False


def has_reasoning_in_record(record: dict) -> bool:
    """Check if a turn record contains reasoning traces."""
    action = record.get('action', '')
    return not is_simple_action(action)


def count_reasoning_traces_in_log(log_path: str) -> Tuple[int, int]:
    """
    Count reasoning traces in a log file.

    Looks for either:
    - "REASONING TRACE:" markers (thinking models)
    - Multi-line responses after "Action:" (free_response models)

    Returns (traces_found, total_turns).
    """
    if not os.path.exists(log_path):
        return 0, 0

    try:
        with open(log_path, 'r') as f:
            content = f.read()
    except IOError:
        return 0, 0

    # Count turns by looking for "It is your turn."
    turns = content.count("It is your turn.")

    # Count explicit reasoning traces
    explicit_traces = content.count("REASONING TRACE:")

    # Also count chain-of-thought responses
    # Split into trial sections first, then look for multi-line Action responses
    cot_traces = 0
    # Split by trial markers
    trial_sections = re.split(r'--- Running Trial \d+/\d+', content)
    for section in trial_sections[1:]:  # Skip content before first trial
        # Find Action response within this trial only
        # Look for text between "Action:" and "A answers:" within the section
        action_match = re.search(r'Action:\s*(.+?)\nA answers:', section, re.DOTALL)
        if action_match:
            action_text = action_match.group(1).strip()
            # Has reasoning if: multiple lines AND substantial length
            # Simple actions like "Pass" or "Ask(B, bag)" are single line and short
            lines = action_text.split('\n')
            if len(lines) > 2 and len(action_text) > 100:
                cot_traces += 1

    # Return the max of explicit traces or CoT traces found
    return max(explicit_traces, cot_traces), turns


def check_log_for_no_reasoning(log_path: str) -> Tuple[bool, int]:
    """
    Check that a log file has NO reasoning traces.

    Returns (has_no_reasoning, traces_found).
    """
    if not os.path.exists(log_path):
        return True, 0

    try:
        with open(log_path, 'r') as f:
            content = f.read()
    except IOError:
        return True, 0

    traces = content.count("REASONING TRACE:")
    return traces == 0, traces


def validate_json_file(filepath: str) -> Dict[str, Any]:
    """
    Validate a single JSON file for reasoning trace consistency.

    For thinking models with free_response=false, checks the corresponding LOG file.
    For free_response=true, checks the action field in JSON.
    For non-thinking models, ensures NO reasoning traces in either location.

    Returns dict with:
    - filepath: the file path
    - is_thinking_model: whether filename indicates thinking
    - has_free_response: whether free_response=true in records
    - should_have_reasoning: combined check
    - records_with_reasoning: count of records with reasoning (in action field)
    - records_without_reasoning: count of records without reasoning
    - log_traces: count of REASONING TRACE markers in log file
    - log_turns: count of turns in log file
    - total_records: total A records
    - is_valid: whether the file meets expectations
    - issues: list of issues found
    """
    result = {
        'filepath': filepath,
        'filename': os.path.basename(filepath),
        'is_thinking_model': False,
        'has_free_response': False,
        'should_have_reasoning': False,
        'records_with_reasoning': 0,
        'records_without_reasoning': 0,
        'log_traces': 0,
        'log_turns': 0,
        'total_records': 0,
        'is_valid': True,
        'issues': []
    }

    filename = os.path.basename(filepath)
    result['is_thinking_model'] = is_thinking_model_by_name(filename)

    # Get corresponding log file path
    log_path = filepath.replace('_game_data.json', '.log')

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        result['is_valid'] = False
        result['issues'].append(f"Failed to load JSON: {e}")
        return result

    # Handle both formats: list of records or dict with turn_records
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = data.get('turn_records', [])
    else:
        result['is_valid'] = False
        result['issues'].append("Unknown JSON format")
        return result

    # Filter to A records only
    a_records = [r for r in records if r.get('character') == 'A']
    result['total_records'] = len(a_records)

    if not a_records:
        result['issues'].append("No A records found")
        return result

    # Check free_response in first record
    result['has_free_response'] = a_records[0].get('free_response', False)

    # Determine if this file should have reasoning
    result['should_have_reasoning'] = result['is_thinking_model'] or result['has_free_response']

    # Count records with/without reasoning in action field
    for record in a_records:
        if has_reasoning_in_record(record):
            result['records_with_reasoning'] += 1
        else:
            result['records_without_reasoning'] += 1

    # Check log file for reasoning traces
    result['log_traces'], result['log_turns'] = count_reasoning_traces_in_log(log_path)

    # Validate expectations based on configuration
    total = result['total_records']
    with_reasoning_json = result['records_with_reasoning']
    log_traces = result['log_traces']

    if result['should_have_reasoning']:
        # Should HAVE reasoning - check that there are SOME reasoning traces
        has_reasoning_in_json = with_reasoning_json > 0
        has_reasoning_in_log = log_traces > 0

        if result['has_free_response']:
            # For free_response=true: reasoning can be in action field OR log file
            if not has_reasoning_in_json and not has_reasoning_in_log:
                result['is_valid'] = False
                result['issues'].append(
                    f"free_response=true but no reasoning found in action fields "
                    f"({with_reasoning_json}/{total}) or log ({log_traces}/{result['log_turns']})"
                )
        elif result['is_thinking_model']:
            # For thinking models with free_response=false: check log file
            if not os.path.exists(log_path):
                result['is_valid'] = False
                result['issues'].append(f"Log file not found: {os.path.basename(log_path)}")
            elif not has_reasoning_in_log:
                result['is_valid'] = False
                result['issues'].append(
                    f"Thinking model but no REASONING TRACE found in log "
                    f"({log_traces}/{result['log_turns']})"
                )
    else:
        # Should NOT have reasoning - check that there are NO (or very few) reasoning traces
        issues_found = []

        # Check action field - allow up to 5% tolerance for noise
        if with_reasoning_json > 0:
            ratio = with_reasoning_json / total if total > 0 else 0
            if ratio > 0.05:
                issues_found.append(
                    f"{with_reasoning_json}/{total} ({ratio*100:.1f}%) action fields have reasoning"
                )

        # Check log file - allow up to 5% tolerance
        if log_traces > 0 and os.path.exists(log_path) and result['log_turns'] > 0:
            ratio = log_traces / result['log_turns']
            if ratio > 0.05:
                issues_found.append(
                    f"{log_traces}/{result['log_turns']} ({ratio*100:.1f}%) log turns have reasoning"
                )

        if issues_found:
            result['is_valid'] = False
            result['issues'].append(
                f"Non-thinking model should have NO reasoning but: " + "; ".join(issues_found)
            )

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Validate reasoning traces in ToM test JSON files'
    )
    parser.add_argument(
        '--dir', '-d',
        default='tom_llm_logs',
        help='Directory containing JSON files (default: tom_llm_logs)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show details for all files, not just failures'
    )
    parser.add_argument(
        '--show-examples', '-e',
        action='store_true',
        help='Show example action strings from failed files'
    )
    args = parser.parse_args()

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, args.dir)

    if not os.path.exists(logs_dir):
        print(f"Error: Directory {logs_dir} not found")
        return 1

    # Find all game_data.json files
    json_files = glob.glob(os.path.join(logs_dir, '*_game_data.json'))

    if not json_files:
        print(f"No *_game_data.json files found in {logs_dir}")
        return 1

    print(f"Validating {len(json_files)} JSON files in {logs_dir}")
    print("=" * 80)

    results = []
    for filepath in sorted(json_files):
        result = validate_json_file(filepath)
        results.append(result)

    # Separate valid and invalid
    valid_results = [r for r in results if r['is_valid']]
    invalid_results = [r for r in results if not r['is_valid']]

    # Summary by category
    thinking_should_have = [r for r in results if r['should_have_reasoning']]
    non_thinking_should_not = [r for r in results if not r['should_have_reasoning']]

    print(f"\n{'Category':<40} | {'Total':<6} | {'Valid':<6} | {'Invalid':<6}")
    print("-" * 70)
    print(f"{'Thinking models (should have reasoning)':<40} | {len(thinking_should_have):<6} | "
          f"{len([r for r in thinking_should_have if r['is_valid']]):<6} | "
          f"{len([r for r in thinking_should_have if not r['is_valid']]):<6}")
    print(f"{'Non-thinking (should NOT have reasoning)':<40} | {len(non_thinking_should_not):<6} | "
          f"{len([r for r in non_thinking_should_not if r['is_valid']]):<6} | "
          f"{len([r for r in non_thinking_should_not if not r['is_valid']]):<6}")
    print("-" * 70)
    print(f"{'TOTAL':<40} | {len(results):<6} | {len(valid_results):<6} | {len(invalid_results):<6}")

    # Show failures
    if invalid_results:
        print(f"\n{'='*80}")
        print("FAILURES:")
        print("=" * 80)
        for result in invalid_results:
            print(f"\n{result['filename']}")
            print(f"  Thinking model by name: {result['is_thinking_model']}")
            print(f"  free_response: {result['has_free_response']}")
            print(f"  Should have reasoning: {result['should_have_reasoning']}")
            print(f"  Action fields with reasoning: {result['records_with_reasoning']}/{result['total_records']}")
            print(f"  Log traces: {result['log_traces']}/{result['log_turns']} turns")
            for issue in result['issues']:
                print(f"  ISSUE: {issue}")

            if args.show_examples:
                # Show example action strings that match the issue
                try:
                    with open(result['filepath'], 'r') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        records = data.get('turn_records', [])
                    else:
                        records = data
                    a_records = [r for r in records if r.get('character') == 'A']
                    if a_records:
                        if result['should_have_reasoning']:
                            # Show an example WITH reasoning (or lack thereof)
                            examples_with = [r for r in a_records if has_reasoning_in_record(r)]
                            if examples_with:
                                print(f"  Example WITH reasoning:")
                                print(f"    {repr(examples_with[0].get('action', '')[:200])}...")
                            else:
                                print(f"  No examples with reasoning found")
                        else:
                            # Show an example that has reasoning (the problem)
                            examples_with = [r for r in a_records if has_reasoning_in_record(r)]
                            if examples_with:
                                print(f"  Example of unexpected reasoning:")
                                print(f"    {repr(examples_with[0].get('action', '')[:200])}...")
                            else:
                                print(f"  No examples with reasoning in action field")
                except Exception as e:
                    print(f"  Could not load examples: {e}")

    # Show verbose details
    if args.verbose and valid_results:
        print(f"\n{'='*80}")
        print("VALID FILES:")
        print("=" * 80)
        for result in valid_results:
            status = "THINK" if result['should_have_reasoning'] else "NO-THINK"
            reasoning_info = ""
            if result['has_free_response']:
                reasoning_info = f"action:{result['records_with_reasoning']}/{result['total_records']}"
            elif result['is_thinking_model']:
                reasoning_info = f"log:{result['log_traces']}/{result['log_turns']}"
            else:
                reasoning_info = f"action:{result['records_with_reasoning']}, log:{result['log_traces']}"
            print(f"  [{status}] {result['filename']}: {reasoning_info}")

    # Final summary
    print(f"\n{'='*80}")
    if invalid_results:
        print(f"RESULT: FAILED - {len(invalid_results)} files have inconsistent reasoning traces")
        return 1
    else:
        print(f"RESULT: PASSED - All {len(results)} files have consistent reasoning traces")
        return 0


if __name__ == '__main__':
    exit(main())
