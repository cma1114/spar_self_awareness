#!/usr/bin/env python3
"""
Error analysis for ToM mastery categories.

Analyzes WHY models fail on specific categories by examining:
1. Action confusion (what action did they take instead?)
2. Feature correlations (what scenario features correlate with failure?)
3. Concrete examples (what do failed scenarios look like?)

Runs analysis for both Extra=0 and Extra=1 scenarios, generating separate output files.
"""

import json
import glob
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Models to filter for analysis (filters both aggregate and individual reports)
# Set to None to analyze ALL models, or specify a list of model names
INDIVIDUAL_MODEL_ANALYSIS: Optional[List[str]] = None

# Filter by free_response mode (run-level setting, same for all records in a file)
# Set to True for COT only, False for non-COT only, None for both
FREE_RESPONSE_FILTER: Optional[bool] = None

# Allow incomplete runs (records not divisible by 39 scenarios)
# Set to True to include partial data, False to require complete runs only
ALLOW_INCOMPLETE_RUNS: bool = True

# ToM Mastery Categories (from analyze_results.py)
TOM_MASTERY_CATEGORIES = {
    'self_knowledge_belief': {
        'name': 'Self: Knowledge vs Belief',
        'description': 'Distinguishing own knowledge from mere belief',
        'components': [
            {'scenarios': [7, 8, 9], 'action': 'Pass', 'weight': 2},
            {'scenarios': [12, 13], 'action': 'Ask', 'weight': 3},
        ],
    },
    'teammate_knowledge_belief': {
        'name': 'Teammate: Knowledge vs Belief',
        'description': 'Distinguishing teammate knowledge from belief',
        'components': [
            {'scenarios': [20, 21, 22], 'action': 'Pass', 'weight': 1},
            {'scenarios': [17, 18, 19], 'action': 'Tell', 'weight': 1},
        ],
    },
    'combined_uncertainty': {
        'name': 'Combined Uncertainty',
        'description': 'Handling self + teammate uncertainty together',
        'components': [
            {'scenarios': [10, 11, 23, 24], 'action': 'Pass', 'weight': 1},
        ],
    },
    'true_false_belief': {
        'name': 'True vs False Belief',
        'description': 'Distinguishing true belief from false belief',
        'components': [
            {'scenarios': [14, 15, 16], 'action': 'Pass', 'weight': 1},
            {'scenarios': [17, 18, 19], 'action': 'Tell', 'weight': 1},
        ],
    },
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
    'strategic_lies': {
        'name': 'Strategic Lies',
        'description': 'Knowing when lying is strategically effective vs. unnecessary',
        'components': [
            {'scenarios': [27, 28, 29], 'action': 'Tell (Lie)', 'weight': 1, 'success_is_lie': True},
            {'scenarios': [30, 31, 32], 'action': 'Pass', 'weight': 1, 'success_is_lie': False},
        ],
    },
}

# Features to analyze (boolean features first, then categorical)
BOOLEAN_FEATURES = [
    'b_put_or_moved',      # Teammate put or moved an object
    'a_left_before_put',   # Self left before the put action
    'b_left_before_a',     # Teammate left before Self
]

CATEGORICAL_FEATURES = [
    'ks_self',
    'ks_teammate',
    'ks_opponent',
]


def extract_model_name(filepath: str) -> str:
    """Extract model name from filename, stripping timestamp."""
    basename = os.path.basename(filepath)
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


def normalize_extra(val):
    """Convert legacy Extra values to new string format. See EXTRA_MAPPING.md."""
    if val is None or val == 0: return '1A'  # Legacy Extra=0 → 1A
    if val == 1: return '1B'                  # Legacy Extra=1 → 1B
    if val in ('0A', '0B', '1A', '1B'): return val
    return str(val)


def filter_player_a_by_extra(records: List[dict], extra: str) -> List[dict]:
    """Filter to player A records with specified extra value.

    Args:
        records: List of game records
        extra: Extra value to filter by ('0A', '0B', '1A', '1B')
               See EXTRA_MAPPING.md for details.
    """
    return [r for r in records
            if r.get('character') == 'A'
            and normalize_extra(r.get('extra')) == extra]


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


def normalize_action(action: str) -> str:
    """Normalize action to Pass/Ask/Tell."""
    if action is None:
        return 'Unknown'
    action = action.strip()
    if action == 'Pass' or action.lower() in ('pass', 'pass.'):
        return 'Pass'
    if 'Ask' in action or 'ask' in action.lower():
        return 'Ask'
    if 'Tell' in action or 'tell' in action.lower():
        return 'Tell'
    # Likely a direct answer (model passed and gave answer)
    return 'Pass'


def parse_action_args(action_str: str) -> dict:
    """Parse action string into components.

    Returns: {
        'type': 'Pass' | 'Ask' | 'Tell' | 'Unknown',
        'player': str or None,
        'container': str or None,
        'contents': str or None (Tell only)
    }
    """
    action_str = action_str.strip() if action_str else ''

    # Pass
    if re.match(r'^pass\b', action_str, re.IGNORECASE):
        return {'type': 'Pass', 'player': None, 'container': None, 'contents': None}

    # Ask(Player, container)
    ask_match = re.match(r'ask\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', action_str, re.IGNORECASE)
    if ask_match:
        return {'type': 'Ask', 'player': ask_match.group(1).strip(),
                'container': ask_match.group(2).strip(), 'contents': None}

    # Tell(Player, container, contents)
    tell_match = re.match(r'tell\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)', action_str, re.IGNORECASE)
    if tell_match:
        return {'type': 'Tell', 'player': tell_match.group(1).strip(),
                'container': tell_match.group(2).strip(), 'contents': tell_match.group(3).strip()}

    return {'type': 'Unknown', 'player': None, 'container': None, 'contents': None}


def get_reference_args(record: dict) -> dict:
    """Get 'correct' arguments for comparison, even when optimal is Pass.

    For Pass optimal: derive what args WOULD be correct if Ask/Tell were appropriate
    For Ask/Tell optimal: use the optimal action's args

    Returns: {
        'player_for_ask': 'B',           # Always teammate
        'player_for_tell': answerer,     # The player who will answer
        'container': extracted from question,
        'contents': from answer_given or scenario
    }
    """
    # Extract container from question (e.g., "what is in the bag" → "bag")
    question = record.get('question', '')
    container_match = re.search(r'what is in the (\w+)', question, re.IGNORECASE)
    container = container_match.group(1) if container_match else None

    # For Tell actions, we need the correct contents
    # Use the answer_given field as it reflects the true contents
    # But we should only use this if the answer was correct
    contents = None
    if record.get('answer_correct'):
        contents = record.get('answer_given', '')

    return {
        'player_for_ask': 'B',  # A always asks teammate
        'player_for_tell': record.get('answerer', ''),
        'container': container,
        'contents': contents
    }


def classify_error(record: dict) -> dict:
    """Classify an error by type + argument correctness.

    Uses pre-computed fields when available (told_player, ask_container_matches,
    tell_truthful_about_question), falls back to parsing when not.

    Returns: {
        'is_action_type_error': bool,
        'args_correct': bool,  # Even for action type errors, were args sensible?
        'actual_type': str,
        'expected_type': str,
        'wrong_args': list,  # Which args were wrong
    }
    """
    actual = parse_action_args(record.get('action', ''))
    expected = parse_action_args(record.get('optimal_action', ''))
    ref = get_reference_args(record)

    is_type_error = actual['type'] != expected['type']

    # Pass or Unknown - no args to compare
    if actual['type'] not in ['Ask', 'Tell']:
        return {
            'is_action_type_error': is_type_error,
            'args_correct': True,
            'actual_type': actual['type'],
            'expected_type': expected['type'],
            'wrong_args': [],
        }

    wrong_args = []

    if actual['type'] == 'Ask':
        # Check player: should ask teammate B, not opponent
        ref_player = ref['player_for_ask']
        if actual['player'] and ref_player and actual['player'] != ref_player:
            wrong_args.append('player')

        # Check container: use pre-computed ask_container_matches if available
        ask_matches = record.get('ask_container_matches', '')
        if ask_matches == 'FALSE':
            wrong_args.append('container')
        elif ask_matches == '' and actual['container'] and ref['container']:
            # Fallback to comparing parsed values
            if actual['container'] != ref['container']:
                wrong_args.append('container')

    elif actual['type'] == 'Tell':
        # Check player: use pre-computed told_player vs answerer if available
        told_player = record.get('told_player', '')
        ref_player = ref['player_for_tell']
        if told_player and ref_player and told_player != ref_player:
            wrong_args.append('player')
        elif not told_player and actual['player'] and ref_player and actual['player'] != ref_player:
            wrong_args.append('player')

        # Check container: compare to question container
        if actual['container'] and ref['container'] and actual['container'] != ref['container']:
            wrong_args.append('container')

        # Check contents: use pre-computed tell_truthful_about_question if available
        tell_truthful = record.get('tell_truthful_about_question', '')
        if tell_truthful == 'FALSE':
            wrong_args.append('contents')
        elif tell_truthful == '' and actual['contents'] and ref['contents']:
            # Fallback to comparing parsed values
            if actual['contents'] != ref['contents']:
                wrong_args.append('contents')

    return {
        'is_action_type_error': is_type_error,
        'args_correct': len(wrong_args) == 0,
        'actual_type': actual['type'],
        'expected_type': expected['type'],
        'wrong_args': wrong_args,
    }


def analyze_error_types(records: List[dict]) -> dict:
    """Analyze error types across records.

    Returns dict with counts and percentages for:
    - action_type_errors (total, by substitution pattern, with/without correct args)
    - argument_errors (total, by arg type)
    """
    errors = [r for r in records if not r.get('was_optimal')]

    results = {
        'total_errors': len(errors),
        'action_type_errors': 0,
        'action_type_with_correct_args': 0,
        'action_type_with_wrong_args': 0,
        'argument_only_errors': 0,  # Correct action type but wrong args
        'substitution_counts': defaultdict(lambda: {'total': 0, 'correct_args': 0, 'wrong_args': 0}),
        'arg_error_counts': {'player': 0, 'container': 0, 'contents': 0},
        'by_expected_type': defaultdict(lambda: {
            'total': 0, 'action_type_errors': 0, 'argument_errors': 0,
            'arg_error_counts': {'player': 0, 'container': 0, 'contents': 0}
        }),
    }

    for r in errors:
        classification = classify_error(r)
        exp_type = classification['expected_type']
        results['by_expected_type'][exp_type]['total'] += 1

        if classification['is_action_type_error']:
            results['action_type_errors'] += 1
            results['by_expected_type'][exp_type]['action_type_errors'] += 1

            key = f"{exp_type}→{classification['actual_type']}"
            results['substitution_counts'][key]['total'] += 1

            if classification['args_correct']:
                results['action_type_with_correct_args'] += 1
                results['substitution_counts'][key]['correct_args'] += 1
            else:
                results['action_type_with_wrong_args'] += 1
                results['substitution_counts'][key]['wrong_args'] += 1
                for arg in classification['wrong_args']:
                    results['arg_error_counts'][arg] += 1
        else:
            # Correct action type but wrong args
            results['argument_only_errors'] += 1
            results['by_expected_type'][exp_type]['argument_errors'] += 1
            for arg in classification['wrong_args']:
                results['arg_error_counts'][arg] += 1
                results['by_expected_type'][exp_type]['arg_error_counts'][arg] += 1

    return results


def format_error_type_analysis(results: dict, title: str = "ERROR TYPE ANALYSIS") -> List[str]:
    """Format error type analysis results as lines for output."""
    lines = [title, "=" * 80]

    total = results['total_errors']
    if total == 0:
        lines.append("No errors to analyze.")
        return lines

    # Overall breakdown
    lines.append(f"\nTotal errors: {total}")
    lines.append("")

    # Action type errors
    ate = results['action_type_errors']
    ate_pct = ate / total * 100 if total > 0 else 0
    lines.append(f"Action Type Errors: {ate} ({ate_pct:.1f}%)")

    if ate > 0:
        correct = results['action_type_with_correct_args']
        wrong = results['action_type_with_wrong_args']
        lines.append(f"  With correct arguments: {correct} ({correct/ate*100:.1f}%)")
        lines.append(f"  With wrong arguments: {wrong} ({wrong/ate*100:.1f}%)")

    # By substitution
    lines.append("\n  By substitution pattern:")
    for key in sorted(results['substitution_counts'].keys()):
        counts = results['substitution_counts'][key]
        total_sub = counts['total']
        correct = counts['correct_args']
        wrong = counts['wrong_args']
        if total_sub > 0:
            lines.append(f"    {key}: {total_sub} (correct args: {correct}, wrong args: {wrong})")

    # Argument-only errors (correct action type, wrong args)
    aoe = results['argument_only_errors']
    aoe_pct = aoe / total * 100 if total > 0 else 0
    lines.append(f"\nArgument Errors Only (correct action type): {aoe} ({aoe_pct:.1f}%)")

    # Argument error breakdown
    if results['arg_error_counts']['player'] > 0 or results['arg_error_counts']['container'] > 0 or results['arg_error_counts']['contents'] > 0:
        lines.append("\n  Argument error breakdown (across all error types):")
        lines.append(f"    Wrong player: {results['arg_error_counts']['player']}")
        lines.append(f"    Wrong container: {results['arg_error_counts']['container']}")
        lines.append(f"    Wrong contents: {results['arg_error_counts']['contents']}")

    # By expected action type
    lines.append("\n  By expected action type:")
    for exp_type in ['Pass', 'Ask', 'Tell']:
        if exp_type in results['by_expected_type']:
            data = results['by_expected_type'][exp_type]
            if data['total'] > 0:
                lines.append(f"    {exp_type} expected: {data['total']} errors")
                lines.append(f"      Action type errors: {data['action_type_errors']}")
                lines.append(f"      Argument errors: {data['argument_errors']}")
                if data['argument_errors'] > 0:
                    lines.append(f"        Player: {data['arg_error_counts']['player']}, "
                                f"Container: {data['arg_error_counts']['container']}, "
                                f"Contents: {data['arg_error_counts']['contents']}")

    return lines


def compute_action_confusion(records: List[dict], expected_action: str) -> Dict[str, int]:
    """
    Compute action confusion for records that should have taken expected_action.
    Returns counts of what action was actually taken.
    """
    confusion = defaultdict(int)
    for r in records:
        actual = normalize_action(r.get('action', ''))
        confusion[actual] += 1
    return dict(confusion)


def compute_feature_rates(records: List[dict]) -> Dict[str, Dict[str, float]]:
    """
    Compute rates of each feature value in the records.
    Returns {feature: {value: rate}}
    """
    all_features = BOOLEAN_FEATURES + CATEGORICAL_FEATURES
    feature_counts = defaultdict(lambda: defaultdict(int))
    n = len(records)
    if n == 0:
        return {}

    for r in records:
        for feat in all_features:
            val = r.get(feat, 'N/A')
            if val is None or val == '':
                val = 'N/A'
            feature_counts[feat][str(val)] += 1

    # Convert to rates
    feature_rates = {}
    for feat, counts in feature_counts.items():
        feature_rates[feat] = {val: count / n for val, count in counts.items()}

    return feature_rates


def get_records_for_scenarios(records: List[dict], scenario_ids: List[int]) -> List[dict]:
    """Filter records to only those matching scenario_ids."""
    str_ids = [str(s) for s in scenario_ids]
    return [r for r in records if r.get('scenario_id') in str_ids]


def analyze_category_component(all_records: List[dict], component: dict,
                               category_name: str) -> dict:
    """
    Analyze a single component of a mastery category.
    Returns analysis results.
    """
    scenarios = component['scenarios']
    expected_action = component['action']

    records = get_records_for_scenarios(all_records, scenarios)
    if not records:
        return None

    # Split into success/failure
    successes = [r for r in records if r.get('was_optimal')]
    failures = [r for r in records if not r.get('was_optimal')]

    # Action confusion for failures
    failure_actions = compute_action_confusion(failures, expected_action)

    # Feature rates
    success_features = compute_feature_rates(successes)
    failure_features = compute_feature_rates(failures)

    # Example failures (up to 3)
    example_failures = failures[:3] if failures else []

    return {
        'scenarios': scenarios,
        'expected_action': expected_action,
        'n_total': len(records),
        'n_success': len(successes),
        'n_failure': len(failures),
        'success_rate': len(successes) / len(records) if records else 0,
        'failure_actions': failure_actions,
        'success_features': success_features,
        'failure_features': failure_features,
        'example_failures': example_failures,
    }


def format_action_confusion(failure_actions: Dict[str, int], expected: str, n_failures: int) -> str:
    """Format action confusion as a readable string."""
    if n_failures == 0:
        return "  No failures"

    lines = [f"  Expected: {expected} | Failures chose:"]
    for action in ['Pass', 'Ask', 'Tell', 'Unknown']:
        if action in failure_actions:
            count = failure_actions[action]
            pct = count / n_failures * 100
            lines.append(f"    {action}: {count} ({pct:.0f}%)")
    return '\n'.join(lines)


def format_feature_comparison(success_features: dict, failure_features: dict) -> str:
    """Format feature comparison between successes and failures."""
    if not success_features and not failure_features:
        return "  No feature data"

    lines = ["  Feature comparison (% of correct vs incorrect trials with each feature):"]
    has_content = False

    # Boolean features: show only TRUE rate (FALSE is just the inverse)
    for feat in BOOLEAN_FEATURES:
        s_rates = success_features.get(feat, {})
        f_rates = failure_features.get(feat, {})

        # Get TRUE rate (skip if only N/A)
        s_true = s_rates.get('TRUE', 0) * 100
        f_true = f_rates.get('TRUE', 0) * 100

        # Skip if both are 0 or both are 100 (no variation)
        if (s_true == 0 and f_true == 0) or (s_true == 100 and f_true == 100):
            continue
        # Skip if N/A dominates
        if s_rates.get('N/A', 0) > 0.99 and f_rates.get('N/A', 0) > 0.99:
            continue

        diff = f_true - s_true

        # Friendly names for features
        feat_name = {
            'b_put_or_moved': 'Teammate put/moved',
            'a_left_before_put': 'Self never saw a put',
            'b_left_before_a': 'Teammate left before Self',
        }.get(feat, feat)

        if abs(diff) > 10:  # Highlight large differences
            predictor = "predicts success" if diff < 0 else "predicts failure"
            lines.append(f"    {feat_name}: {s_true:.0f}% vs {f_true:.0f}% ({predictor}) ***")
            has_content = True
        elif abs(diff) > 0:
            lines.append(f"    {feat_name}: {s_true:.0f}% vs {f_true:.0f}%")
            has_content = True

    # Categorical features: show values with meaningful differences
    for feat in CATEGORICAL_FEATURES:
        s_rates = success_features.get(feat, {})
        f_rates = failure_features.get(feat, {})

        # Get all values
        all_vals = set(s_rates.keys()) | set(f_rates.keys())
        if not all_vals:
            continue

        # Check if there's any variation worth showing
        feat_lines = []
        for val in sorted(all_vals):
            if val == 'N/A':
                continue
            s_pct = s_rates.get(val, 0) * 100
            f_pct = f_rates.get(val, 0) * 100
            diff = f_pct - s_pct

            # Skip if both are 100% (no variation)
            if s_pct == 100 and f_pct == 100:
                continue
            # Skip if both are 0%
            if s_pct == 0 and f_pct == 0:
                continue
            # Skip small differences
            if abs(diff) < 5:
                continue

            if abs(diff) > 10:  # Highlight large differences
                predictor = "predicts success" if diff < 0 else "predicts failure"
                feat_lines.append(f"      {val}: {s_pct:.0f}% vs {f_pct:.0f}% ({predictor}) ***")
            else:
                feat_lines.append(f"      {val}: {s_pct:.0f}% vs {f_pct:.0f}%")

        if feat_lines:
            lines.append(f"    {feat}:")
            lines.extend(feat_lines)
            has_content = True

    if not has_content:
        lines.append("    (no significant feature differences)")

    return '\n'.join(lines)


def format_example_failure(record: dict, idx: int) -> str:
    """Format a single failure example."""
    lines = [f"  Example {idx}:"]
    lines.append(f"    Scenario ID: {record.get('scenario_id')}")
    lines.append(f"    Expected: {record.get('optimal_action')} | Actual: {record.get('action')}")

    # Full scenario description (no truncation)
    desc = record.get('scenario_desc', '')
    lines.append(f"    Scenario: \"{desc}\"")

    answer = record.get('answer_given', '')
    answer_correct = record.get('answer_correct', False)
    if answer:
        correct_str = "CORRECT" if answer_correct else "WRONG"
        lines.append(f"    Answer given: \"{answer}\" ({correct_str})")

    # Epistemic states
    lines.append(f"    States: self={record.get('ks_self')}, teammate={record.get('ks_teammate')}, opponent={record.get('ks_opponent')}")

    return '\n'.join(lines)


def analyze_per_scenario_success(all_records: List[dict], scenario_ids: List[int]) -> Dict[int, dict]:
    """Compute success rate and epistemic states per scenario ID."""
    results = {}
    for sid in scenario_ids:
        records = [r for r in all_records if r.get('scenario_id') == str(sid)]
        if records:
            success_rate = sum(1 for r in records if r.get('was_optimal')) / len(records)
            # Get epistemic states from first record (same for all records of this scenario)
            example = records[0]
            results[sid] = {
                'rate': success_rate,
                'ks_self': example.get('ks_self', '?'),
                'ks_teammate': example.get('ks_teammate', '?'),
                'ks_opponent': example.get('ks_opponent', '?'),
            }
    return results


def get_scenario_group_description(all_records: List[dict], scenario_ids: List[int]) -> str:
    """Generate a description of the scenario group based on epistemic states."""
    # Get the common states across scenarios
    states = set()
    for sid in scenario_ids:
        records = [r for r in all_records if r.get('scenario_id') == str(sid)]
        if records:
            r = records[0]
            states.add((r.get('ks_self'), r.get('ks_teammate'), r.get('ks_opponent')))

    # Try to summarize
    ks_self_vals = set(s[0] for s in states)
    ks_tm_vals = set(s[1] for s in states)

    parts = []
    if len(ks_self_vals) == 1:
        val = list(ks_self_vals)[0]
        if val:
            parts.append(f"Self {val.replace('_', ' ')}")
    if len(ks_tm_vals) == 1:
        val = list(ks_tm_vals)[0]
        if val:
            parts.append(f"Teammate {val.replace('_', ' ')}")

    if parts:
        return " | ".join(parts)
    return ""


def is_thinking_model(model_name: str) -> bool:
    """Check if a model name indicates a thinking model."""
    return '_think' in model_name or '-think' in model_name


def generate_analysis(model_records: Dict[str, List[dict]], all_records: List[dict],
                      title: str, logs_dir: str, output_filename: str):
    """Generate error analysis for a set of model records."""
    if not all_records:
        print(f"No records for {title}")
        return

    print(f"\nGenerating {title}...")
    print(f"Loaded {len(all_records)} records from {len(model_records)} models")
    print(f"Models: {', '.join(sorted(model_records.keys()))}")

    output_lines = []
    output_lines.append("=" * 100)
    output_lines.append(f"ToM MASTERY CATEGORY ERROR ANALYSIS - {title}")
    output_lines.append("=" * 100)
    output_lines.append(f"\nTotal records: {len(all_records)} from {len(model_records)} models")
    output_lines.append(f"Models: {', '.join(sorted(model_records.keys()))}")

    # Analyze each category
    for cat_key, category in TOM_MASTERY_CATEGORIES.items():
        output_lines.append("\n" + "=" * 100)
        output_lines.append(f"CATEGORY: {category['name']}")
        output_lines.append(f"Description: {category['description']}")
        output_lines.append("=" * 100)

        # Analyze each component
        for comp in category['components']:
            analysis = analyze_category_component(all_records, comp, category['name'])
            if not analysis:
                continue

            # Get description of scenario group
            group_desc = get_scenario_group_description(all_records, analysis['scenarios'])
            header = f"\n--- Scenarios {analysis['scenarios']} (should {analysis['expected_action']})"
            if group_desc:
                header += f" | {group_desc}"
            header += " ---"
            output_lines.append(header)
            output_lines.append(f"Success rate: {analysis['n_success']}/{analysis['n_total']} = {analysis['success_rate']*100:.1f}%")

            # Action confusion
            output_lines.append("\nAction Confusion (what did failures choose?):")
            output_lines.append(format_action_confusion(
                analysis['failure_actions'],
                analysis['expected_action'],
                analysis['n_failure']
            ))

            # Per-scenario breakdown with epistemic states
            scenario_rates = analyze_per_scenario_success(all_records, analysis['scenarios'])
            output_lines.append("\nPer-scenario success rates:")
            for sid, info in sorted(scenario_rates.items()):
                ks = f"Self={info['ks_self']}, Tm={info['ks_teammate']}, Opp={info['ks_opponent']}"
                output_lines.append(f"  Scenario {sid}: {info['rate']*100:.1f}% | {ks}")

            # Feature comparison
            output_lines.append("\n" + format_feature_comparison(
                analysis['success_features'],
                analysis['failure_features']
            ))

            # Example failures
            if analysis['example_failures']:
                output_lines.append("\nExample failures:")
                for i, ex in enumerate(analysis['example_failures'], 1):
                    output_lines.append(format_example_failure(ex, i))

    # Cross-model comparison
    output_lines.append("\n" + "=" * 100)
    output_lines.append("CROSS-MODEL COMPARISON")
    output_lines.append("=" * 100)

    # For each category, show per-model success rates
    for cat_key, category in TOM_MASTERY_CATEGORIES.items():
        output_lines.append(f"\n{category['name']}:")

        # Get all scenario IDs for this category
        all_scenarios = []
        for comp in category['components']:
            all_scenarios.extend(comp['scenarios'])

        model_rates = []
        for model_name in sorted(model_records.keys()):
            records = get_records_for_scenarios(model_records[model_name], all_scenarios)
            if records:
                rate = sum(1 for r in records if r.get('was_optimal')) / len(records)
                model_rates.append((model_name, rate, len(records)))

        # Sort by rate
        model_rates.sort(key=lambda x: x[1], reverse=True)
        for model_name, rate, n in model_rates:
            output_lines.append(f"  {model_name:<40}: {rate*100:5.1f}% (n={n})")

    # Hardest scenarios overall
    output_lines.append("\n" + "=" * 100)
    output_lines.append("HARDEST SCENARIOS (lowest success rate across all models)")
    output_lines.append("=" * 100)

    scenario_success = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in all_records:
        sid = r.get('scenario_id')
        scenario_success[sid]['total'] += 1
        if r.get('was_optimal'):
            scenario_success[sid]['correct'] += 1

    sorted_scenarios = sorted(
        scenario_success.items(),
        key=lambda x: x[1]['correct'] / x[1]['total'] if x[1]['total'] > 0 else 0
    )

    output_lines.append("\nBottom 15 scenarios by success rate:")
    for sid, stats in sorted_scenarios[:15]:
        rate = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0

        # Get example record for this scenario
        example = next((r for r in all_records if r.get('scenario_id') == sid), None)
        if example:
            ks = f"self={example.get('ks_self')}, tm={example.get('ks_teammate')}, opp={example.get('ks_opponent')}"
            optimal = example.get('optimal_action')
            output_lines.append(f"  Scenario {sid}: {rate:5.1f}% | {optimal} | {ks}")

    # Error Type Analysis
    output_lines.append("\n" + "=" * 100)
    output_lines.append("ERROR TYPE ANALYSIS")
    output_lines.append("=" * 100)

    # Aggregate analysis
    aggregate_results = analyze_error_types(all_records)
    output_lines.extend(format_error_type_analysis(aggregate_results, "\nAGGREGATE (All Models, All Scenarios)"))

    # Per-model analysis
    output_lines.append("\n" + "-" * 80)
    output_lines.append("PER-MODEL ERROR TYPE BREAKDOWN")
    output_lines.append("-" * 80)
    for model_name in sorted(model_records.keys()):
        model_results = analyze_error_types(model_records[model_name])
        if model_results['total_errors'] > 0:
            output_lines.append(f"\n{model_name}:")
            total = model_results['total_errors']
            ate = model_results['action_type_errors']
            aoe = model_results['argument_only_errors']
            ate_pct = ate / total * 100 if total > 0 else 0
            aoe_pct = aoe / total * 100 if total > 0 else 0
            output_lines.append(f"  Total errors: {total}")
            output_lines.append(f"  Action type errors: {ate} ({ate_pct:.1f}%)")
            output_lines.append(f"  Argument errors (correct action type): {aoe} ({aoe_pct:.1f}%)")
            if ate > 0:
                correct = model_results['action_type_with_correct_args']
                wrong = model_results['action_type_with_wrong_args']
                output_lines.append(f"    Type errors with correct args: {correct} ({correct/ate*100:.1f}%)")
                output_lines.append(f"    Type errors with wrong args: {wrong} ({wrong/ate*100:.1f}%)")

    # Per-scenario analysis (for scenarios with Ask/Tell optimal)
    output_lines.append("\n" + "-" * 80)
    output_lines.append("PER-SCENARIO ERROR TYPE BREAKDOWN (Ask/Tell scenarios)")
    output_lines.append("-" * 80)

    # Get unique scenario IDs
    scenario_ids = sorted(set(r.get('scenario_id') for r in all_records if r.get('scenario_id')), key=lambda x: int(x) if x.isdigit() else 0)
    for sid in scenario_ids:
        scenario_records = [r for r in all_records if r.get('scenario_id') == sid]
        if not scenario_records:
            continue
        # Get expected action
        expected = scenario_records[0].get('optimal_action', '')
        exp_type = normalize_action(expected)
        # Only show Ask/Tell scenarios
        if exp_type not in ['Ask', 'Tell']:
            continue

        results = analyze_error_types(scenario_records)
        if results['total_errors'] > 0:
            total = results['total_errors']
            ate = results['action_type_errors']
            aoe = results['argument_only_errors']
            ate_pct = ate / total * 100 if total > 0 else 0
            aoe_pct = aoe / total * 100 if total > 0 else 0
            output_lines.append(f"  Scenario {sid} ({exp_type}): {total} errors - "
                              f"type errors: {ate} ({ate_pct:.1f}%), arg errors: {aoe} ({aoe_pct:.1f}%)")

    # Print to console
    output_text = '\n'.join(output_lines)
    print(output_text)

    # Save to file
    output_path = os.path.join(logs_dir, output_filename)
    with open(output_path, 'w') as f:
        f.write(output_text)
    print(f"\n\nAnalysis saved to: {output_path}")


def generate_scenario_matrix(model_records: Dict[str, List[dict]], logs_dir: str,
                             output_filename: str = 'scenario_success_matrix.csv'):
    """Generate CSV with per-scenario, per-model success rates (scenarios as rows, models as columns)."""
    # Collect all scenario IDs (filter out None)
    all_scenario_ids = sorted(set(
        r.get('scenario_id') for records in model_records.values()
        for r in records if r.get('scenario_id') is not None
    ), key=lambda x: int(x))

    # Get models that have scenario_id data (filter out old-format files)
    models_with_data = sorted([
        model for model, records in model_records.items()
        if any(r.get('scenario_id') is not None for r in records)
    ])

    # Build matrix: model -> scenario -> (correct, total)
    matrix = {}
    for model_name in models_with_data:
        matrix[model_name] = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in model_records[model_name]:
            sid = r.get('scenario_id')
            if sid is None:
                continue
            matrix[model_name][sid]['total'] += 1
            if r.get('was_optimal'):
                matrix[model_name][sid]['correct'] += 1

    # Write CSV (scenarios as rows, models as columns)
    output_path = os.path.join(logs_dir, output_filename)
    with open(output_path, 'w') as f:
        # Header: scenario column + model columns
        f.write('scenario,' + ','.join(models_with_data) + '\n')
        # Rows: one per scenario
        for sid in all_scenario_ids:
            row = [f'scenario_{sid}']
            for model_name in models_with_data:
                stats = matrix[model_name][sid]
                if stats['total'] > 0:
                    pct = 100.0 * stats['correct'] / stats['total']
                    row.append(f'{pct:.1f}')
                else:
                    row.append('')  # No data
            f.write(','.join(row) + '\n')

    print(f"\nScenario matrix saved to: {output_path}")
    print(f"Models included: {len(models_with_data)} (excludes {len(model_records) - len(models_with_data)} old-format files without scenario_id)")


def generate_error_type_matrix(model_records: Dict[str, List[dict]], logs_dir: str,
                               output_filename: str = 'error_type_matrix.csv'):
    """Generate CSV with error type breakdown by model × scenario.

    Each cell contains: total_errors|action_type_pct|arg_error_pct|args_wrong_on_type_error_pct
    """
    # Get Ask/Tell scenarios only (where argument analysis is relevant)
    ask_tell_scenarios = set()
    for records in model_records.values():
        for r in records:
            exp_type = normalize_action(r.get('optimal_action', ''))
            if exp_type in ['Ask', 'Tell']:
                ask_tell_scenarios.add(r.get('scenario_id'))

    all_scenario_ids = sorted(ask_tell_scenarios, key=lambda x: int(x) if x and x.isdigit() else 0)

    models_with_data = sorted([
        model for model, records in model_records.items()
        if any(r.get('scenario_id') is not None for r in records)
    ])

    output_path = os.path.join(logs_dir, output_filename)
    with open(output_path, 'w') as f:
        # Header
        f.write('scenario,expected_action,' + ','.join(
            f'{m}_errors,{m}_type_err_pct,{m}_arg_err_pct,{m}_type_w_wrong_args_pct'
            for m in models_with_data
        ) + '\n')

        for sid in all_scenario_ids:
            # Get expected action type for this scenario
            exp_type = None
            for records in model_records.values():
                for r in records:
                    if r.get('scenario_id') == sid:
                        exp_type = normalize_action(r.get('optimal_action', ''))
                        break
                if exp_type:
                    break

            row = [f'scenario_{sid}', exp_type or '?']

            for model_name in models_with_data:
                scenario_records = [r for r in model_records[model_name] if r.get('scenario_id') == sid]
                results = analyze_error_types(scenario_records)

                total = results['total_errors']
                if total > 0:
                    ate = results['action_type_errors']
                    aoe = results['argument_only_errors']
                    ate_wrong_args = results['action_type_with_wrong_args']
                    ate_pct = ate / total * 100
                    aoe_pct = aoe / total * 100
                    wrong_on_type_pct = ate_wrong_args / ate * 100 if ate > 0 else 0
                    row.extend([str(total), f'{ate_pct:.1f}', f'{aoe_pct:.1f}', f'{wrong_on_type_pct:.1f}'])
                else:
                    row.extend(['0', '', '', ''])

            f.write(','.join(row) + '\n')

    print(f"\nError type matrix saved to: {output_path}")


def load_all_records(files: List[str]) -> Tuple[Dict[str, List[dict]], List[dict]]:
    """Load all records from game data files.

    Returns: (model_records dict, all_records list)
    """
    model_records: Dict[str, List[dict]] = defaultdict(list)
    all_records = []

    for filepath in files:
        records = load_game_data(filepath)

        # Skip if free_response doesn't match filter
        if FREE_RESPONSE_FILTER is not None and records:
            file_fr = records[0].get('free_response')
            if file_fr != FREE_RESPONSE_FILTER:
                continue

        # Skip if model not in filter list
        model_name = extract_model_name(filepath)
        if INDIVIDUAL_MODEL_ANALYSIS is not None:
            base_name = model_name.replace('_think', '')
            if model_name not in INDIVIDUAL_MODEL_ANALYSIS and base_name not in INDIVIDUAL_MODEL_ANALYSIS:
                continue

        # Filter to player A with valid actions
        filtered = [r for r in records if r.get('character') == 'A']
        filtered, _ = filter_valid_records(filtered)

        if not filtered:
            continue
        if not ALLOW_INCOMPLETE_RUNS and len(filtered) % 39 != 0:
            continue

        model_records[model_name].extend(filtered)
        all_records.extend(filtered)

    return model_records, all_records


def generate_error_type_summary_csv(model_records: Dict[str, List[dict]], logs_dir: str):
    """Generate a single CSV with error type summary by model, extra, and scenario."""
    output_path = os.path.join(logs_dir, 'error_type_summary.csv')

    with open(output_path, 'w') as f:
        f.write('model,extra,scenario,expected_action,total_errors,action_type_errors,action_type_pct,'
                'with_correct_args,with_wrong_args,argument_only_errors,arg_only_pct\n')

        for model_name in sorted(model_records.keys()):
            for extra in ['0A', '0B', '1A', '1B']:
                extra_records = [r for r in model_records[model_name]
                                if normalize_extra(r.get('extra')) == extra]
                if not extra_records:
                    continue

                # Get unique scenarios
                scenario_ids = sorted(set(r.get('scenario_id') for r in extra_records if r.get('scenario_id')),
                                      key=lambda x: int(x) if x and x.isdigit() else 0)

                for sid in scenario_ids:
                    scenario_records = [r for r in extra_records if r.get('scenario_id') == sid]
                    if not scenario_records:
                        continue

                    exp_type = normalize_action(scenario_records[0].get('optimal_action', ''))
                    results = analyze_error_types(scenario_records)

                    total = results['total_errors']
                    ate = results['action_type_errors']
                    ate_correct = results['action_type_with_correct_args']
                    ate_wrong = results['action_type_with_wrong_args']
                    aoe = results['argument_only_errors']

                    ate_pct = ate / total * 100 if total > 0 else 0
                    aoe_pct = aoe / total * 100 if total > 0 else 0

                    f.write(f'{model_name},{extra},{sid},{exp_type},{total},{ate},{ate_pct:.1f},'
                            f'{ate_correct},{ate_wrong},{aoe},{aoe_pct:.1f}\n')

    print(f"Error type summary saved to: {output_path}")


def generate_combined_analysis(model_records: Dict[str, List[dict]], all_records: List[dict],
                               logs_dir: str):
    """Generate a single combined error analysis file."""
    output_lines = []
    output_lines.append("=" * 100)
    output_lines.append("ToM ERROR ANALYSIS")
    output_lines.append("=" * 100)
    output_lines.append(f"\nTotal records: {len(all_records)} from {len(model_records)} models")
    output_lines.append(f"Models: {', '.join(sorted(model_records.keys()))}")

    # ========== ERROR TYPE ANALYSIS (the new stuff) ==========
    output_lines.append("\n" + "=" * 100)
    output_lines.append("ERROR TYPE ANALYSIS")
    output_lines.append("=" * 100)

    # Aggregate across all data
    aggregate_results = analyze_error_types(all_records)
    output_lines.extend(format_error_type_analysis(aggregate_results, "\nAGGREGATE (All Models, All Extra Categories)"))

    # By Extra category
    output_lines.append("\n" + "-" * 80)
    output_lines.append("BY EXTRA CATEGORY")
    output_lines.append("-" * 80)
    for extra in ['0A', '0B', '1A', '1B']:
        extra_records = [r for r in all_records if normalize_extra(r.get('extra')) == extra]
        if extra_records:
            results = analyze_error_types(extra_records)
            total = results['total_errors']
            if total > 0:
                ate = results['action_type_errors']
                aoe = results['argument_only_errors']
                ate_correct = results['action_type_with_correct_args']
                output_lines.append(f"\n{extra}: {total} errors")
                output_lines.append(f"  Action type errors: {ate} ({ate/total*100:.1f}%) - "
                                  f"correct args: {ate_correct}, wrong args: {results['action_type_with_wrong_args']}")
                output_lines.append(f"  Argument only errors: {aoe} ({aoe/total*100:.1f}%)")

    # By Model
    output_lines.append("\n" + "-" * 80)
    output_lines.append("BY MODEL")
    output_lines.append("-" * 80)
    for model_name in sorted(model_records.keys()):
        results = analyze_error_types(model_records[model_name])
        total = results['total_errors']
        if total > 0:
            ate = results['action_type_errors']
            aoe = results['argument_only_errors']
            ate_correct = results['action_type_with_correct_args']
            output_lines.append(f"\n{model_name}: {total} errors")
            output_lines.append(f"  Action type errors: {ate} ({ate/total*100:.1f}%) - "
                              f"correct args: {ate_correct}, wrong args: {results['action_type_with_wrong_args']}")
            output_lines.append(f"  Argument only errors: {aoe} ({aoe/total*100:.1f}%)")
            # Substitution patterns
            if results['substitution_counts']:
                output_lines.append("  Substitutions:")
                for key, counts in sorted(results['substitution_counts'].items()):
                    output_lines.append(f"    {key}: {counts['total']}")

    # By Scenario (Ask/Tell only)
    output_lines.append("\n" + "-" * 80)
    output_lines.append("BY SCENARIO (Ask/Tell scenarios only)")
    output_lines.append("-" * 80)
    scenario_ids = sorted(set(r.get('scenario_id') for r in all_records if r.get('scenario_id')),
                          key=lambda x: int(x) if x and x.isdigit() else 0)
    for sid in scenario_ids:
        scenario_records = [r for r in all_records if r.get('scenario_id') == sid]
        if not scenario_records:
            continue
        exp_type = normalize_action(scenario_records[0].get('optimal_action', ''))
        if exp_type not in ['Ask', 'Tell']:
            continue

        results = analyze_error_types(scenario_records)
        total = results['total_errors']
        if total > 0:
            ate = results['action_type_errors']
            aoe = results['argument_only_errors']
            output_lines.append(f"  Scenario {sid} ({exp_type}): {total} errors - "
                              f"type: {ate} ({ate/total*100:.0f}%), args: {aoe} ({aoe/total*100:.0f}%)")

    # ========== MASTERY CATEGORY ANALYSIS (existing) ==========
    output_lines.append("\n" + "=" * 100)
    output_lines.append("MASTERY CATEGORY ANALYSIS")
    output_lines.append("=" * 100)

    for cat_key, category in TOM_MASTERY_CATEGORIES.items():
        output_lines.append(f"\n--- {category['name']} ---")
        output_lines.append(f"Description: {category['description']}")

        for comp in category['components']:
            analysis = analyze_category_component(all_records, comp, category['name'])
            if not analysis:
                continue

            scenarios_str = ', '.join(str(s) for s in analysis['scenarios'])
            output_lines.append(f"\n  Scenarios [{scenarios_str}] → {analysis['expected_action']}")
            output_lines.append(f"  Success: {analysis['n_success']}/{analysis['n_total']} = {analysis['success_rate']*100:.1f}%")

            if analysis['n_failure'] > 0:
                output_lines.append(f"  Failures chose: {dict(analysis['failure_actions'])}")

    # Save to single file
    output_path = os.path.join(logs_dir, 'error_analysis.txt')
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))
    print(f"\nAnalysis saved to: {output_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, 'tom_llm_logs')

    if not os.path.exists(logs_dir):
        print(f"Error: {logs_dir} not found")
        return

    # Find all game data files
    pattern = os.path.join(logs_dir, '*_game_data.json')
    files = glob.glob(pattern)
    print(f"Found {len(files)} game data files")

    # Load all records
    model_records, all_records = load_all_records(files)
    print(f"Loaded {len(all_records)} records from {len(model_records)} models")
    print(f"Models: {', '.join(sorted(model_records.keys()))}")

    if not all_records:
        print("No records found")
        return

    # Generate single combined analysis file
    generate_combined_analysis(model_records, all_records, logs_dir)

    # Generate single CSV summary
    generate_error_type_summary_csv(model_records, logs_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
