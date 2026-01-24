#!/usr/bin/env python3
"""
Error analysis for ToM mastery categories.

Analyzes WHY models fail on specific categories by examining:
1. Action confusion (what action did they take instead?)
2. Feature correlations (what scenario features correlate with failure?)
3. Concrete examples (what do failed scenarios look like?)

Focuses on Extra=0 scenarios only.
"""

import json
import glob
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

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
        return json.load(f)


def filter_extra0_player_a(records: List[dict]) -> List[dict]:
    """Filter to only Extra=0, player A records."""
    return [r for r in records
            if r.get('character') == 'A'
            and (r.get('extra') is None or r.get('extra') == 0)]


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
    print(f"Loaded {len(all_records)} Extra=0 records from {len(model_records)} models")
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

    # Print to console
    output_text = '\n'.join(output_lines)
    print(output_text)

    # Save to file
    output_path = os.path.join(logs_dir, output_filename)
    with open(output_path, 'w') as f:
        f.write(output_text)
    print(f"\n\nAnalysis saved to: {output_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, 'tom_llm_logs')

    if not os.path.exists(logs_dir):
        print(f"Error: {logs_dir} not found")
        return

    # Find all game data files
    pattern = os.path.join(logs_dir, '*_game_data.json')
    files = glob.glob(pattern)

    # Load and classify records by thinking vs non-thinking
    thinking_model_records: Dict[str, List[dict]] = defaultdict(list)
    nonthinking_model_records: Dict[str, List[dict]] = defaultdict(list)
    thinking_all_records = []
    nonthinking_all_records = []

    for filepath in files:
        records = load_game_data(filepath)
        filtered = filter_extra0_player_a(records)

        if len(filtered) == 0 or len(filtered) % 39 != 0:
            continue

        model_name = extract_model_name(filepath)

        if is_thinking_model(model_name):
            thinking_model_records[model_name].extend(filtered)
            thinking_all_records.extend(filtered)
        else:
            nonthinking_model_records[model_name].extend(filtered)
            nonthinking_all_records.extend(filtered)

    print(f"Found {len(thinking_model_records)} thinking models, {len(nonthinking_model_records)} non-thinking models")

    # Find models that have BOTH thinking and non-thinking versions
    # Thinking model names end with _think, so strip that to get base name
    thinking_base_names = {name.replace('_think', '') for name in thinking_model_records.keys()}
    nonthinking_names = set(nonthinking_model_records.keys())

    # Models with both versions
    paired_base_names = thinking_base_names & nonthinking_names
    print(f"Models with both thinking and non-thinking versions: {sorted(paired_base_names)}")

    # Filter to only paired models
    paired_thinking_records: Dict[str, List[dict]] = {}
    paired_nonthinking_records: Dict[str, List[dict]] = {}
    paired_thinking_all = []
    paired_nonthinking_all = []

    for base_name in paired_base_names:
        think_name = f"{base_name}_think"
        if think_name in thinking_model_records:
            paired_thinking_records[think_name] = thinking_model_records[think_name]
            paired_thinking_all.extend(thinking_model_records[think_name])
        if base_name in nonthinking_model_records:
            paired_nonthinking_records[base_name] = nonthinking_model_records[base_name]
            paired_nonthinking_all.extend(nonthinking_model_records[base_name])

    # Combine all records for the original combined analysis
    all_model_records: Dict[str, List[dict]] = defaultdict(list)
    all_records = []
    for model_name, records in thinking_model_records.items():
        all_model_records[model_name].extend(records)
        all_records.extend(records)
    for model_name, records in nonthinking_model_records.items():
        all_model_records[model_name].extend(records)
        all_records.extend(records)

    # Generate combined analysis (all models)
    generate_analysis(
        all_model_records,
        all_records,
        "ALL MODELS (Extra=0 only)",
        logs_dir,
        'error_analysis.txt'
    )

    # Generate separate analyses for PAIRED thinking and non-thinking models only
    generate_analysis(
        paired_thinking_records,
        paired_thinking_all,
        "THINKING MODELS (paired only, Extra=0)",
        logs_dir,
        'error_analysis_thinking.txt'
    )

    generate_analysis(
        paired_nonthinking_records,
        paired_nonthinking_all,
        "NON-THINKING MODELS (paired only, Extra=0)",
        logs_dir,
        'error_analysis_nonthinking.txt'
    )


if __name__ == '__main__':
    main()
