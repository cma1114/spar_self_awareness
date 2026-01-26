#!/usr/bin/env python3
"""
Surface Heuristic Analysis for ToM Test Scenarios.

Analyzes whether certain events or event patterns can predict the correct action,
enabling models to succeed via "shortcut" heuristics rather than true epistemic reasoning.

Analysis includes:
1. Single event presence (has_put, you_leaves, etc.)
2. Positional patterns (first/last event type)
3. Character-relative order (you_leave_before_b, etc.)
4. Consecutive event pairs (put→leave, leave→move, etc.)
5. Container-relative features (put_to_queried, etc.)
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class FeatureStats:
    """Statistics for a single feature's predictive power."""
    feature_name: str
    total_with_feature: int
    action_counts: Dict[str, int]  # action -> count
    action_percentages: Dict[str, float]  # action -> percentage
    max_action: str  # Most common action when feature is present
    max_percentage: float  # Percentage of max action


def load_scenarios(filepath: str) -> Tuple[List[dict], List[str]]:
    """Load scenarios from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    scenarios = data.get('scenarios', data)  # Handle both old and new format
    chars = data.get('chars', ['A', 'B', 'C', 'D'])
    return scenarios, chars


def extract_features(scenario: dict, player: str = 'A', teammate: str = 'B') -> Dict[str, Any]:
    """Extract all heuristic features from a scenario."""
    events = scenario.get('events', [])
    correct_action = scenario.get('correct_action', 'Pass')
    queried = scenario.get('question_container', 'bag')
    extra = scenario.get('extra', 0)

    features = {
        'correct_action': correct_action,
        'extra': extra,
        'queried_container': queried,
        'num_events': len(events),
    }

    # Event type counts
    event_types = ['put', 'move', 'leave', 'enter', 'remove']
    for et in event_types:
        features[f'num_{et}'] = sum(1 for e in events if e.get('event_type') == et)
        features[f'has_{et}'] = features[f'num_{et}'] > 0

    # First and last event types
    if events:
        features['first_event_type'] = events[0].get('event_type')
        features['last_event_type'] = events[-1].get('event_type')
        for et in event_types:
            features[f'first_is_{et}'] = features['first_event_type'] == et
            features[f'last_is_{et}'] = features['last_event_type'] == et
    else:
        features['first_event_type'] = None
        features['last_event_type'] = None
        for et in event_types:
            features[f'first_is_{et}'] = False
            features[f'last_is_{et}'] = False

    # Character-specific events
    features['you_leaves'] = any(
        e.get('event_type') == 'leave' and e.get('character') == player
        for e in events
    )
    features['you_enters'] = any(
        e.get('event_type') == 'enter' and e.get('character') == player
        for e in events
    )
    features['you_puts'] = any(
        e.get('event_type') == 'put' and e.get('character') == player
        for e in events
    )
    features['you_moves'] = any(
        e.get('event_type') == 'move' and e.get('character') == player
        for e in events
    )
    features['b_leaves'] = any(
        e.get('event_type') == 'leave' and e.get('character') == teammate
        for e in events
    )
    features['b_enters'] = any(
        e.get('event_type') == 'enter' and e.get('character') == teammate
        for e in events
    )
    features['b_puts'] = any(
        e.get('event_type') == 'put' and e.get('character') == teammate
        for e in events
    )
    features['b_moves'] = any(
        e.get('event_type') == 'move' and e.get('character') == teammate
        for e in events
    )
    features['b_put_or_moved'] = features['b_puts'] or features['b_moves']

    # Find positions
    you_leave_pos = -1
    b_leave_pos = -1
    first_put_pos = -1

    for i, e in enumerate(events):
        et = e.get('event_type')
        char = e.get('character')

        if et == 'leave' and char == player and you_leave_pos == -1:
            you_leave_pos = i
        if et == 'leave' and char == teammate and b_leave_pos == -1:
            b_leave_pos = i
        if et == 'put' and first_put_pos == -1:
            first_put_pos = i

    features['you_leave_position'] = you_leave_pos
    features['b_leave_position'] = b_leave_pos
    features['first_put_position'] = first_put_pos

    # Relative order features
    features['you_leave_before_b'] = (
        you_leave_pos != -1 and
        (b_leave_pos == -1 or you_leave_pos < b_leave_pos)
    )
    features['b_leave_before_you'] = (
        b_leave_pos != -1 and
        (you_leave_pos == -1 or b_leave_pos < you_leave_pos)
    )
    features['you_leave_before_any_put'] = (
        you_leave_pos != -1 and
        (first_put_pos == -1 or you_leave_pos < first_put_pos)
    )
    features['you_leave_after_any_put'] = (
        you_leave_pos != -1 and
        first_put_pos != -1 and
        you_leave_pos > first_put_pos
    )

    # You leave and re-enter (complex scenario pattern)
    features['you_leave_then_enter'] = features['you_leaves'] and features['you_enters']

    # Container-relative features
    features['put_to_queried'] = any(
        e.get('event_type') == 'put' and e.get('container') == queried
        for e in events
    )
    features['put_to_other'] = any(
        e.get('event_type') == 'put' and e.get('container') != queried
        for e in events
    )
    features['move_from_queried'] = any(
        e.get('event_type') == 'move' and e.get('from_container') == queried
        for e in events
    )
    features['move_to_queried'] = any(
        e.get('event_type') == 'move' and e.get('to_container') == queried
        for e in events
    )

    # Consecutive event pairs
    pairs = []
    for i in range(len(events) - 1):
        t1 = events[i].get('event_type')
        t2 = events[i+1].get('event_type')
        pairs.append((t1, t2))

    features['event_pairs'] = pairs

    # Common pair features
    common_pairs = [
        ('put', 'leave'), ('leave', 'put'), ('leave', 'move'), ('move', 'leave'),
        ('put', 'move'), ('move', 'put'), ('leave', 'leave'), ('move', 'move'),
        ('leave', 'enter'), ('enter', 'move'), ('enter', 'put')
    ]
    for p1, p2 in common_pairs:
        features[f'has_pair_{p1}_{p2}'] = (p1, p2) in pairs

    # Character-specific pairs
    features['you_put_then_leave'] = False
    features['you_leave_then_enter'] = False
    features['b_put_then_leave'] = False

    for i in range(len(events) - 1):
        e1, e2 = events[i], events[i+1]
        t1, c1 = e1.get('event_type'), e1.get('character')
        t2, c2 = e2.get('event_type'), e2.get('character')

        if t1 == 'put' and c1 == player and t2 == 'leave' and c2 == player:
            features['you_put_then_leave'] = True
        if t1 == 'leave' and c1 == player and t2 == 'enter' and c2 == player:
            features['you_leave_then_enter'] = True
        if t1 == 'put' and c1 == teammate and t2 == 'leave' and c2 == teammate:
            features['b_put_then_leave'] = True

    # Epistemic state features (from scenario metadata)
    features['ks_self'] = scenario.get('ks_self', '')
    features['ks_teammate'] = scenario.get('ks_teammate', '')
    features['ks_opponent'] = scenario.get('ks_opponent', '')

    return features


def analyze_feature(scenarios_features: List[dict], feature_name: str) -> Optional[FeatureStats]:
    """Analyze predictive power of a single feature."""
    # Filter scenarios where feature is True (for boolean features)
    # or not None/empty (for other features)
    matching = []
    for sf in scenarios_features:
        val = sf.get(feature_name)
        if isinstance(val, bool):
            if val:
                matching.append(sf)
        elif val is not None and val != '' and val != -1:
            matching.append(sf)

    if len(matching) == 0:
        return None

    # Count actions
    action_counts = defaultdict(int)
    for sf in matching:
        action = sf.get('correct_action', 'Pass')
        action_counts[action] += 1

    total = len(matching)
    action_percentages = {a: (c / total) * 100 for a, c in action_counts.items()}

    max_action = max(action_counts.keys(), key=lambda a: action_counts[a])
    max_percentage = action_percentages[max_action]

    return FeatureStats(
        feature_name=feature_name,
        total_with_feature=total,
        action_counts=dict(action_counts),
        action_percentages=action_percentages,
        max_action=max_action,
        max_percentage=max_percentage
    )


def analyze_pair_feature(scenarios_features: List[dict], pair: Tuple[str, str]) -> Optional[FeatureStats]:
    """Analyze predictive power of an event pair."""
    matching = [sf for sf in scenarios_features if pair in sf.get('event_pairs', [])]

    if len(matching) == 0:
        return None

    action_counts = defaultdict(int)
    for sf in matching:
        action = sf.get('correct_action', 'Pass')
        action_counts[action] += 1

    total = len(matching)
    action_percentages = {a: (c / total) * 100 for a, c in action_counts.items()}

    max_action = max(action_counts.keys(), key=lambda a: action_counts[a])
    max_percentage = action_percentages[max_action]

    return FeatureStats(
        feature_name=f'pair_{pair[0]}_{pair[1]}',
        total_with_feature=total,
        action_counts=dict(action_counts),
        action_percentages=action_percentages,
        max_action=max_action,
        max_percentage=max_percentage
    )


def analyze_minority_predictors(scenarios_features: List[dict], target_action: str,
                                bool_features: List[str]) -> List[Tuple[str, float, int, int, float]]:
    """Find features that predict a specific action better than baseline.

    Returns list of (feature_name, lift, count_with_target, count_total, percentage)
    sorted by lift descending.
    """
    total = len(scenarios_features)
    baseline_count = sum(1 for sf in scenarios_features if sf.get('correct_action') == target_action)
    if baseline_count == 0 or total == 0:
        return []
    baseline_pct = baseline_count / total

    results = []
    for fname in bool_features:
        matching = [sf for sf in scenarios_features if sf.get(fname, False)]
        if len(matching) < 2:  # Need at least 2 samples
            continue

        target_count = sum(1 for sf in matching if sf.get('correct_action') == target_action)
        if target_count == 0:
            continue

        feature_pct = target_count / len(matching)
        lift = feature_pct / baseline_pct if baseline_pct > 0 else 0

        results.append((fname, lift, target_count, len(matching), feature_pct * 100))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def generate_report(scenarios_features: List[dict], title: str) -> str:
    """Generate analysis report for a set of scenarios."""
    lines = []
    lines.append("=" * 80)
    lines.append(title)
    lines.append("=" * 80)
    lines.append(f"\nTotal scenarios: {len(scenarios_features)}")

    # Overall action distribution
    action_dist = defaultdict(int)
    for sf in scenarios_features:
        action_dist[sf.get('correct_action', 'Pass')] += 1

    lines.append("\nOverall correct action distribution:")
    for action, count in sorted(action_dist.items()):
        pct = count / len(scenarios_features) * 100
        lines.append(f"  {action}: {count} ({pct:.1f}%)")

    # Boolean features to analyze
    bool_features = [
        'has_put', 'has_move', 'has_leave', 'has_enter', 'has_remove',
        'first_is_put', 'first_is_leave', 'first_is_move',
        'last_is_put', 'last_is_leave', 'last_is_move',
        'you_leaves', 'you_enters', 'you_puts', 'you_moves',
        'b_leaves', 'b_enters', 'b_puts', 'b_moves', 'b_put_or_moved',
        'you_leave_before_b', 'b_leave_before_you',
        'you_leave_before_any_put', 'you_leave_after_any_put',
        'you_leave_then_enter',
        'put_to_queried', 'put_to_other', 'move_from_queried', 'move_to_queried',
        'has_pair_put_leave', 'has_pair_leave_put', 'has_pair_leave_move',
        'has_pair_move_leave', 'has_pair_put_move', 'has_pair_move_put',
        'has_pair_leave_leave', 'has_pair_move_move', 'has_pair_leave_enter',
        'has_pair_enter_move', 'has_pair_enter_put',
        'you_put_then_leave', 'b_put_then_leave',
    ]

    # Analyze each feature
    feature_stats = []
    for fname in bool_features:
        stats = analyze_feature(scenarios_features, fname)
        if stats and stats.total_with_feature > 0:
            feature_stats.append(stats)

    # Sort by predictive power (max_percentage descending)
    feature_stats.sort(key=lambda s: s.max_percentage, reverse=True)

    # Report perfect heuristics (>=95%)
    perfect = [s for s in feature_stats if s.max_percentage >= 95]
    if perfect:
        lines.append("\n" + "=" * 80)
        lines.append("PERFECT/NEAR-PERFECT HEURISTICS (>=95%)")
        lines.append("=" * 80)
        for s in perfect:
            lines.append(f"\n{s.feature_name}:")
            lines.append(f"  Scenarios with feature: {s.total_with_feature}")
            lines.append(f"  Predicts: {s.max_action} ({s.max_percentage:.1f}%)")
            for action, pct in sorted(s.action_percentages.items()):
                lines.append(f"    {action}: {s.action_counts[action]} ({pct:.1f}%)")

    # Report strong heuristics (80-95%)
    strong = [s for s in feature_stats if 80 <= s.max_percentage < 95]
    if strong:
        lines.append("\n" + "=" * 80)
        lines.append("STRONG HEURISTICS (80-95%)")
        lines.append("=" * 80)
        for s in strong:
            lines.append(f"\n{s.feature_name}:")
            lines.append(f"  Scenarios with feature: {s.total_with_feature}")
            lines.append(f"  Predicts: {s.max_action} ({s.max_percentage:.1f}%)")
            for action, pct in sorted(s.action_percentages.items()):
                lines.append(f"    {action}: {s.action_counts[action]} ({pct:.1f}%)")

    # Report moderate heuristics (60-80%)
    moderate = [s for s in feature_stats if 60 <= s.max_percentage < 80]
    if moderate:
        lines.append("\n" + "=" * 80)
        lines.append("MODERATE HEURISTICS (60-80%)")
        lines.append("=" * 80)
        for s in moderate:
            lines.append(f"\n{s.feature_name}:")
            lines.append(f"  Scenarios with feature: {s.total_with_feature}")
            lines.append(f"  Predicts: {s.max_action} ({s.max_percentage:.1f}%)")

    # Analyze all unique event pairs
    lines.append("\n" + "=" * 80)
    lines.append("EVENT PAIR ANALYSIS")
    lines.append("=" * 80)

    all_pairs = set()
    for sf in scenarios_features:
        all_pairs.update(sf.get('event_pairs', []))

    pair_stats = []
    for pair in all_pairs:
        stats = analyze_pair_feature(scenarios_features, pair)
        if stats and stats.total_with_feature >= 3:  # At least 3 occurrences
            pair_stats.append(stats)

    pair_stats.sort(key=lambda s: s.max_percentage, reverse=True)

    for s in pair_stats[:20]:  # Top 20 pairs
        lines.append(f"\n{s.feature_name.replace('pair_', '')}:")
        lines.append(f"  Occurrences: {s.total_with_feature}")
        lines.append(f"  Best predictor: {s.max_action} ({s.max_percentage:.1f}%)")

    # Feature correlation with epistemic states
    lines.append("\n" + "=" * 80)
    lines.append("EPISTEMIC STATE CORRELATION")
    lines.append("=" * 80)

    for ks_field in ['ks_self', 'ks_teammate', 'ks_opponent']:
        lines.append(f"\n{ks_field}:")
        ks_values = defaultdict(lambda: defaultdict(int))
        for sf in scenarios_features:
            ks_val = sf.get(ks_field, '')
            action = sf.get('correct_action', 'Pass')
            if ks_val:
                ks_values[ks_val][action] += 1

        for ks_val, actions in sorted(ks_values.items()):
            total = sum(actions.values())
            lines.append(f"  {ks_val}: (N={total})")
            for action, count in sorted(actions.items()):
                pct = count / total * 100
                lines.append(f"    {action}: {count} ({pct:.1f}%)")

    # Analyze minority class predictors (Ask and Tell)
    lines.append("\n" + "=" * 80)
    lines.append("MINORITY CLASS PREDICTORS (Lift over baseline)")
    lines.append("=" * 80)

    lines.append("\nBaseline rates:")
    total = len(scenarios_features)
    for action in ['Ask Teammate', 'Tell Teammate']:
        count = sum(1 for sf in scenarios_features if sf.get('correct_action') == action)
        pct = count / total * 100 if total > 0 else 0
        lines.append(f"  {action}: {count}/{total} ({pct:.1f}%)")

    for target_action in ['Ask Teammate', 'Tell Teammate']:
        lines.append(f"\nFeatures that predict '{target_action}':")

        predictors = analyze_minority_predictors(scenarios_features, target_action, bool_features)
        if not predictors:
            lines.append("  (no predictors found)")
        else:
            for fname, lift, count, total_feat, pct in predictors[:10]:
                lines.append(f"  {fname}: {count}/{total_feat} ({pct:.1f}%) [lift={lift:.2f}x]")

    # Key insight section
    lines.append("\n" + "=" * 80)
    lines.append("KEY INSIGHTS")
    lines.append("=" * 80)

    # Check if any surface feature perfectly predicts a minority class
    ask_predictors = analyze_minority_predictors(scenarios_features, 'Ask Teammate', bool_features)
    tell_predictors = analyze_minority_predictors(scenarios_features, 'Tell Teammate', bool_features)

    perfect_ask = [p for p in ask_predictors if p[4] >= 95]  # 95%+ accuracy
    perfect_tell = [p for p in tell_predictors if p[4] >= 95]

    if perfect_ask:
        lines.append(f"\nPerfect predictors for 'Ask Teammate': {len(perfect_ask)}")
        for fname, lift, count, total_feat, pct in perfect_ask:
            lines.append(f"  {fname}: {pct:.1f}% (N={total_feat})")
    else:
        lines.append("\nNo surface features perfectly predict 'Ask Teammate'")

    if perfect_tell:
        lines.append(f"\nPerfect predictors for 'Tell Teammate': {len(perfect_tell)}")
        for fname, lift, count, total_feat, pct in perfect_tell:
            lines.append(f"  {fname}: {pct:.1f}% (N={total_feat})")
    else:
        lines.append("\nNo surface features perfectly predict 'Tell Teammate'")

    # Baseline gaming analysis
    baseline_pass_pct = sum(1 for sf in scenarios_features if sf.get('correct_action') == 'Pass') / total * 100
    lines.append(f"\nBaseline 'always Pass' accuracy: {baseline_pass_pct:.1f}%")
    lines.append("(A model that always says 'Pass' would achieve this accuracy)")

    return "\n".join(lines)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scenarios_file = os.path.join(script_dir, 'scenarios_generated4.json')
    logs_dir = os.path.join(script_dir, 'tom_llm_logs')

    if not os.path.exists(scenarios_file):
        print(f"Error: {scenarios_file} not found")
        return

    os.makedirs(logs_dir, exist_ok=True)

    # Load scenarios
    scenarios, chars = load_scenarios(scenarios_file)
    print(f"Loaded {len(scenarios)} scenarios")

    # Extract features for all scenarios
    all_features = [extract_features(s) for s in scenarios]

    # Split by Extra value
    extra0_features = [f for f in all_features if f.get('extra', 0) == 0 or f.get('extra') is None]
    extra1_features = [f for f in all_features if f.get('extra') == 1]

    print(f"Extra=0 scenarios: {len(extra0_features)}")
    print(f"Extra=1 scenarios: {len(extra1_features)}")

    # Generate reports
    report_all = generate_report(all_features, "HEURISTIC ANALYSIS - ALL SCENARIOS")
    report_extra0 = generate_report(extra0_features, "HEURISTIC ANALYSIS - EXTRA=0 (BASE)")
    report_extra1 = generate_report(extra1_features, "HEURISTIC ANALYSIS - EXTRA=1 (COMPLEX)")

    # Combine into single report
    full_report = report_all + "\n\n" + "=" * 80 + "\n" * 3 + report_extra0 + "\n\n" + "=" * 80 + "\n" * 3 + report_extra1

    # Save reports
    output_path = os.path.join(logs_dir, 'heuristic_analysis.txt')
    with open(output_path, 'w') as f:
        f.write(full_report)

    print(f"\nReport saved to: {output_path}")

    # Print summary to console
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Find perfect heuristics
    for label, features in [("ALL", all_features), ("Extra=0", extra0_features), ("Extra=1", extra1_features)]:
        bool_features = [
            'has_put', 'has_move', 'has_leave', 'has_enter',
            'you_leaves', 'you_enters', 'b_leaves', 'b_put_or_moved',
            'you_leave_before_b', 'you_leave_after_any_put',
            'you_leave_then_enter', 'put_to_queried',
        ]

        perfect_count = 0
        strong_count = 0
        for fname in bool_features:
            stats = analyze_feature(features, fname)
            if stats:
                if stats.max_percentage >= 95:
                    perfect_count += 1
                elif stats.max_percentage >= 80:
                    strong_count += 1

        print(f"\n{label}:")
        print(f"  Perfect heuristics (>=95%): {perfect_count}")
        print(f"  Strong heuristics (80-95%): {strong_count}")


if __name__ == '__main__':
    main()
