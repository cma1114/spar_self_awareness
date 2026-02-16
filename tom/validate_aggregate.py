#!/usr/bin/env python3
"""Aggregate validation for pre-generated scenarios.

This script validates cross-scenario constraints that can only be checked
at the aggregate level (after all scenarios are generated).

Checks:
1. Coverage: All variants (0A, 0B, 1A, 1B) present for each scenario ID
2. ECT ordering: ECT(1B) > ECT(1A) for each scenario ID
3. SIT relationship: 0B should have more events than 0A on average
"""

import sys
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple
from tom_helpers import load_scenarios, Scenario


def check_coverage(scenarios: List[Scenario]) -> Tuple[int, List[str]]:
    """Verify all expected variants were generated.

    Args:
        scenarios: List of all scenarios

    Returns:
        Tuple of (error_count, list of error messages)
    """
    by_id_extra = defaultdict(set)
    by_id_rep = defaultdict(lambda: defaultdict(set))

    for s in scenarios:
        if s.id and s.extra:
            by_id_extra[s.id].add(s.extra)
            if s.rep is not None:
                by_id_rep[s.id][s.rep].add(s.extra)

    errors = []
    expected = {'0A', '0B', '1A', '1B'}

    # Check overall coverage per ID
    for scenario_id, extras in sorted(by_id_extra.items()):
        missing = expected - extras
        if missing:
            errors.append(f"ID={scenario_id}: missing variants {sorted(missing)}")

    # Check per-rep coverage
    for scenario_id, reps in sorted(by_id_rep.items()):
        for rep, extras in sorted(reps.items()):
            missing = expected - extras
            if missing:
                errors.append(f"ID={scenario_id} rep={rep}: missing variants {sorted(missing)}")

    return len(errors), errors


def check_ect_ordering(scenarios: List[Scenario]) -> Tuple[int, List[str]]:
    """Verify ECT(1B) > ECT(1A) for each scenario.

    Args:
        scenarios: List of all scenarios

    Returns:
        Tuple of (error_count, list of error messages)
    """
    # Group by (id, rep)
    by_id_rep = defaultdict(dict)
    for s in scenarios:
        if s.id and s.extra and s.rep is not None and s.epistemic_transitions:
            key = (s.id, s.rep)
            by_id_rep[key][s.extra] = s.epistemic_transitions['total']

    errors = []
    for (scenario_id, rep), ect_by_extra in sorted(by_id_rep.items()):
        if '1A' in ect_by_extra and '1B' in ect_by_extra:
            ect_1a = ect_by_extra['1A']
            ect_1b = ect_by_extra['1B']
            if ect_1b <= ect_1a:
                errors.append(
                    f"ID={scenario_id} rep={rep}: ECT(1B)={ect_1b} not > ECT(1A)={ect_1a}"
                )

    return len(errors), errors


def check_sit_relationship(scenarios: List[Scenario]) -> Tuple[int, List[str], Dict]:
    """Verify SIT relationships between variants.

    Expected: 0B has more events than 0A on average (higher situational load).

    Args:
        scenarios: List of all scenarios

    Returns:
        Tuple of (warning_count, list of warnings, stats dict)
    """
    # Collect SIT values by extra
    sit_by_extra = defaultdict(list)
    for s in scenarios:
        if s.extra and s.situation_event_count is not None:
            sit_by_extra[s.extra].append(s.situation_event_count)

    warnings = []
    stats = {}

    for extra in ['0A', '0B', '1A', '1B']:
        if extra in sit_by_extra:
            values = sit_by_extra[extra]
            avg = sum(values) / len(values) if values else 0
            stats[extra] = {
                'avg': avg,
                'min': min(values) if values else 0,
                'max': max(values) if values else 0,
                'count': len(values)
            }

    # Check 0B > 0A on average
    if '0A' in stats and '0B' in stats:
        if stats['0B']['avg'] <= stats['0A']['avg']:
            warnings.append(
                f"0B avg SIT ({stats['0B']['avg']:.1f}) not > 0A avg SIT ({stats['0A']['avg']:.1f})"
            )

    return len(warnings), warnings, stats


def check_ect_distribution(scenarios: List[Scenario]) -> Dict:
    """Compute ECT distribution statistics.

    Args:
        scenarios: List of all scenarios

    Returns:
        Stats dict with ECT distribution by Extra
    """
    ect_by_extra = defaultdict(list)
    for s in scenarios:
        if s.extra and s.epistemic_transitions:
            ect_by_extra[s.extra].append(s.epistemic_transitions['total'])

    stats = {}
    for extra in ['0A', '0B', '1A', '1B']:
        if extra in ect_by_extra:
            values = ect_by_extra[extra]
            avg = sum(values) / len(values) if values else 0
            stats[extra] = {
                'avg': avg,
                'min': min(values) if values else 0,
                'max': max(values) if values else 0,
                'count': len(values)
            }

    return stats


def validate_aggregate(filepath: str, verbose: bool = False) -> bool:
    """Run all aggregate validations on a scenario file.

    Args:
        filepath: Path to the scenario JSON file
        verbose: Print detailed statistics

    Returns:
        True if all checks pass, False otherwise
    """
    print(f"Loading scenarios from {filepath}...")
    scenarios, _, _ = load_scenarios(filepath)
    print(f"Loaded {len(scenarios)} scenarios")
    print()

    all_passed = True

    # 1. Coverage check
    print("=== Coverage Check ===")
    coverage_errors, coverage_msgs = check_coverage(scenarios)
    if coverage_errors == 0:
        print("PASS: All variants present for each scenario ID")
    else:
        print(f"FAIL: {coverage_errors} coverage errors")
        for msg in coverage_msgs[:10]:
            print(f"  {msg}")
        if coverage_errors > 10:
            print(f"  ... and {coverage_errors - 10} more")
        all_passed = False
    print()

    # 2. ECT ordering check
    print("=== ECT Ordering Check ===")
    ect_errors, ect_msgs = check_ect_ordering(scenarios)
    if ect_errors == 0:
        print("PASS: ECT(1B) > ECT(1A) for all scenarios")
    else:
        print(f"FAIL: {ect_errors} ECT ordering violations")
        for msg in ect_msgs[:10]:
            print(f"  {msg}")
        if ect_errors > 10:
            print(f"  ... and {ect_errors - 10} more")
        all_passed = False
    print()

    # 3. SIT relationship check
    print("=== SIT Relationship Check ===")
    sit_warnings, sit_msgs, sit_stats = check_sit_relationship(scenarios)
    if sit_warnings == 0:
        print("PASS: 0B has higher avg SIT than 0A")
    else:
        print(f"WARNING: {sit_warnings} SIT relationship issues")
        for msg in sit_msgs:
            print(f"  {msg}")

    if verbose or sit_warnings > 0:
        print("SIT Statistics:")
        for extra in ['0A', '0B', '1A', '1B']:
            if extra in sit_stats:
                s = sit_stats[extra]
                print(f"  {extra}: avg={s['avg']:.1f}, min={s['min']}, max={s['max']}, n={s['count']}")
    print()

    # 4. ECT distribution (info only)
    if verbose:
        print("=== ECT Distribution (Info) ===")
        ect_stats = check_ect_distribution(scenarios)
        for extra in ['0A', '0B', '1A', '1B']:
            if extra in ect_stats:
                s = ect_stats[extra]
                print(f"  {extra}: avg={s['avg']:.1f}, min={s['min']}, max={s['max']}, n={s['count']}")
        print()

    # Summary
    print("=" * 50)
    if all_passed:
        print("SUMMARY: All aggregate checks passed")
    else:
        print("SUMMARY: Some checks failed - see details above")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Validate aggregate scenario properties')
    parser.add_argument('--filepath', help='Path to the scenario JSON file', default='scenarios_standardized.json')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed statistics')
    args = parser.parse_args()

    success = validate_aggregate(args.filepath, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
