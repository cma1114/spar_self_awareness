#!/usr/bin/env python3
"""Pre-generate all scenarios for standardized testing.

This script generates all scenarios (26 IDs × 4 Extra × n reps) to a single JSON file.
All models should use the same pre-generated file to ensure fair comparison in research.

Usage:
    python pre_generate_scenarios.py --reps 5 --seed 42 --output experiments/scenarios_v1.json

The output file can then be used with tom_test_new.py:
    python tom_test_new.py --model gpt-4o --scenario_file experiments/scenarios_v1.json
"""

import argparse
import tempfile
import os
from tom_helpers import read_specs_from_csv, load_scenarios, save_scenarios, CharacterType
from generate_tom_scenarios_new import generate_scenarios_from_tuples

# Import MASTERY_SCENARIO_IDS from tom_test_new
from tom_test_new import MASTERY_SCENARIO_IDS


def pre_generate(num_reps: int, seed_base: int, output_file: str):
    """Generate all scenarios (specs × reps) to a single file.

    Args:
        num_reps: Number of repetitions
        seed_base: Base seed for reproducibility
        output_file: Path to output JSON file
    """
    # Read all specs and filter to mastery scenarios only
    specs = read_specs_from_csv('ToM - scenarios.csv')
    specs = [s for s in specs if int(s['Id']) in MASTERY_SCENARIO_IDS]

    # Group specs by scenario ID so all Extra variants (0A, 0B, 1A, 1B) are
    # generated together in a single call to generate_scenarios_from_tuples.
    # This is REQUIRED for the adaptive filler logic to work: it needs both
    # 1A and 1B in the same call to balance their SIT event counts.
    from collections import defaultdict
    specs_by_id = defaultdict(list)
    for spec in specs:
        specs_by_id[spec['Id']].append(spec)
    sorted_ids = sorted(specs_by_id.keys(), key=int)

    total_specs = len(specs)
    print(f"Generating scenarios for {total_specs} specs × {num_reps} reps = {total_specs * num_reps} total")
    print(f"  ({len(sorted_ids)} scenario IDs, grouped for SIT balancing)")
    print(f"Base seed: {seed_base}")

    # Character types (matches tom_test_new.py)
    chartypes = [
        CharacterType.LIVE_PLAYER,
        CharacterType.HONEST_OPPONENT,
        CharacterType.DISHONEST_TEAMMATE,
        CharacterType.DISHONEST_OPPONENT,
    ]

    all_scenarios = []
    chars = None
    ctypes = None

    # Aggregate stats across all calls
    total_violations = 0
    violations_by_id = defaultdict(int)
    violations_by_extra = defaultdict(int)
    total_sit_gaps = 0
    sit_gaps_by_id = {}

    # Create temp file for intermediate generation
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        for rep in range(num_reps):
            rep_1indexed = rep + 1  # 1-indexed rep number for display and storage
            print(f"  Generating rep {rep_1indexed}/{num_reps}...")
            for id_idx, scenario_id in enumerate(sorted_ids):
                id_specs = specs_by_id[scenario_id]
                # Seed: one per (ID, rep) combination
                seed = seed_base + rep_1indexed * 1000 + id_idx

                stats = generate_scenarios_from_tuples(id_specs, tmp_path, seed=seed, chartypes=chartypes)
                scenarios, chars, ctypes = load_scenarios(tmp_path)

                # Aggregate violation stats
                if stats:
                    v = stats['violations']
                    total_violations += v['total']
                    for sid, count in v['by_id'].items():
                        violations_by_id[sid] += count
                    for extra, count in v['by_extra'].items():
                        violations_by_extra[extra] += count
                    sg = stats['sit_gaps']
                    total_sit_gaps += sg['total']
                    for sid, info in sg['by_id'].items():
                        if sid not in sit_gaps_by_id:
                            sit_gaps_by_id[sid] = {'count': 0, 'max_gap': 0, 'sits_1a': [], 'sits_1b': []}
                        sit_gaps_by_id[sid]['count'] += info.get('count', 1)
                        sit_gaps_by_id[sid]['max_gap'] = max(sit_gaps_by_id[sid]['max_gap'], info.get('max_gap', 0))
                        sit_gaps_by_id[sid]['sits_1a'].extend(info.get('sits_1a', []))
                        sit_gaps_by_id[sid]['sits_1b'].extend(info.get('sits_1b', []))

                # Tag each scenario with rep number (1-indexed to match user interface)
                for s in scenarios:
                    s.rep = rep_1indexed

                all_scenarios.extend(scenarios)
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save all scenarios to output file
    save_scenarios(all_scenarios, output_file, chars, ctypes)

    print(f"\nGenerated {len(all_scenarios)} scenarios to {output_file}")
    print(f"  {len(specs)} specs × {num_reps} reps = {len(specs) * num_reps} expected")

    # Verify
    if len(all_scenarios) != len(specs) * num_reps:
        print(f"  WARNING: Expected {len(specs) * num_reps}, got {len(all_scenarios)}")

    # Show breakdown by Extra
    extra_counts = {}
    for s in all_scenarios:
        extra = s.extra or 'unknown'
        extra_counts[extra] = extra_counts.get(extra, 0) + 1
    print(f"  By Extra: {extra_counts}")

    # Print violation summary
    print(f"\n=== Violation Summary ===")
    print(f"Teammate belief violations: {total_violations}")
    if violations_by_id:
        print(f"  By ID:")
        for sid in sorted(violations_by_id.keys(), key=int):
            print(f"    {sid}: {violations_by_id[sid]}")
    if violations_by_extra:
        print(f"  By Extra:")
        for extra in ['0A', '0B', '1A', '1B']:
            if extra in violations_by_extra:
                print(f"    {extra}: {violations_by_extra[extra]}")
    print(f"SIT gap violations (>3): {total_sit_gaps}")
    if sit_gaps_by_id:
        print(f"  ID | count | max_gap | 1A range   | 1B range")
        print(f"  ---|-------|---------|------------|----------")
        for sid in sorted(sit_gaps_by_id.keys(), key=int):
            info = sit_gaps_by_id[sid]
            sits_1a = info.get('sits_1a', [])
            sits_1b = info.get('sits_1b', [])
            range_1a = f"{min(sits_1a)}-{max(sits_1a)}" if sits_1a else "?"
            range_1b = f"{min(sits_1b)}-{max(sits_1b)}" if sits_1b else "?"
            print(f"  {sid:>2} | {info['count']:>5} | {info['max_gap']:>7} | {range_1a:>10} | {range_1b:>8}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-generate scenarios for standardized testing.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--reps',
        type=int,
        default=10,
        help='Number of repetitions (default: 10)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Base seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='scenarios_standardized.json',
        help='Output file path (default: scenarios_standardized.json)'
    )
    args = parser.parse_args()

    pre_generate(args.reps, args.seed, args.output)