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

    # Reorder specs to match tom_test_new.py ordering: ID7_0A, ID7_0B, ID7_1A, ID7_1B, ID8_0A, ...
    from collections import defaultdict
    spec_by_id = defaultdict(dict)
    for spec in specs:
        spec_by_id[spec['Id']][spec['Extra']] = spec
    reordered_specs = []
    for id_str in sorted(spec_by_id.keys(), key=int):
        for extra_val in ['0A', '0B', '1A', '1B']:
            if extra_val in spec_by_id[id_str]:
                reordered_specs.append(spec_by_id[id_str][extra_val])
    specs = reordered_specs

    print(f"Generating scenarios for {len(specs)} specs × {num_reps} reps = {len(specs) * num_reps} total")
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

    # Create temp file for intermediate generation
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        for rep in range(num_reps):
            rep_1indexed = rep + 1  # 1-indexed rep number for display and storage
            print(f"  Generating rep {rep_1indexed}/{num_reps}...")
            for spec_idx, spec in enumerate(specs):
                # Seed formula matches tom_test_new.py multi-rep logic:
                # base_seed + rep_num * 1000 + spec_idx
                # Note: tom_test_new.py uses 1-indexed rep_num in seed calculation
                seed = seed_base + rep_1indexed * 1000 + spec_idx

                generate_scenarios_from_tuples([spec], tmp_path, seed=seed, chartypes=chartypes)
                scenarios, chars, ctypes = load_scenarios(tmp_path)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-generate scenarios for standardized testing.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--reps',
        type=int,
        default=5,
        help='Number of repetitions (default: 5)'
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
