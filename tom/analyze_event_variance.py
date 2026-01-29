#!/usr/bin/env python3
"""Analyze event count variance across scenario generations."""

import sys
import os
import warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress warnings during generation
warnings.filterwarnings('ignore')

from tom_helpers import read_specs_from_csv, Scenario
from generate_tom_scenarios_new import generate_scenarios_from_tuples, _count_visible_events, count_epistemic_category_transitions
import tempfile
import json
from collections import defaultdict, Counter
import statistics

def analyze_variance(num_seeds=100, output_file='tom/sit_variance_results.txt'):
    """Generate scenarios with different seeds and analyze variance."""
    specs = read_specs_from_csv('tom/ToM - scenarios.csv')

    # Group specs by (Id, Extra)
    spec_lookup = {}
    for spec in specs:
        key = (spec['Id'], spec['Extra'])
        spec_lookup[key] = spec

    # Get unique IDs
    unique_ids = sorted(set(spec['Id'] for spec in specs), key=lambda x: int(x))

    results = []

    print(f"Analyzing {len(unique_ids)} scenario IDs x 2 (Extra=0,1) x {num_seeds} seeds...")

    for scenario_id in unique_ids:
        for extra in [0, 1]:
            key = (scenario_id, extra)
            if key not in spec_lookup:
                continue

            spec = spec_lookup[key]
            sit_counts = []

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                tmpfile = f.name

            try:
                for seed in range(num_seeds):
                    generate_scenarios_from_tuples([spec], outfile=tmpfile, seed=seed)

                    with open(tmpfile, 'r') as f:
                        data = json.load(f)

                    if data.get('scenarios'):
                        scenario_data = data['scenarios'][0]
                        sit_counts.append(scenario_data.get('situation_event_count', 0))
            finally:
                os.unlink(tmpfile)

            if sit_counts:
                results.append({
                    'id': scenario_id,
                    'extra': extra,
                    'sit': sit_counts,
                    'distribution': Counter(sit_counts),
                })
                print(f"  ID {scenario_id}, Extra={extra}: done")

    # Write detailed output file
    with open(output_file, 'w') as f:
        f.write(f"SIT (Situation Tracking) Variance Analysis\n")
        f.write(f"Seeds per scenario: {num_seeds}\n")
        f.write(f"=" * 70 + "\n\n")

        for r in results:
            f.write(f"Scenario ID={r['id']}, Extra={r['extra']}\n")
            f.write("-" * 40 + "\n")

            dist = r['distribution']
            sit_values = sorted(dist.keys())

            if len(sit_values) == 1:
                f.write(f"  FIXED: Always {sit_values[0]} visible events\n")
            else:
                f.write(f"  Range: {min(sit_values)} - {max(sit_values)}\n")
                f.write(f"  Mean:  {statistics.mean(r['sit']):.2f}\n")
                f.write(f"  Stdev: {statistics.stdev(r['sit']):.2f}\n")
                f.write(f"\n  Distribution:\n")
                for sit_val in sit_values:
                    count = dist[sit_val]
                    pct = 100 * count / num_seeds
                    bar = '#' * int(pct / 2)
                    f.write(f"    SIT={sit_val:2d}: {count:3d} ({pct:5.1f}%) {bar}\n")
            f.write("\n")

        # Summary table
        f.write("=" * 70 + "\n")
        f.write("SUMMARY TABLE\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'ID':>3} {'E':>1} {'Min':>4} {'Max':>4} {'Mean':>6} {'Stdev':>6} {'Distribution'}\n")
        f.write("-" * 70 + "\n")

        for r in results:
            dist = r['distribution']
            sit_values = sorted(dist.keys())
            mn = min(sit_values)
            mx = max(sit_values)
            mean = statistics.mean(r['sit'])
            std = statistics.stdev(r['sit']) if len(r['sit']) > 1 else 0

            # Compact distribution string
            dist_str = ', '.join(f"{v}:{dist[v]}" for v in sit_values)

            f.write(f"{r['id']:>3} {r['extra']:>1} {mn:>4} {mx:>4} {mean:>6.1f} {std:>6.2f}   {dist_str}\n")

    print(f"\nResults written to: {output_file}")
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=100, help='Number of seeds to test')
    parser.add_argument('--output', type=str, default='tom/sit_variance_results.txt', help='Output file')
    args = parser.parse_args()
    analyze_variance(args.seeds, args.output)
