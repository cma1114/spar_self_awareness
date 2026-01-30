#!/usr/bin/env python3
"""Analyze teammate belief violations to understand which specs cause them."""

import sys
import os
import warnings
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tom_helpers import read_specs_from_csv
from generate_tom_scenarios_new import generate_scenarios_from_tuples
import tempfile

def analyze_violations(num_seeds=50):
    """Generate scenarios and count violations by spec."""
    specs = read_specs_from_csv('tom/ToM - scenarios.csv')

    # Group specs by (Id, Extra)
    spec_lookup = {}
    for spec in specs:
        key = (spec['Id'], spec['Extra'])
        spec_lookup[key] = spec

    unique_ids = sorted(set(spec['Id'] for spec in specs), key=lambda x: int(x))

    # Capture warnings
    violation_counts = defaultdict(lambda: {'count': 0, 'examples': []})

    print(f"Analyzing {len(unique_ids)} scenario IDs x 4 (Extra=0A,0B,1A,1B) x {num_seeds} seeds for violations...\n")

    for scenario_id in unique_ids:
        for extra in ['0A', '0B', '1A', '1B']:
            key = (scenario_id, extra)
            if key not in spec_lookup:
                continue

            spec = spec_lookup[key]

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                tmpfile = f.name

            try:
                for seed in range(num_seeds):
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        generate_scenarios_from_tuples([spec], outfile=tmpfile, seed=seed)

                        for warning in w:
                            if 'Teammate belief violation' in str(warning.message):
                                violation_counts[key]['count'] += 1
                                if len(violation_counts[key]['examples']) < 3:
                                    violation_counts[key]['examples'].append(str(warning.message))
            finally:
                os.unlink(tmpfile)

    # Report
    print("=" * 80)
    print("TEAMMATE BELIEF VIOLATIONS BY SCENARIO")
    print("=" * 80)
    print()

    total_violations = 0
    violating_specs = []

    for (scenario_id, extra), data in sorted(violation_counts.items(), key=lambda x: (int(x[0][0]), x[0][1])):
        if data['count'] > 0:
            spec = spec_lookup[(scenario_id, extra)]
            total_violations += data['count']
            violating_specs.append((scenario_id, extra, data['count'], spec))
            print(f"ID={scenario_id}, Extra={extra}: {data['count']} violations ({data['count']/num_seeds*100:.0f}% of seeds)")
            print(f"  Spec: Self={spec['KS_Self'].value}, Teammate={spec['KS_Teammate'].value}, Opponent={spec['KS_Opponent'].value}, Answerer={spec['Answerer']}")
            if data['examples']:
                print(f"  Example: {data['examples'][0][:100]}...")
            print()

    print("=" * 80)
    print(f"SUMMARY: {total_violations} total violations across {len(violating_specs)} specs")
    print("=" * 80)

    # Group by spec pattern
    print("\nViolation patterns:")
    patterns = defaultdict(list)
    for scenario_id, extra, count, spec in violating_specs:
        pattern = (spec['KS_Self'].value, spec['KS_Teammate'].value, spec['KS_Opponent'].value, spec['Answerer'])
        patterns[pattern].append((scenario_id, extra, count))

    for pattern, instances in sorted(patterns.items(), key=lambda x: -sum(c for _, _, c in x[1])):
        total = sum(c for _, _, c in instances)
        ids = ', '.join(f"{sid}E{e}" for sid, e, _ in instances)
        print(f"  {pattern}: {total} violations (IDs: {ids})")

if __name__ == '__main__':
    analyze_violations()
