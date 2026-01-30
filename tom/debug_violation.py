#!/usr/bin/env python3
"""Debug a specific violation scenario."""

import sys
import os
import warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tom_helpers import read_specs_from_csv
from generate_tom_scenarios_new import generate_scenarios_from_tuples
import tempfile
import json

def debug_scenario(scenario_id='29', extra='1B', seed=0, verbose=True):
    """Generate and inspect a specific scenario. Returns True if violation found."""
    specs = read_specs_from_csv('tom/ToM - scenarios.csv')

    spec = None
    for s in specs:
        if s['Id'] == scenario_id and s['Extra'] == extra:
            spec = s
            break

    if spec is None:
        return False

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        tmpfile = f.name

    has_violation = False
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            generate_scenarios_from_tuples([spec], outfile=tmpfile, seed=seed)

            if w:
                has_violation = True
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"VIOLATION FOUND at seed {seed}")
                    print(f"{'='*60}")
                    print(f"Spec: ID={spec['Id']}, Extra={spec['Extra']}")
                    print(f"  Self={spec['KS_Self'].value}")
                    print(f"  Teammate={spec['KS_Teammate'].value}")
                    print(f"  Opponent={spec['KS_Opponent'].value}")
                    print(f"  Answerer={spec['Answerer']}")
                    print()
                    for warning in w:
                        print(f"WARNING: {warning.message}")
                    print()

                    with open(tmpfile, 'r') as f:
                        data = json.load(f)

                    if data.get('scenarios'):
                        scenario = data['scenarios'][0]
                        print(f"Present initially: {scenario['present_initially']}")
                        print(f"Question container: {scenario['question_container']}")
                        print()
                        print("Events:")
                        for idx, event in enumerate(scenario['events']):
                            event_str = f"  {idx}: {event['event_type']}"
                            if event.get('character'):
                                event_str += f" by {event['character']}"
                            if event.get('container'):
                                event_str += f" → {event['item']} in {event['container']}"
                            elif event.get('from_container'):
                                event_str += f" → {event['item']} from {event['from_container']} to {event['to_container']}"
                            print(event_str)
    finally:
        os.unlink(tmpfile)

    return has_violation

if __name__ == '__main__':
    # Find first seed with violation for ID 29
    print("Searching for violation in ID 29, Extra=1B...")
    for seed in range(100):
        if debug_scenario(scenario_id='29', extra='1B', seed=seed, verbose=True):
            break
    else:
        print("No violations found in first 100 seeds")
