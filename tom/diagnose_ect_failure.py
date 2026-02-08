"""
Diagnose why 1B scenarios lose accuracy ECTs compared to 1A.
Traces through event sequences to understand what's happening.
"""
import random
import sys
sys.path.insert(0, '/Users/christopherackerman/repos/spar_self_awareness/tom')

from tom_helpers import EpistemicState, read_specs_from_csv
from generate_tom_scenarios_new import (
    Scenario_Builder, count_epistemic_category_transitions,
    insert_filler_events, insert_extra_events_with_revelation
)
import copy

# IDs that failed validation
FAILING_IDS = [2, 4, 5, 6, 8]

def trace_ects(scenario):
    """Count ECTs and return detailed breakdown."""
    result = count_epistemic_category_transitions(scenario)
    return result

def print_events(scenario, label=""):
    """Print all events in a scenario."""
    print(f"\n{label} Events:")
    for i, e in enumerate(scenario.events):
        print(f"  {i}: {e}")

def diagnose_scenario(spec, seed=0):
    """Diagnose a single scenario."""
    spec_id = spec.get('Id')
    print(f"\n{'='*70}")
    print(f"ID {spec_id}: Self={spec.get('KS_Self')}, Answerer={spec.get('Answerer')}")
    print(f"{'='*70}")

    # Generate base scenario
    rng = random.Random(seed)
    builder = Scenario_Builder(spec, rng)
    base_scenario = builder.build_scenario()

    print(f"\nBase scenario question_container: {base_scenario.question_container}")
    print(f"Base scenario answerer: {base_scenario.who_answers}")
    print_events(base_scenario, "Base")

    base_ects = trace_ects(base_scenario)
    print(f"\nBase ECTs: {base_ects}")

    # Create 1A (with filler)
    scenario_1a = copy.deepcopy(base_scenario)
    scenario_1a.extra = "1A"
    rng_1a = random.Random(seed + 1)
    insert_filler_events(scenario_1a, rng_1a)

    print_events(scenario_1a, "1A (with filler)")
    ects_1a = trace_ects(scenario_1a)
    print(f"\n1A ECTs: {ects_1a}")

    # Create 1B (with revelation)
    scenario_1b = copy.deepcopy(base_scenario)
    scenario_1b.extra = "1B"
    rng_1b = random.Random(seed + 2)
    insert_extra_events_with_revelation(scenario_1b, scenario_1b.who_answers, rng_1b, spec)

    print_events(scenario_1b, "1B (with revelation)")
    ects_1b = trace_ects(scenario_1b)
    print(f"\n1B ECTs: {ects_1b}")

    # Compare
    print(f"\n--- COMPARISON ---")
    print(f"1A: certainty={ects_1a['certainty']}, accuracy={ects_1a['accuracy']}, total={ects_1a['total']}")
    print(f"1B: certainty={ects_1b['certainty']}, accuracy={ects_1b['accuracy']}, total={ects_1b['total']}")

    if ects_1b['total'] <= ects_1a['total']:
        print(f"PROBLEM: 1B ECT ({ects_1b['total']}) <= 1A ECT ({ects_1a['total']})")
    else:
        print(f"OK: 1B ECT ({ects_1b['total']}) > 1A ECT ({ects_1a['total']})")

    return ects_1a, ects_1b

def main():
    specs = read_specs_from_csv('ToM - scenarios.csv')
    print(f"Loaded {len(specs)} specs")

    for spec in specs:
        spec_id = spec.get('Id')  # Note: 'Id' not 'ID'
        extra = spec.get('Extra', '')
        # Skip non-1A/1B or if ID doesn't match
        if extra not in ('1A', '1B'):
            continue
        # Handle both string and int IDs
        if isinstance(spec_id, str) and spec_id.isdigit():
            spec_id = int(spec_id)
        if spec_id in FAILING_IDS and extra == '1A':  # Only diagnose 1A specs
            diagnose_scenario(spec, seed=0)

if __name__ == '__main__':
    main()
