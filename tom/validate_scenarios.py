#!/usr/bin/env python3
"""Validate scenario generation produces correct epistemic states.

This script generates all scenarios from the CSV specs and verifies that
the resulting epistemic states match what was specified.
"""

import sys
import os
import argparse
from typing import List, Dict, Optional

from tom_helpers import (
    EpistemicState, Scenario, load_scenarios, read_specs_from_csv,
    CharacterType
)
from generate_tom_scenarios_new import generate_scenarios_from_tuples


def compute_actual_epistemic_state(
    scenario: Scenario,
    character: str,
    is_self: bool = False
) -> str:
    """Compute a character's actual epistemic state from scenario events.

    Args:
        scenario: The scenario to analyze
        character: The character name (A, B, C, D)
        is_self: Whether this is the Self character (uses KNOWS_X/BELIEVES_X)

    Returns:
        Epistemic state string: KNOWS_TRUTH, BELIEVES_TRUTH, BELIEVES_FALSE,
        UNKNOWN, KNOWS_X, or BELIEVES_X
    """
    target = scenario.question_container
    present = set(scenario.present_initially)

    # Track who has observed the target container
    observed_target: Dict[str, bool] = {c: False for c in ['A', 'B', 'C', 'D']}

    # Track beliefs when each character left
    beliefs_when_left: Dict[str, Optional[str]] = {}

    # Track actual container contents
    contents: Dict[str, Optional[str]] = {'bag': None, 'box': None, 'basket': None}

    for event in scenario.events:
        if event.event_type == 'put':
            contents[event.container] = event.item
            if event.container == target:
                for c in present:
                    observed_target[c] = True

        elif event.event_type == 'move':
            # Handle move from one container to another
            if event.from_container:
                contents[event.from_container] = None
            if event.to_container:
                contents[event.to_container] = event.item
            # Mark observation if move involves target
            if event.from_container == target or event.to_container == target:
                for c in present:
                    observed_target[c] = True

        elif event.event_type == 'remove':
            contents[event.container] = None
            if event.container == target:
                for c in present:
                    observed_target[c] = True

        elif event.event_type == 'leave':
            char = event.character
            if observed_target.get(char, False):
                beliefs_when_left[char] = contents[target]
            present.discard(char)

        elif event.event_type == 'enter':
            present.add(event.character)
            # CRITICAL: Entering does NOT give knowledge of container contents
            # Must witness a put/move event after entering

    final_contents = contents[target]

    # Determine epistemic state
    if character in present:
        # Character is present at end
        if observed_target.get(character, False):
            return 'KNOWS_X' if is_self else 'KNOWS_TRUTH'
        else:
            return 'UNKNOWN'
    else:
        # Character left the room
        if character not in beliefs_when_left:
            # Left before seeing anything about target
            return 'UNKNOWN'

        belief = beliefs_when_left[character]
        if is_self:
            # Self uses BELIEVES_X when they left (don't know if belief is true)
            return 'BELIEVES_X'

        # For others, compare belief to reality
        if belief == final_contents:
            return 'BELIEVES_TRUTH'
        else:
            return 'BELIEVES_FALSE'


def validate_scenario(scenario: Scenario, spec: dict, verbose: bool = False) -> List[str]:
    """Validate a single scenario matches its spec.

    Args:
        scenario: The generated scenario
        spec: The original specification
        verbose: Print detailed debug info

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Map character roles
    self_char = 'A'
    teammate_char = 'B'

    # Compute actual states
    actual_self = compute_actual_epistemic_state(scenario, self_char, is_self=True)
    actual_teammate = compute_actual_epistemic_state(scenario, teammate_char)
    actual_c = compute_actual_epistemic_state(scenario, 'C')
    actual_d = compute_actual_epistemic_state(scenario, 'D')

    # Convert expected states to comparable format
    expected_self = spec['KS_Self'].value.replace(' ', '_').upper()
    expected_teammate = spec['KS_Teammate'].value.replace(' ', '_').upper()
    expected_opponent = spec['KS_Opponent'].value.replace(' ', '_').upper()

    if verbose:
        print(f"  Target: {scenario.question_container}")
        print(f"  Events: {[(e.event_type, e.character) for e in scenario.events]}")
        print(f"  Present initially: {scenario.present_initially}")
        print(f"  Self: expected={expected_self}, actual={actual_self}")
        print(f"  Teammate: expected={expected_teammate}, actual={actual_teammate}")
        print(f"  C: {actual_c}, D: {actual_d}, expected opponent: {expected_opponent}")

    if actual_self != expected_self:
        errors.append(f"Self: expected {expected_self}, got {actual_self}")
    if actual_teammate != expected_teammate:
        errors.append(f"Teammate: expected {expected_teammate}, got {actual_teammate}")

    # For opponent validation:
    # - When Answerer=Opponent, check the specific answering opponent
    # - When Answerer=Self or Teammate, check if EITHER opponent matches
    if spec['Answerer'] == 'Opponent':
        opponent_char = scenario.who_answers
        actual_opponent = actual_c if opponent_char == 'C' else actual_d
        if actual_opponent != expected_opponent:
            errors.append(f"Opponent ({opponent_char}): expected {expected_opponent}, got {actual_opponent}")
    else:
        # Either opponent can have the expected state
        if actual_c != expected_opponent and actual_d != expected_opponent:
            errors.append(f"Opponent: expected {expected_opponent}, got C={actual_c}, D={actual_d}")

    return errors


def main():
    """Run validation on all scenarios."""
    parser = argparse.ArgumentParser(description='Validate scenario generation')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed debug info for failures')
    parser.add_argument('--id', type=str, help='Only validate specific scenario ID')
    parser.add_argument('--extra', type=str, help='Only validate specific Extra value (0A, 0B, 1A, 1B)')
    args = parser.parse_args()

    specs = read_specs_from_csv('ToM - scenarios.csv')

    # Filter if requested
    if args.id:
        specs = [s for s in specs if s['Id'] == args.id]
    if args.extra:
        specs = [s for s in specs if s['Extra'] == args.extra]

    if not specs:
        print("No matching specs found")
        return True

    tmp_file = 'validation_tmp.json'

    failed = 0
    passed = 0
    error_details = []

    chartypes = [
        CharacterType.LIVE_PLAYER,
        CharacterType.HONEST_OPPONENT,
        CharacterType.DISHONEST_TEAMMATE,
        CharacterType.DISHONEST_OPPONENT,
        CharacterType.NEUTRAL
    ]

    print(f"Validating {len(specs)} scenarios...")
    print()

    for i, spec in enumerate(specs):
        spec_id = spec['Id']
        extra = spec['Extra']

        try:
            # Generate scenario
            generate_scenarios_from_tuples([spec], outfile=tmp_file, seed=i, chartypes=chartypes)

            # Load generated scenarios
            scenarios, _, _ = load_scenarios(tmp_file)

            if not scenarios:
                error_msg = f"ERROR ID={spec_id} Extra={extra}: No scenarios generated"
                print(error_msg)
                error_details.append(error_msg)
                failed += 1
                continue

            scenario = scenarios[0]

            # Validate
            if args.verbose:
                print(f"\n--- ID={spec_id} Extra={extra} ---")
                print(f"  Spec: Self={spec['KS_Self'].value}, Teammate={spec['KS_Teammate'].value}, Opponent={spec['KS_Opponent'].value}")
                print(f"  Answerer={spec['Answerer']}")

            errors = validate_scenario(scenario, spec, verbose=args.verbose)
            if errors:
                error_msg = f"FAIL ID={spec_id} Extra={extra}: {'; '.join(errors)}"
                print(error_msg)
                error_details.append(error_msg)
                failed += 1
            else:
                passed += 1
                if args.verbose:
                    print(f"  PASS")

        except Exception as e:
            import traceback
            error_msg = f"ERROR ID={spec_id} Extra={extra}: {type(e).__name__}: {e}"
            print(error_msg)
            if args.verbose:
                traceback.print_exc()
            error_details.append(error_msg)
            failed += 1

    # Cleanup temp file
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    # Summary
    print()
    print("=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(specs)}")

    if failed > 0:
        print()
        print("Failed scenarios:")
        for detail in error_details:
            print(f"  {detail}")
        return False
    else:
        print("All scenarios validated successfully!")
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
