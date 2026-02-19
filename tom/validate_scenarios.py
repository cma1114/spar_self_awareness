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
# Note: generate_scenarios_from_tuples is imported lazily in main() to avoid circular import


def compute_actual_epistemic_state(
    scenario: Scenario,
    character: str,
    is_self: bool = False
) -> str:
    """Compute a character's ACTUAL epistemic state from scenario events.

    This computes the character's actual state, regardless of what A knows.

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
            if event.from_container:
                contents[event.from_container] = None
            if event.to_container:
                contents[event.to_container] = event.item
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
            return 'UNKNOWN'

        belief = beliefs_when_left[character]
        if is_self:
            return 'BELIEVES_X'

        if belief == final_contents:
            return 'BELIEVES_TRUTH'
        else:
            return 'BELIEVES_FALSE'


def a_can_determine_state(scenario: Scenario, character: str) -> bool:
    """Check if A can determine another character's epistemic state.

    For A to determine a character's state:
    1. A must know what the character believes (A saw them observe target OR A saw them leave with a belief)
    2. A must know the current truth (to compare belief to reality)

    Args:
        scenario: The scenario to analyze
        character: The character name (B, C, D) - not A

    Returns:
        True if A can determine the character's state, False if UNKNOWN from A's perspective
    """
    target = scenario.question_container
    present = set(scenario.present_initially)
    a_present = 'A' in present

    # Track observations
    observed_target: Dict[str, bool] = {c: False for c in ['A', 'B', 'C', 'D']}
    a_saw_char_observe: Dict[str, bool] = {c: False for c in ['A', 'B', 'C', 'D']}
    a_knows_char_belief: Dict[str, bool] = {c: False for c in ['A', 'B', 'C', 'D']}

    contents: Dict[str, Optional[str]] = {'bag': None, 'box': None, 'basket': None}
    a_last_known_contents: Optional[str] = None

    for event in scenario.events:
        if event.event_type == 'put':
            contents[event.container] = event.item
            if event.container == target:
                for c in present:
                    observed_target[c] = True
                    if a_present:
                        a_saw_char_observe[c] = True
                if a_present:
                    a_last_known_contents = event.item

        elif event.event_type == 'move':
            if event.from_container:
                contents[event.from_container] = None
            if event.to_container:
                contents[event.to_container] = event.item
            if event.from_container == target or event.to_container == target:
                for c in present:
                    observed_target[c] = True
                    if a_present:
                        a_saw_char_observe[c] = True
                if a_present:
                    a_last_known_contents = contents[target]

        elif event.event_type == 'remove':
            contents[event.container] = None
            if event.container == target:
                for c in present:
                    observed_target[c] = True
                    if a_present:
                        a_saw_char_observe[c] = True
                if a_present:
                    a_last_known_contents = None

        elif event.event_type == 'leave':
            char = event.character
            if observed_target.get(char, False) and a_present:
                a_knows_char_belief[char] = True
            if char == 'A':
                a_present = False
            present.discard(char)

        elif event.event_type == 'enter':
            if event.character == 'A':
                a_present = True
            present.add(event.character)

    final_contents = contents[target]

    # A knows the truth only if A is present at the end AND observed the target
    # If A left the room, A only has a BELIEF, not knowledge (even if it matches reality)
    a_knows_truth = a_present and observed_target.get('A', False)

    if character in present:
        # Character is present at end - they KNOW_TRUTH if they observed.
        # A can determine this just by having witnessed them observe - no need to know contents.
        return a_saw_char_observe.get(character, False)
    else:
        # Character left - A can determine if A knows their belief AND A knows truth
        return a_knows_char_belief.get(character, False) and a_knows_truth


def validate_epistemic_state(
    scenario: Scenario,
    character: str,
    expected: str,
    is_self: bool = False,
    verbose: bool = False
) -> Optional[str]:
    """Validate a character's epistemic state against expected value.

    Uses hybrid approach:
    - For KNOWS_TRUTH, BELIEVES_TRUTH, BELIEVES_FALSE: Check actual state
    - For UNKNOWN: Check that A cannot determine the character's state

    Args:
        scenario: The scenario to validate
        character: The character to check (A, B, C, D)
        expected: Expected state string (KNOWS_TRUTH, BELIEVES_TRUTH, etc.)
        is_self: Whether this is the Self character (A)
        verbose: Print debug info

    Returns:
        Error message if validation fails, None if valid
    """
    actual = compute_actual_epistemic_state(scenario, character, is_self=is_self)

    if expected == 'UNKNOWN':
        # For UNKNOWN: A must NOT be able to determine the character's state
        if character == 'A':
            # Self is UNKNOWN if A left the room after seeing target
            if actual != 'UNKNOWN':
                return f"{character}: expected UNKNOWN (A left), but actual state is {actual}"
        else:
            # For others: A cannot determine their state
            can_determine = a_can_determine_state(scenario, character)
            if can_determine:
                return f"{character}: expected UNKNOWN from A's perspective, but A can determine state ({actual})"
    else:
        # For definite states: compare actual to expected
        if actual != expected:
            return f"{character}: expected {expected}, got {actual}"

    return None


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

    # Convert expected states to comparable format
    expected_self = spec['KS_Self'].value.replace(' ', '_').upper()
    expected_teammate = spec['KS_Teammate'].value.replace(' ', '_').upper()
    expected_opponent = spec['KS_Opponent'].value.replace(' ', '_').upper()

    if verbose:
        actual_self = compute_actual_epistemic_state(scenario, self_char, is_self=True)
        actual_teammate = compute_actual_epistemic_state(scenario, teammate_char)
        actual_c = compute_actual_epistemic_state(scenario, 'C')
        actual_d = compute_actual_epistemic_state(scenario, 'D')
        print(f"  Target: {scenario.question_container}")
        print(f"  Events: {[(e.event_type, e.character) for e in scenario.events]}")
        print(f"  Present initially: {scenario.present_initially}")
        print(f"  Self: expected={expected_self}, actual={actual_self}")
        print(f"  Teammate: expected={expected_teammate}, actual={actual_teammate}")
        print(f"  C: {actual_c}, D: {actual_d}, expected opponent: {expected_opponent}")

    # Validate Self (A)
    err = validate_epistemic_state(scenario, self_char, expected_self, is_self=True, verbose=verbose)
    if err:
        errors.append(f"Self: {err}")

    # Validate Teammate (B)
    err = validate_epistemic_state(scenario, teammate_char, expected_teammate, verbose=verbose)
    if err:
        errors.append(f"Teammate: {err}")

    # Validate Opponent
    # - When Answerer=Opponent, check the specific answering opponent
    # - When Answerer=Self or Teammate, check if EITHER opponent matches
    if spec['Answerer'] == 'Opponent':
        opponent_char = scenario.who_answers
        err = validate_epistemic_state(scenario, opponent_char, expected_opponent, verbose=verbose)
        if err:
            errors.append(f"Opponent ({opponent_char}): {err}")
    else:
        # Either opponent can have the expected state
        err_c = validate_epistemic_state(scenario, 'C', expected_opponent, verbose=verbose)
        err_d = validate_epistemic_state(scenario, 'D', expected_opponent, verbose=verbose)
        if err_c and err_d:
            errors.append(f"Opponent: neither C nor D match expected {expected_opponent} (C: {err_c}, D: {err_d})")

    return errors


def main():
    """Run validation on all scenarios."""
    # Lazy import to avoid circular dependency with generate_tom_scenarios_new
    from generate_tom_scenarios_new import generate_scenarios_from_tuples

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
