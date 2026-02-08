"""
Tests for ECT counting logic in count_epistemic_category_transitions().

Issue #1: Does entering a room with non-empty target grant knowledge?
Per documentation: NO - entering does NOT grant knowledge.
Per code (lines 602-616): YES - if target non-empty, grants knowledge on enter.
"""
from tom_helpers import Event, Scenario
from generate_tom_scenarios_new import count_epistemic_category_transitions


def test_enter_nonempty_target():
    """C sees put, leaves, enters when target non-empty.

    Expected per docs: ECT #1 only (leave), NO ECT #2 on enter.
    Bug behavior: Code grants knowledge on enter, counts ECT #2.
    """
    scenario = Scenario(
        round_num=1,
        whose_turn='A',
        who_answers='A',
        question_container='bag',
        present_initially=['A', 'B', 'C'],
        events=[
            Event('put', 'B', container='bag', item='apple'),  # C sees this
            Event('leave', 'C'),  # ECT #1 for C (knowledge -> belief)
            Event('enter', 'C'),  # Should NOT trigger ECT #2
        ],
    )
    ects = count_epistemic_category_transitions(scenario)

    expected_certainty = 1  # Only the leave

    print(f"test_enter_nonempty_target:")
    print(f"  Actual ECTs: {ects}")
    print(f"  Expected certainty: {expected_certainty}")

    if ects['certainty'] == expected_certainty:
        print(f"  PASS")
        return True
    else:
        print(f"  FAIL - BUG CONFIRMED: Enter grants knowledge when target non-empty")
        return False


def test_enter_empty_target():
    """C sees put, leaves, target emptied, enters.

    When target is empty, enter correctly does not grant knowledge.
    """
    scenario = Scenario(
        round_num=1,
        whose_turn='A',
        who_answers='A',
        question_container='bag',
        present_initially=['A', 'B', 'C'],
        events=[
            Event('put', 'B', container='bag', item='apple'),  # C sees
            Event('leave', 'C'),  # ECT #1
            Event('move', 'B', from_container='bag', to_container='box', item='apple'),  # Empty bag
            Event('enter', 'C'),  # Target empty, no knowledge granted
        ],
    )
    ects = count_epistemic_category_transitions(scenario)

    expected_certainty = 1  # Only the leave

    print(f"\ntest_enter_empty_target:")
    print(f"  Actual ECTs: {ects}")
    print(f"  Expected certainty: {expected_certainty}")

    if ects['certainty'] == expected_certainty:
        print(f"  PASS")
        return True
    else:
        print(f"  FAIL")
        return False


def test_revelation_pattern():
    """Leave, target emptied, enter, witness move.

    The revelation pattern: leave, target emptied, enter, witness move.
    """
    scenario = Scenario(
        round_num=1,
        whose_turn='A',
        who_answers='A',
        question_container='bag',
        present_initially=['A', 'B', 'C'],
        events=[
            Event('put', 'B', container='bag', item='apple'),  # C sees
            Event('leave', 'C'),  # ECT #1
            Event('move', 'B', from_container='bag', to_container='box', item='apple'),  # Empty bag
            Event('enter', 'C'),  # Target empty, no knowledge
            Event('move', 'B', from_container='box', to_container='bag', item='apple'),  # ECT #2
        ],
    )
    ects = count_epistemic_category_transitions(scenario)

    expected_certainty = 2  # #1 on leave, #2 on witnessing move

    print(f"\ntest_revelation_pattern:")
    print(f"  Actual ECTs: {ects}")
    print(f"  Expected certainty: {expected_certainty}")

    if ects['certainty'] == expected_certainty:
        print(f"  PASS")
        return True
    else:
        print(f"  FAIL")
        return False


def test_multiple_enters_nonempty_target():
    """Multiple characters enter when target is non-empty.

    Both C and D leave and re-enter. A is inside, so no ECT #2 on enters.
    """
    scenario = Scenario(
        round_num=1,
        whose_turn='A',
        who_answers='A',
        question_container='bag',
        present_initially=['A', 'B', 'C', 'D'],
        events=[
            Event('put', 'B', container='bag', item='apple'),  # All see
            Event('leave', 'C'),  # ECT #1 for C
            Event('leave', 'D'),  # ECT #1 for D
            Event('enter', 'C'),  # A inside, no ECT #2
            Event('enter', 'D'),  # A inside, no ECT #2
        ],
    )
    ects = count_epistemic_category_transitions(scenario)

    expected_certainty = 2  # Two leaves, no ECT #2 on enters (A is inside)

    print(f"\ntest_multiple_enters_nonempty_target:")
    print(f"  Actual ECTs: {ects}")
    print(f"  Expected certainty: {expected_certainty}")

    if ects['certainty'] == expected_certainty:
        print(f"  PASS")
        return True
    else:
        print(f"  FAIL - Enter granted knowledge when A was inside")
        return False


def test_enter_when_a_outside():
    """When A is outside, other characters entering SHOULD count ECT #2.

    A leaves first, then C enters. A can't observe what C sees inside,
    so A assumes C might gain knowledge.
    """
    scenario = Scenario(
        round_num=1,
        whose_turn='A',
        who_answers='A',
        question_container='bag',
        present_initially=['A', 'B', 'C'],
        events=[
            Event('put', 'B', container='bag', item='apple'),  # All see
            Event('leave', 'C'),  # ECT #1 for C
            Event('leave', 'A'),  # ECT #1 for A, now A is outside
            Event('enter', 'C'),  # A outside, so ECT #2 for C (A assumes C might know)
        ],
    )
    ects = count_epistemic_category_transitions(scenario)

    # ECT #1 for C leaving, ECT #1 for A leaving, ECT #2 for C entering (A outside)
    expected_certainty = 3

    print(f"\ntest_enter_when_a_outside:")
    print(f"  Actual ECTs: {ects}")
    print(f"  Expected certainty: {expected_certainty}")

    if ects['certainty'] == expected_certainty:
        print(f"  PASS")
        return True
    else:
        print(f"  FAIL - Expected {expected_certainty}, got {ects['certainty']}")
        return False


def test_a_enters_no_ect2():
    """When A enters, A should NOT get ECT #2 (A knows their own observations).

    A leaves, then A enters. A knows containers are opaque.
    """
    scenario = Scenario(
        round_num=1,
        whose_turn='A',
        who_answers='A',
        question_container='bag',
        present_initially=['A', 'B', 'C'],
        events=[
            Event('put', 'B', container='bag', item='apple'),  # A sees
            Event('leave', 'A'),  # ECT #1 for A
            Event('enter', 'A'),  # A enters - no ECT #2 (A knows they must witness event)
        ],
    )
    ects = count_epistemic_category_transitions(scenario)

    expected_certainty = 1  # Only ECT #1 when A left

    print(f"\ntest_a_enters_no_ect2:")
    print(f"  Actual ECTs: {ects}")
    print(f"  Expected certainty: {expected_certainty}")

    if ects['certainty'] == expected_certainty:
        print(f"  PASS")
        return True
    else:
        print(f"  FAIL - A should not get ECT #2 on enter")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("ECT COUNTING TESTS - Enter and ECT #2 behavior")
    print("=" * 60)
    print()

    results = []
    results.append(("test_enter_nonempty_target", test_enter_nonempty_target()))
    results.append(("test_enter_empty_target", test_enter_empty_target()))
    results.append(("test_revelation_pattern", test_revelation_pattern()))
    results.append(("test_multiple_enters_nonempty_target", test_multiple_enters_nonempty_target()))
    results.append(("test_enter_when_a_outside", test_enter_when_a_outside()))
    results.append(("test_a_enters_no_ect2", test_a_enters_no_ect2()))

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    failed = sum(1 for _, r in results if not r)
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\nFailed tests:")
        for name, r in results:
            if not r:
                print(f"  - {name}")
