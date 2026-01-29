# Epistemic Metrics Definitions

This document defines the two metrics used to measure scenario complexity:
**Situation Tracking** and **Epistemic Category Transitions (ECTs)**.

## Situation Tracking Events

Events the **player (A) can see**. Put and move events are only
counted when A is in the room (perspective\_present = True). Leave and enter
events are **always** visible regardless of A's location.

Events happening while A is out of the room do **not** count toward situation
tracking, but they **can** contribute to ECT (since absent characters'
epistemic states change).

## Epistemic Category Transitions (ECTs)

Per-character, per-target-container (the queried container). A character's
epistemic state has two axes:

- **Certainty**: knowledge (directly observed current state) vs belief/uncertainty (stale or inferred)
- **Accuracy**: true (matches reality) vs false (doesn't match reality)

### Transition Types

| # | Trigger | Transition | Axis |
|---|---------|------------|------|
| 1 | Character **leaves** the room after seeing ≥1 event on target container | knowledge → belief/uncertainty | Certainty |
| 2 | Character (who previously had belief) **sees** a put/move on target container | belief/uncertainty → knowledge | Certainty |
| 3 | Target container contents **change while character is absent** (and character had a true belief) | true → false belief | Accuracy |
| 4 | Target container contents **change while character is present**, after having gone through #3 | false → true belief | Accuracy |

### What Is NOT an ECT

**Initial learning** — when a character first sees a put/move to the target
container, gaining knowledge from an unknown state. This is initialization,
not a tracking demand.

### Notes

- A single event can trigger transitions on **both axes** simultaneously.
  For example, a character who re-enters the room and sees a put that reverts
  the target container contents triggers both #2 (belief → knowledge) and #4
  (false → true), counting as 2 ECTs.
- **Total ECT** for a scenario = sum across all characters, all four
  transition types.
- ECTs are computed only for the **target (queried) container**, not all
  containers.

## Design Goal

Extra=0 and Extra=1 scenarios should have:
- **Similar** situation tracking demands (visible event counts)
- Extra=1 should **always** have more ECTs than its Extra=0 counterpart

This is achieved by:
1. Adding neutral **filler events** to Extra=0 scenarios (operations on the
   non-target container + leave/enter pairs) to equalize visible event counts
2. Extra=1's inserted events create epistemic divergence (characters leaving,
   missing changes, returning) that generate additional ECTs
