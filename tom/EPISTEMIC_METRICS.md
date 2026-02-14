# Epistemic Metrics Definitions

This document defines the two metrics used to measure scenario complexity:
**Situation Tracking** and **Epistemic Category Transitions (ECTs)**.

## Situation Tracking Events

Events the **player (A) can see**. Put and move events are only
counted when A is in the room (perspective\_present = True). Leave and enter
events are **always** visible regardless of A's location.

Put and move events while A is out do **not** count toward situation tracking
or ECT #2/#3/#4. However, enter and leave events are always visible, so ECT #1
(knowledge → belief when leaving) can still be triggered even when A is absent.

**Important caveat for ECT #1 when A is absent:** A can only count ECT #1 for
another character if A previously observed that character gaining knowledge
(i.e., A saw them witness a target event before A left). If A doesn't know
whether the character ever saw a target event, A cannot count ECT #1 for them.

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
| 4 | Target container contents **change while character is absent**, after having gone through #3 | false → true belief | Accuracy |

### ECT #2 and Room Entry

**When A is inside the room:** Containers are opaque. A character entering
does NOT automatically gain knowledge — they must witness a put/move event.
ECT #2 is only counted when they actually see an event on the target container.

**When A is outside the room:** A cannot observe what characters inside actually
see. When a non-A character enters, A assumes they *might* gain knowledge, so
ECT #2 is counted. (A entering never triggers ECT #2 for A — A knows their own
observations require witnessing an event.)

### What Is NOT an ECT

**Initial learning** — when a character first sees a put/move to the target
container, gaining knowledge from an unknown state. This is initialization,
not a tracking demand.

### Notes

- A single event cannot trigger transitions on **both axes** simultaneously.
  For example, a character who re-enters the room and sees a put that reverts
  the target container contents triggers #2 (belief → knowledge) but not #4
  (false → true), since the character *knows* the truth rather than believes it (true/false epistemic state parameters only apply to beliefs).
- **Total ECT** for a scenario = sum across all characters, all four
  transition types.
- ECTs are computed only for the **target (queried) container**, not all
  containers.

## Design Goal

The Extra field controls scenario complexity (see `EXTRA_MAPPING.md` for full details):

| Code | Name |
|------|------|
| 0A | Minimal Events |
| 0B | Event Load |
| 1A | Minimal ECT |
| 1B | ECT Load |

For paired scenarios (1A vs 1B, or legacy Extra=0 vs Extra=1):
- **Similar** situation tracking demands (visible event counts)
- 1B should **always** have more ECTs than its 1A counterpart

This is achieved by:
1. Adding neutral **filler events** to Extra=0 scenarios (operations on the
   non-target container + leave/enter pairs) to equalize visible event counts
2. Extra=1's inserted events create epistemic divergence (characters leaving,
   missing changes, returning) that generate additional ECTs

## Color Coding (for charts)

```python
EXTRA_CATEGORIES = {
    '0A': {'name': 'Minimal Events', 'short': 'Min Events', 'color': '#9b59b6'},  # Purple
    '0B': {'name': 'Event Load', 'short': 'Event Load', 'color': '#f39c12'},      # Orange
    '1A': {'name': 'Minimal ECT', 'short': 'Min ECT', 'color': '#3498db'},        # Blue
    '1B': {'name': 'ECT Load', 'short': 'ECT Load', 'color': '#e74c3c'},          # Red
}
```