# Expanded UNKNOWN Analysis for Self=BELIEVES_X Scenarios

## Epistemic States (from tom_helpers.py)

```python
class EpistemicState(Enum):
    BELIEVES_TRUTH = "Believes Truth"
    BELIEVES_FALSE = "Believes False"
    KNOWS_TRUTH = "Knows Truth"
    KNOWS_X = "Knows X"
    BELIEVES_X = "Believes X"
    UNKNOWN = "Unknown"
```

## How States Are Used

| Role | Possible States | Meaning |
|------|-----------------|---------|
| Self (A) | KNOWS_X | A is present at end, knows current target state |
| Self (A) | BELIEVES_X | A saw target, then left - has belief but uncertain if still current |
| Teammate/Opponent | KNOWS_TRUTH | Present at end, knows the final (true) state |
| Teammate/Opponent | BELIEVES_TRUTH | Left with belief that matches final reality, AND A knows this |
| Teammate/Opponent | BELIEVES_FALSE | Left with belief that doesn't match final reality, AND A knows this |
| Teammate/Opponent | UNKNOWN | A doesn't know whether B's belief is true or false |

**Key distinction:**
- Self uses X-states (about what A knows/believes)
- Others use TRUTH/FALSE/UNKNOWN-states (about whether their belief matches reality, from A's perspective)

---

## What UNKNOWN Actually Means

UNKNOWN does NOT mean "B has no information" or "B never saw anything."

**UNKNOWN means: A doesn't know whether what B believes is true or false.**

This implies:
- B may have a belief (B may have seen target events)
- But A cannot verify if B's belief matches reality
- From A's perspective, B's belief accuracy is unknown

From EpistemicType enum:
```python
TEAMMATE_HAS_UNKNOWN_BELIEF = "teammate_has_unknown_belief"
# Player has left so doesn't know whether whatever teammate believes is true or false
```

---

## States Are From A's Perspective

**When A is present:**
- A can observe what others see and do
- A can classify others as BELIEVES_TRUE, BELIEVES_FALSE, KNOWS_TRUTH, or UNKNOWN

**When A is absent (BELIEVES_X):**
- A can't observe what happens after A leaves
- If B is still in the room at the end → B KNOWS_TRUTH (B sees the final state)
- If B is also out of the room → A can't determine if B's belief matches reality → B is UNKNOWN

---

## Scenario Event Structure for Self=BELIEVES_X

Events are sequential (no simultaneous actions):

```
1. [Optional: leave_immediately_group leaves]
2. First put: INTERMEDIATE item in target
3. Characters in exclude_false leave
4. Second put: FINAL item in target (queried_item)
5. [Characters in exclude_true leave]
6. End: Characters in include remain
```

**A's trajectory for BELIEVES_X:**
- A sees at least one put → A has knowledge
- A leaves before the end (at step 3 OR step 5)
- A is not present at the end → A has a BELIEF (not knowledge)

**The X in BELIEVES_X means A doesn't know if the belief is true or false:**
- If A leaves at step 3: A saw intermediate, missed final → belief is FALSE
- If A leaves at step 5: A saw final → belief is TRUE
- Either way, A doesn't know if anyone changed the target after A left

---

## Possible B Trajectories for Teammate=UNKNOWN

| Case | When B leaves | What B sees | B's actual state | A's ability to classify |
|------|---------------|-------------|------------------|------------------------|
| 1 | Before any puts | Nothing | No belief | A knows B has no info |
| 2 | After a put, before A | Some put(s) | Has a belief | A saw what B saw, but A doesn't know final reality |
| 3 | After A leaves | Unknown to A | Unknown to A | A knows B left after A, but doesn't know what B saw |
| 4 | Never (present at end) | Everything | KNOWS final | Invalid - B KNOWS, not UNKNOWN |

**Hard constraint:** B must NOT be in the room at the end (Case 4), otherwise B KNOWS.

**Valid cases for UNKNOWN:**
- Case 1: B left before seeing anything (current implementation)
- Case 2: B saw something, left before A - A knows what B believes but A doesn't know the final reality
- Case 3: B left after A - A knows B left after A, but A doesn't know what B saw in the meantime

---

## Implementation Status

**IMPLEMENTED:** All 3 cases with 33/33/33 random distribution:
- Case 1: B leaves before any puts (B has no belief)
- Case 2: B leaves after intermediate put, before A leaves (B has belief, A knows it but can't verify)
- Case 3: B leaves after A leaves, before final put (A doesn't know what B saw)

All three cases result in UNKNOWN from A's perspective - A cannot determine whether B's belief (if any) is true or false.

**Key changes:**
- `exclude_flexible_unknown` set for characters with flexible timing
- `exclude_after_self` set for Case 3 (leave after A, before final put)
- ECT #1 counts for all leave events (even after A leaves) since leave events are visible

---

## Why A Can't Classify B in Cases 2-3

**To classify B as BELIEVES_TRUTH or BELIEVES_FALSE, A needs BOTH:**
1. Knowledge of what B believes (A saw what B saw before B left)
2. Knowledge of the final reality (A knows the final target state)

**Case 2 (B leaves before A, after seeing intermediate):**
- A sees B witness the intermediate put
- A sees B leave
- A knows what B believes (the intermediate state) ✓
- A leaves before the final put
- A doesn't know the final reality ✗
- A can't compare B's belief to reality → UNKNOWN

**Case 3 (B leaves after A):**
- A leaves first
- A knows B was present when A left, and B is not present at the end → A knows B left sometime after A
- A doesn't know what B saw in the meantime ✗
- A doesn't know the final reality ✗
- A can't classify B → UNKNOWN

---

## ECT Analysis

**ECTs are counted from A's perspective** - only transitions that A can observe or track count toward the scenario's epistemic complexity.

For each character, ECT transitions are:
| # | Trigger | Transition |
|---|---------|------------|
| 1 | Character leaves after seeing ≥1 event on target | knowledge → belief |
| 2 | Character (with belief) sees put/move on target | belief → knowledge |
| 3 | Target changes while character absent (with TRUE belief) | true → false belief |
| 4 | Target changes while character present (after #3) | false → true belief |

**For Self=BELIEVES_X:**
- A sees at least one put → initial learning (not ECT)
- A leaves → ECT #1 for A (knowledge → belief) - A tracks this
- Put/move events after A leaves → A can't see them, so no ECT #2/#3/#4

**For B (from A's perspective):**
- Case 1 (B leaves before puts, while A present): A sees B leave with no info → 0 ECTs for B
- Case 2 (B sees something, leaves before A): A sees B witness put, A sees B leave → ECT #1 for B
- Case 3 (B leaves after A): A sees B leave (enter/leave always visible) → ECT #1 for B if B had seen target events before leaving

**ECT Visibility Rules:**

ECTs are counted based on what A can observe:
- **When A is in the room:** All events are visible → all ECT types (#1, #2, #3, #4) can be counted
- **When A is out of the room:** Only enter/leave events are visible → only ECT #1 can be counted

This means after A leaves:
- **#1 for A** (at moment A leaves): YES - this happens as A leaves
- **#1 for other characters:** YES - A sees all enter/leave events regardless of A's location
- **#2 for any character:** NO - requires witnessing a put/move, which A can't see while out
- **#3 for any character:** NO - requires seeing target contents change, which A can't see while out
- **#4 for any character:** NO - requires seeing target contents change, which A can't see while out

**Key insight:** A's states are KNOWS_X and BELIEVES_X - A doesn't track whether their own belief is true or false. The TRUE/FALSE distinction only applies to other characters, and only when A can observe it.

---

## Affected Scenarios

| ID | Answerer | Self | Teammate | Opponent |
|----|----------|------|----------|----------|
| 10 | Self | Believes X | Unknown | Unknown |
| 11 | Self | Believes X | Unknown | Knows Truth |
| 23 | Teammate | Believes X | Unknown | Unknown |
| 24 | Teammate | Believes X | Unknown | Knows Truth |
| 37 | Opponent | Believes X | Unknown | Knows Truth |

---

## Implementation Plan

### 1. Add `exclude_flexible_unknown` category

Characters in this set must leave before the end, but timing is flexible:
- Can leave before puts (current behavior, Case 1)
- Can leave after intermediate, before A leaves (Case 2)
- Can leave after A leaves, before final (Case 3)

### 2. Modify `plan_availability()` for affected scenarios

For scenarios where Self=BELIEVES_X and Teammate=UNKNOWN:
Replace `exclude_unknown.add(teammate)` with `exclude_flexible_unknown.add(teammate)`

### 3. Handle flexible timing in `build_scenario()`

```python
for who in sorted(self.exclude_flexible_unknown):
    r = self.rng.random()
    if r < 0.33:
        leave_immediately_group.add(who)  # Case 1: before puts
    elif r < 0.67:
        # Case 2: after intermediate, before A (put in exclude_false but before self_char)
        self.exclude_false.add(who)
    else:
        # Case 3: after A leaves, before final put
        # Need new mechanism: leave after A but before second put
        self.exclude_after_self.add(who)  # New category needed
```

Case 3 uses `exclude_after_self` set - characters who leave after A but before the final put.

### 4. Extra=1B: Add B re-entry

For Extra=1B scenarios, B can re-enter after A leaves then leave again:
- Adds 2 SIT events (enter + leave)
- B gains knowledge upon entering (sees current target state)
- B leaving triggers ECT #1 (knowledge → belief) - this counts since leave events are visible to A
- A's classification of B remains UNKNOWN (A doesn't know what B saw)

---

## Verification Checklist

**All verified:**
- [x] B's leave timing varies across seeds (~33% each case)
- [x] B is NEVER present at the end (would make B KNOWS)
- [x] A still has BELIEVES_X (saw intermediate, left before final)
- [x] UNKNOWN semantics preserved: A can't classify B's belief as true or false
- [x] ECT #1 counts for B's re-entry leave (leave events visible to A)

---

## Summary

**UNKNOWN = A doesn't know whether B's belief is true or false**

To classify B as BELIEVES_TRUTH or BELIEVES_FALSE, A needs:
1. To know what B believes (A saw what B witnessed)
2. To know the final reality (A knows the final target state)

If A is missing either piece, B is UNKNOWN.

**For Self=BELIEVES_X scenarios with Teammate=UNKNOWN:**
- A left before the end → A doesn't know the final reality
- B also left before the end → B is not present to have KNOWS_TRUTH
- Since A doesn't know the final reality, A can't classify B's belief as TRUE or FALSE
- B is UNKNOWN regardless of when B leaves (as long as B also isn't present at the end)

**Valid B trajectories (all result in UNKNOWN):**
1. B leaves before seeing anything
2. B sees something, leaves before A
3. B leaves after A

All three preserve the UNKNOWN semantics because A lacks knowledge of the final reality.
