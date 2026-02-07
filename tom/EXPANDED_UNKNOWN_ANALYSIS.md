# Expanded UNKNOWN Analysis for Self=BELIEVES_X Scenarios

## Key Principle: Everything is from Player A's Perspective

All epistemic states and ECT counts are from **A's perspective** - what A can observe, track, and reason about. Events happening while A is absent are invisible to A and don't affect ECT counts.

---

## Epistemic States (from A's Perspective)

| State | Meaning |
|-------|---------|
| KNOWS_X / KNOWS_TRUTH | A is present at the end, witnessed the final state |
| BELIEVES_X | A saw something on the target, then left. A has a belief but doesn't know if it's still current |
| UNKNOWN | A has no information about the target (never saw any target events) |

**These are A's states about A's own knowledge.**

For other characters (from A's perspective):
- A can observe what others did while A was present
- A cannot know what happened after A left
- If A knows B left before A, A knows B left (but not what B knows)
- If B left after A, A doesn't know B's current status

---

## Scenario Structure for Self=BELIEVES_X

```
1. [Optional: Some characters leave before any puts]
2. First put on target (A sees this, A has KNOWLEDGE)
3. A leaves → A now has BELIEVES_X
4. [Everything after this is invisible to A]
```

**Once A leaves, A's ECT count is fixed.** Whatever happens in the room after A leaves doesn't affect what A can track.

### A's ECT Count

| Event | A's Transition | ECT? |
|-------|----------------|------|
| First put on target | A gains KNOWLEDGE | No (initial learning, not a transition) |
| A leaves | A has BELIEVES_X | No (leaving doesn't change certainty - A still believes what A saw) |

**A's ECTs: 0** (for the base scenario - A gains knowledge once, then leaves)

Wait - let me reconsider. ECT #1 is "Character leaves after seeing ≥1 event on target → knowledge → belief". So:

| Event | A's Transition | ECT? |
|-------|----------------|------|
| First put on target | UNKNOWN → KNOWLEDGE | No (initial learning) |
| A leaves | KNOWLEDGE → BELIEF | ECT #1 |

**A's ECTs: 1** (from leaving after witnessing)

---

## Possible B Trajectories (Teammate=UNKNOWN)

From the scenario spec, B should be UNKNOWN. What does this mean from A's perspective?

| Case | When B Leaves | What A Observes | B's State (from A's view) |
|------|---------------|-----------------|---------------------------|
| 1 | Before any puts | A sees B leave before target events | A knows B is UNKNOWN |
| 2 | After first put, before A | A sees B leave after seeing target | A knows B has some info (not UNKNOWN) |
| 3 | After A leaves | A doesn't see B leave | A doesn't know B's status |

**Key insight for Cases 2 and 3:**
- In Case 2, if A sees B witness the first put before B leaves, then from A's view, B is NOT unknown - B has information
- In Case 3, A doesn't know what happened to B

**For the spec to say Teammate=UNKNOWN, Case 1 is the cleanest interpretation:**
- B leaves before any target events
- A sees B leave
- A knows B has no information about the target
- Therefore A knows B can't help

---

## What "UNKNOWN" Really Means

UNKNOWN is not a belief state. It's the absence of any belief or knowledge about the target.

- NOT "true belief" or "false belief"
- NOT "B believes X but X is wrong"
- Simply: B has no information

From A's perspective:
- If A knows B left before seeing anything → A knows B is UNKNOWN
- If A doesn't know what B saw → A is uncertain about B's state (which is different)

---

## ECT Counting (from A's Perspective)

ECTs track epistemic transitions that A can observe or infer:

| # | Trigger | Transition | Requires A present? |
|---|---------|------------|---------------------|
| 1 | Character leaves after seeing target event | knowledge → belief | A must see the character leave |
| 2 | Character (with belief) sees put/move on target | belief → knowledge | A must see both the character and the event |
| 3 | Target changes while character absent | true → false belief | A must see the target change |
| 4 | Target changes while character present (after #3) | false → true belief | A must see the target change |

**If A is absent, A can't count any ECTs for events A doesn't see.**

---

## Scenarios 10, 11, 23, 24, 37

| ID | Answerer | Self | Teammate | Opponent |
|----|----------|------|----------|----------|
| 10 | Self | Believes X | Unknown | Unknown |
| 11 | Self | Believes X | Unknown | Knows Truth |
| 23 | Teammate | Believes X | Unknown | Unknown |
| 24 | Teammate | Believes X | Unknown | Knows Truth |
| 37 | Opponent | Believes X | Unknown | Knows Truth |

**Current implementation:** B leaves before any puts (B is truly UNKNOWN from A's view)

**Proposed expansion:** B can leave at various times, but from A's perspective:
- If B leaves before first put → A knows B is UNKNOWN
- If B leaves after first put (while A present) → A knows B has info (not UNKNOWN per spec)
- If B leaves after A → A doesn't know B's state

**Implication:** For the spec to say "Teammate=UNKNOWN", A must be able to determine that B has no information. This requires A to see B leave before any target events.

---

## The Real Question

What does "Teammate=UNKNOWN" mean in the context of these scenarios?

**Interpretation 1 (strict):** B is truly UNKNOWN - left before seeing anything. A knows this.

**Interpretation 2 (from A's uncertainty):** A doesn't know what B knows (because A left). From A's view, B's state is uncertain.

If Interpretation 2, then:
- B could leave anytime (before A, after A, whenever)
- A can't verify B's knowledge state because A isn't there
- The "UNKNOWN" label reflects A's uncertainty, not B's actual state

---

## What Should We Implement?

If the goal is:
1. **A knows B is UNKNOWN** → B must leave before any target events (current behavior)
2. **A is uncertain about B** → B can leave anytime, A just doesn't know B's state

Please clarify which interpretation is correct for these scenarios.
