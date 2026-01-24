# Fix: Extra=1 Scenario Generation

## Problem

The `insert_extra_events()` function in `generate_tom_scenarios_new.py` was generating meaningless Extra=1 scenarios.

**Example of broken Scenario 1 Extra=1:**
```
You leave the room. You enter the room. C puts banana in bag. D leaves. B leaves.
```

**Issues:**
1. Leave happens at position 0 (before any put) - Self leaves before seeing anything, which is meaningless
2. Leave and enter are adjacent with nothing between them - "You leave. You enter." is pointless

## Root Cause

In the original `insert_extra_events()` (lines 380-412):
- `leave_insert_pos = rng.randint(0, constraint_idx)` allowed leave at position 0
- `enter_insert_pos = rng.randint(leave_insert_pos, constraint_idx)` could equal leave_insert_pos, making them adjacent

## Solution

Added two constraints to `insert_extra_events()`:

### Constraint 1: Put before leave (for Believes/Knows characters)

If the leaver's epistemic state contains "Believes" or "Knows" (anything except UNKNOWN), a put to the queried container must occur before their leave event. This ensures the character has actually seen something before leaving.

**Logic:**
- Determine leaver's state from CSV column based on Answerer type
- If state != UNKNOWN, find first put to queried container and set `min_leave_pos = first_put_idx + 1`
- If no valid position exists (min_leave_pos > constraint_idx), don't insert extra events

### Constraint 2: Gap events between leave/enter

At least one event must occur between leave and enter. The type depends on who leaves:

- **If Self leaves**: Gap events must be **leave/enter actions** by other characters (not put/move), since Self wouldn't see put/move while absent
- **If Teammate/Opponent leaves**: Any event type is valid as gap

**Logic:**
- Find valid leave positions that have appropriate gap events after them
- Find valid enter positions that have appropriate gap events between leave and enter
- Use `rng.choice()` to randomly select from valid positions
- If no valid positions exist, don't insert extra events

## Files Modified

### generate_tom_scenarios_new.py

1. **Updated function signature** (line 380):
```python
def insert_extra_events(scenario: Scenario, answerer: str, player: str,
                        answerer_state: EpistemicState, spec: dict, rng: random.Random) -> None:
```
Added `spec` parameter to access the leaver's epistemic state from CSV.

2. **Added constraint logic** (lines 409-487):
   - Determine leaver's CSV state based on Answerer type
   - Calculate min_leave_pos based on first put index (if required)
   - Find valid leave positions with appropriate gap events
   - Find valid enter positions with appropriate gap events
   - Only insert if valid positions exist

3. **Updated call site** (line 536):
```python
insert_extra_events(scenario, answerer, actor, _get_answerer_state(row), row, rng)
```
Added `row` (spec) as parameter.

## Behavior Changes

### Before (broken)
- Scenario 1 Extra=1 (Self=Knows X): `You leave. You enter. C puts banana in bag.`
- Self leaves before seeing anything - violates the "Knows" requirement

### After (fixed)
- Scenario 1 Extra=1 (Self=Knows X): Same as Extra=0 (no extra events inserted)
- No valid position exists for Self to leave/enter while maintaining "Knows X" state
- This is correct - Self must be present the whole time to know the truth

### Scenarios with UNKNOWN state
- Scenario 23 Extra=1 (Teammate=UNKNOWN): `B leaves. C leaves. B enters. ...`
- Leave can be at position 0 since UNKNOWN doesn't require seeing a put
- Gap event (C leaves) exists between B's leave and enter

## Validation

All 78 scenarios generate successfully:
- All `_validate_invariants()` checks pass
- Constraint 1: "Believes" or "Knows" characters have a put before their first leave
- Constraint 2: At least one event between leave and enter
- Constraint 3: When Self leaves, gap events are leave/enter (not put/move)
