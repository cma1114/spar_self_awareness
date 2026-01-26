# ToM Test Changelog

## 2026-01-26: Add Extra Events for Self=Believes X Scenarios

### Summary
Added dedicated `insert_extra_events_believes_x()` function to insert extra events for Self=Believes X scenarios (10-13). Previously, Extra=1 versions of these scenarios were identical to Extra=0.

### Problem
Scenarios 10-13 (all have Self=Believes X) produced **identical** Extra=0 and Extra=1 outputs because the dispatch logic didn't handle BELIEVES_X answerer state:

**Dispatch flow:**
1. `_get_answerer_state()` returns BELIEVES_X for Self=Believes X
2. But dispatch checked for KNOWS_TRUTH, BELIEVES_TRUTH, BELIEVES_FALSE - not BELIEVES_X
3. Fell through to generic `insert_extra_events()` which silently failed due to constraint violations

**Root cause in `_get_answerer_state()`:**
```python
if spec['Answerer'] == 'Self':
    if spec['KS_Self'] == EpistemicState.KNOWS_X:
        return EpistemicState.KNOWS_TRUTH
    else:  # BELIEVES_X
        return EpistemicState.UNKNOWN  # BUG: Should return BELIEVES_X
```

### Solution

1. **Fixed `_get_answerer_state()`** to return `BELIEVES_X` instead of `UNKNOWN` for Self=Believes X:
```python
else:  # BELIEVES_X
    return EpistemicState.BELIEVES_X
```

2. **Added `insert_extra_events_believes_x()` function** (~lines 800-855):
   - Finds where answerer (Self) leaves
   - Inserts move_away and move_back events after Self leaves
   - Self doesn't see these moves, maintaining their uncertain belief
   - Adds 2 events to increase scenario complexity

3. **Added dispatch case for BELIEVES_X** (~line 993):
```python
elif answerer_state == EpistemicState.BELIEVES_X:
    insert_extra_events_believes_x(scenario, answerer, rng)
```

### Example: Scenario 11 Extra=1

**Before fix (identical to Extra=0):**
```
You, B, C, and D are in a room. Inside the room are an empty bag and an empty box.
C puts an orange in the bag. ... You leave the room. ...
```

**After fix:**
```
You, B, C, and D are in a room. Inside the room are an empty bag and an empty box.
C puts an orange in the bag. ... You leave the room. ...
[2 move events happen after Self leaves - not visible to Self]
```

### Files Modified
- **generate_tom_scenarios_new.py**:
  - Fixed `_get_answerer_state()` (line 388)
  - Added `insert_extra_events_believes_x()` function (lines 800-855)
  - Added dispatch case for BELIEVES_X (line 993)

### Verification
- All 78 scenarios generate successfully across 5 seeds
- All Self=Believes X scenarios (10-13) Extra=1 now have at least 2 move events after answerer leaves
- Extra=1 scenarios average 7.1 events vs 4.0 for Extra=0

---

## 2025-01-26: Fix Self=Believes X Scenarios Missing Put Actions

### Summary
Fixed bug where Self=Believes X scenarios could have no put/move actions, producing broken output with only leave events.

### Problem
Scenario 12 Extra=1 (and similar Self=Believes X scenarios) sometimes produced:
```
You, B, C, and D are in a room. Inside the room are an empty bag and an empty box.
You leave the room. ... C leaves the room. ... D leaves the room. ...
```

No put action was generated, so Self couldn't form any belief.

### Root Cause
In `build_scenario()`, characters in `self.exclude` (including Self with Believes X) were randomly distributed into three groups:
- `exclude_false` (33%) - leaves after first put
- `exclude_true` (33%) - leaves after second put
- `leave_immediately_group` (33%) - leaves before any put

When Self ended up in `leave_immediately_group`, they left before any put action. But "Believes X" requires seeing a put to form a belief, making the scenario logically invalid.

### Solution
Added constraint in `build_scenario()` (line ~243): if Self is in `leave_immediately_group` and in `self.exclude`, move them to `exclude_false` so they see at least the first put before leaving.

### Files Modified
- **generate_tom_scenarios_new.py**: Added Self=Believes X constraint after exclude distribution loop

### Verification
- All 78 scenarios now generate with put/move actions
- Scenario 12 Extra=1 tested across 5 seeds - all have puts before Self leaves

---

## 2025-01-26: Alternate Actors for Consecutive Actions

### Summary
Extra=1 scenarios now prefer different characters for consecutive actions, avoiding unrealistic sequences where the same character performs many actions in a row.

### Problem
Insert functions selected actors independently via `rng.choice()`, allowing the same character to perform 5+ consecutive actions:
```
You remove the stapler from the box.
You move the brick from the bag to the box.
You put a ball in the bag.
You remove the ball from the bag.
You move the brick from the box to the bag.
```

### Solution
Added `_choose_different_actor()` helper function that prefers a different character from the last one when multiple characters are present.

**New behavior:**
```
You remove the stapler from the box.
C moves the brick from the bag to the box.
You put a ball in the bag.
C removes the ball from the bag.
You move the brick from the box to the bag.
```

### Files Modified
- **generate_tom_scenarios_new.py**:
  - Added `_choose_different_actor()` helper function
  - `insert_extra_puts()`: Use alternating actor selection for 5 events
  - `insert_extra_events_with_revelation()`: Make mover2 different from mover1
  - `insert_extra_events_believes_true()`: Make mover2 different from mover1
  - `insert_extra_events_believes_false()`: Make mover2 different from mover1

### Notes
When only one character remains present (all others have left), that character must do all actions - this is unavoidable and correct behavior.

---

## 2025-01-26: Add Random Character Leave Events While Player Away

### Summary
Extra=1 scenarios now have a 50% chance to include other characters leaving the room while the player is away. Previously, player leave/enter events were always adjacent with no other character movement between them.

### Problem
Extra=1 insert functions created rigid sequences:
```
Player leaves → Move away → Player enters → Move back
```

This was unrealistic - other characters should occasionally leave while the player is away.

### Solution
Modified all three Extra=1 insert functions to optionally add character leave events:

**New sequence (50% probability):**
```
Player leaves → Random character leaves → Move away → Player enters → Move back
```

**Constraints:**
- Must keep at least one character present to perform moves
- Cannot make mover characters leave
- Cannot make characters leave who have events later in the base scenario

### Example
```
Before: You leave the room. You enter the room. D moves the ball from the box to the bag.
After:  You leave the room. C leaves the room. You enter the room. D moves the ball from the box to the bag.
```

### Files Modified
- **generate_tom_scenarios_new.py**:
  - `insert_extra_events_with_revelation()`: Added random character leave logic
  - `insert_extra_events_believes_true()`: Added random character leave logic
  - `insert_extra_events_believes_false()`: Added random character leave logic

### Verification
Tested across multiple seeds - scenarios now show variety in character movement patterns.

---

## 2025-01-25: Add Ellipsis Mode

### Summary
Added `ELLIPSIS_MODE` parameter that adds "..." after every action (put/move/leave/enter) to give models a visual sense of time passing between events.

### Configuration
Set `ELLIPSIS_MODE` in `tom_test_new.py` (line 60):
- `False` (default): No ellipses
- `True`: Add "..." after every action

When `ELLIPSIS_MODE=True`, `PAUSE_MODE` is forced to "none" (no "Time goes by.").

### Example Output
```
Without: B puts a ball in the box. You leave the room. You enter the room.
With:    B puts a ball in the box. ... You leave the room. ... You enter the room. ...
```

### Files Modified
- **tom_test_new.py**: Added `ELLIPSIS_MODE` constant, passed to `get_description_for()`
- **tom_helpers.py**: Added `ellipsis_mode` parameter, appends "..." after each action

---

## 2025-01-25: Fix Move Event Description Rendering

### Problem
Move events in scenario descriptions only showed the destination, not the source:
```
B moves the banana to the bag.
```
This was confusing in Extra=1 scenarios where the player leaves and returns - they'd see a move "to the bag" when they last saw the item already in the bag.

### Solution
Updated `get_description_for()` in `tom_helpers.py` to include `from_container` when rendering move events:

**Before:**
```python
lines.append(f"{actor} {verb_move} the {event.item} to the {event.to_container}.")
```

**After:**
```python
if event.from_container and event.to_container:
    lines.append(f"{actor} {verb_move} the {event.item} from the {event.from_container} to the {event.to_container}.")
else:
    lines.append(f"{actor} {verb_move} the {event.item} to the {event.to_container}.")
```

### Example
```
Before: D moves the ball to the box.
After:  D moves the ball from the bag to the box.
```

### Files Modified
- **tom_helpers.py**: Updated move event rendering (line 155)

---

## 2025-01-25: Fix Extra=1 Generation for 16 Failing Scenarios

### Problem
16 scenario types produced **identical** Extra=0 and Extra=1 outputs because the existing `insert_extra_events()` logic couldn't insert meaningful events for these scenarios.

**Affected scenarios by answerer state:**
- KNOWS_TRUTH: 1, 3, 7, 9, 20, 22, 33, 35 (8 scenarios)
- BELIEVES_TRUTH: 27, 29 (2 scenarios)
- BELIEVES_FALSE: 17, 18, 19, 30, 31, 32 (6 scenarios)

### Epistemic Model
Adopted a new epistemic model for Extra=1 scenarios:
- Containers are opaque - you can't see inside just by being present
- You learn contents ONLY by witnessing put/move events
- Leaving creates uncertainty (things may change while away)
- Returning doesn't automatically restore knowledge
- A put/move witnessed after returning reveals the truth

### Solution

**1. New insert functions for each answerer state:**

| Function | Answerer State | Pattern |
|----------|---------------|---------|
| `insert_extra_events_with_revelation()` | KNOWS_TRUTH | Answerer leaves → state changes → returns → witnesses final move (revelation) |
| `insert_extra_events_believes_true()` | BELIEVES_TRUTH | Answerer leaves → state changes and reverts → belief still matches truth |
| `insert_extra_events_believes_false()` | BELIEVES_FALSE | Answerer leaves → more events happen → belief still doesn't match truth |

**2. Updated dispatch logic** to route to appropriate function based on `_get_answerer_state()`.

**3. Fixed `set_phase` constraint bug** (line 281): Added `self_phase < 3` check. Previously, when Self wasn't leaving (phase 3), the constraint incorrectly moved opponents out of `exclude_false`, causing them not to leave when they should have.

### Example: Scenario 7 (KNOWS_TRUTH)
```
Extra=0: D puts orange in bag. C leaves.
Extra=1: D puts orange in bag. You leave. C moves orange to box. You enter. C moves orange to bag. C leaves.
```

Epistemic trace for Extra=1:
1. See put → know orange in bag
2. Leave → lose certainty
3. Miss move to box (was away)
4. Return → still don't know (containers opaque)
5. See move to bag → NOW KNOW truth

### Files Modified
- **generate_tom_scenarios_new.py**:
  - Added `insert_extra_events_with_revelation()` (lines 541-607)
  - Added `insert_extra_events_believes_true()` (lines 610-656)
  - Added `insert_extra_events_believes_false()` (lines 659-715)
  - Updated dispatch logic (lines 662-672)
  - Fixed `set_phase` constraint (line 281)

### Verification
All 16 scenarios produce different Extra=0 vs Extra=1 outputs across 10 random seeds.

---

## 2025-01-25: Add "Time goes by." Pause Mode

### Summary
Added a configurable pause text after "You leave the room." to clarify that events can happen while the player is away.

### Configuration
Set `PAUSE_MODE` in `tom_test_new.py` (line 56):

| Value | Behavior |
|-------|----------|
| `"none"` | Default - no change to output |
| `"extra1"` | Add "Time goes by." only in Extra=1 scenarios |
| `"all"` | Add "Time goes by." in all scenarios where player leaves |

### Example Output
**Before:**
```
You leave the room. You enter the room. C puts an apple in the bag.
```

**After (with pause_mode="extra1" on Extra=1 scenario):**
```
You leave the room. Time goes by. You enter the room. C puts an apple in the bag.
```

### Files Modified
- **tom_helpers.py**: Added `pause_mode` parameter to `get_description_for()` method
- **tom_test_new.py**: Added `PAUSE_MODE` constant, `pause_mode` field in `TurnRecord`, updated call site

### Logging
The `pause_mode` setting is recorded in `game_data.json` for each turn.

---

## 2025-01-25: Extra=1 Error Analysis

### Summary
Extended `analyze_errors.py` to run parallel analysis for Extra=1 scenarios.

### Output Files
The script now generates both Extra=0 and Extra=1 analysis files:

| Extra=0 | Extra=1 |
|---------|---------|
| `error_analysis.txt` | `error_analysis_extra1.txt` |
| `error_analysis_thinking.txt` | `error_analysis_thinking_extra1.txt` |
| `error_analysis_nonthinking.txt` | `error_analysis_nonthinking_extra1.txt` |
| `scenario_success_matrix.csv` | `scenario_success_matrix_extra1.csv` |

### Files Modified
- **analyze_errors.py**: Refactored into `run_analysis_for_extra()` helper, parameterized filter function

---

## 2025-01-25: Scenario 12 Both-Opponents-Leave-Before-Self Bug

### Problem
In Scenario 12 (Self=Believes X, Teammate=Knows Truth, Opponent=Unknown), both opponents could randomly leave the room before Self leaves, creating unrealistic scenarios.

**Constraint violated:** At least one opponent must still be present when Self leaves.

### Root Cause
1. **Phase assignment**: Each excluded character independently assigned to a leave phase with 33% probability. Both opponents could end up in earlier phases than Self.
2. **Within-phase ordering**: Even within the same phase, characters leave in random order. An opponent in the same phase as Self might leave first.

### Solution
1. Added character reference storage (`self_char`, `opponent1`, `opponent2`) to `Scenario_Builder`
2. Added phase constraint: if both opponents would leave before Self, move one to Self's phase
3. Added `order_self_first()` helper: within Self's phase, Self leaves first

### Files Modified
- **generate_tom_scenarios_new.py**: Lines 57-60, 102-105, ~250-270

### Verification
100/100 Scenario 12 instances passed - all have at least one opponent present when Self leaves.

---

## 2025-01-25: Extra=1 Scenario Generation Fix

### Problem
The `insert_extra_events()` function was generating meaningless Extra=1 scenarios.

**Example of broken output:**
```
You leave the room. You enter the room. C puts banana in bag.
```

**Issues:**
1. Leave at position 0 (before any put) - Self leaves before seeing anything
2. Leave and enter adjacent with nothing between them

### Root Cause
- `leave_insert_pos = rng.randint(0, constraint_idx)` allowed leave at position 0
- `enter_insert_pos = rng.randint(leave_insert_pos, constraint_idx)` could equal leave_insert_pos

### Solution
Added two constraints to `insert_extra_events()`:

1. **Put before leave**: For "Believes" or "Knows" characters, a put to the queried container must occur before their leave event.

2. **Gap events between leave/enter**: At least one event must occur between leave and enter.
   - If Self leaves: gap must be leave/enter actions (not put/move, since Self wouldn't see them)
   - If Teammate/Opponent leaves: any event type is valid

### Files Modified
- **generate_tom_scenarios_new.py**: Updated `insert_extra_events()` signature and logic (lines 380-487, 536)

### Verification
All 78 scenarios generate successfully with proper constraints.
