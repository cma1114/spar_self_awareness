# Issue: Missing 1B Scenario for ID 37

## Status: RESOLVED

## Problem Statement
When running `python pre_generate_scenarios.py --reps 10`, only 1039 of 1040 expected scenarios are generated. One scenario (ID 37 Extra=1B) is skipped with the warning:
```
ID 37 Extra=1B: Could not add extra events after 10 retries (containers used by later events). Skipping 1B variant.
```

## Resolution

### Root Cause
The failure occurred at **line 1162** (no valid movers), NOT at container selection as initially hypothesized.

Debug output revealed:
```
DEBUG ID37 1B: line 1162 - no valid movers. present_after_leave={'A', 'B'}, teammates_to_exclude={'A', 'B'}
```

The issue: When both A and B were present after the answerer (C or D) leaves, and both were in `teammates_to_exclude`, there were no valid movers to perform the revelation events.

### Fix Applied
Added fallback logic in `insert_extra_events_with_revelation()` (around line 1167):

```python
valid_movers = [c for c in sorted(present_after_leave) if c not in teammates_to_exclude]
if not valid_movers:
    # Fallback: For the revelation pattern, the moves are temporary (item returns to target
    # before any teammates leave). The base scenario's subsequent events will set the final
    # state. So we can use any present character as a mover - the validation will catch
    # any actual violations later.
    if present_after_leave:
        valid_movers = sorted(present_after_leave)
    else:
        return  # No one present to do the moves
```

### Why This Works
The `teammates_to_exclude` check was too conservative for the revelation pattern because:
1. It's computed from the FULL base scenario (who leaves with beliefs)
2. But the revelation pattern inserts events BETWEEN the initial PUT and subsequent base events
3. The base scenario's MOVE and final PUT still run afterwards, setting the final state
4. The validation catches any actual belief violations

### Verification
```
Generated 1040 scenarios to scenarios_standardized.json
  104 specs × 10 reps = 1040 expected
  By Extra: {'0A': 260, '0B': 260, '1A': 260, '1B': 260}

=== Violation Summary ===
Teammate belief violations: 0
SIT gap violations (>3): 0
```

## Previous Work (Still Valid)

### SIT Gap Fixes (COMPLETED - Working)
1. Added `compute_filler_capacity()` helper function (~line 2044)
2. Modified `insert_extra_events_believes_x()` to accept `skip_optional_events` parameter
3. Modified `insert_extra_events_with_revelation()` to accept `skip_optional_events` parameter
4. Modified call site in `generate_scenarios_from_tuples()` to compute skip flags based on filler capacity
5. Result: SIT gap violations reduced from 16 to 0

## ID 37 Specification
```
Id=37, Answerer=Opponent, KS_Self=BELIEVES_X, KS_Teammate=UNKNOWN, KS_Opponent=KNOWS_TRUTH, Action=Pass
```

## Key Files
- `/tom/generate_tom_scenarios_new.py`:
  - `insert_extra_events_with_revelation()`: lines 1083-1270 (fix applied here)
  - Retry loop in `generate_scenarios_from_tuples()`: lines 2496-2615
