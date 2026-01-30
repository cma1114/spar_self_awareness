# Extra Field Mapping - MUST READ BEFORE CODE CHANGES

## Critical Invariant

**Existing behavior must NOT change.**

| Old Value | New Value | Generation Function | Behavior |
|-----------|-----------|---------------------|----------|
| `Extra=0` or `None` | `'1A'` | `insert_filler_events()` | UNCHANGED |
| `Extra=1` | `'1B'` | `insert_extra_events_*()` + `apply_name_variation()` | UNCHANGED |

## New Scenarios (Event Load Experiment)

| New Value | Generation Function | Behavior |
|-----------|---------------------|----------|
| `'0A'` | `insert_n_filler_events(n=0)` | Minimal events, no ECT |
| `'0B'` | `insert_n_filler_events(n=3)` | Higher load, no ECT |

## Comparison Mapping Rules

**Every `== 0` must become `== '1A'`**
**Every `== 1` must become `== '1B'`**
**`pause_mode == "extra1"` checks use `== '1B'` only (not `in ('1A', '1B')`)**

## Backward Compatibility

```python
def normalize_extra(val):
    """Convert legacy Extra values to new string format."""
    if val is None or val == 0: return '1A'  # Legacy Extra=0 → 1A
    if val == 1: return '1B'                  # Legacy Extra=1 → 1B
    if val in ('0A', '0B', '1A', '1B'): return val
    return str(val)
```

## Summary

- **0A**: NEW - Minimal event load (no filler)
- **0B**: NEW - Higher event load (+3 filler)
- **1A**: Was Extra=0 - filler for SIT parity
- **1B**: Was Extra=1 - extra events with ECT
