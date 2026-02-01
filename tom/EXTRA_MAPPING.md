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

## Summary - Category Names

| Code | Display Name | Description |
|------|--------------|-------------|
| 0A | **Minimal Events** | NEW - No filler events, no ECT |
| 0B | **Event Load** | NEW - Higher event load (+3 filler), no ECT |
| 1A | **Minimal ECT** | Was Extra=0 - Filler events for SIT parity, has ECT |
| 1B | **ECT Load** | Was Extra=1 - Extra events with ECT and more complexity |

## Color Coding (for charts)

```python
EXTRA_CATEGORIES = {
    '0A': {'name': 'Minimal Events', 'short': 'Min Events', 'color': '#9b59b6'},  # Purple
    '0B': {'name': 'Event Load', 'short': 'Event Load', 'color': '#f39c12'},      # Orange
    '1A': {'name': 'Minimal ECT', 'short': 'Min ECT', 'color': '#3498db'},        # Blue
    '1B': {'name': 'ECT Load', 'short': 'ECT Load', 'color': '#e74c3c'},          # Red
}
```
