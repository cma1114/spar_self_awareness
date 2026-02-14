# Design Decisions: Extra=0 vs Extra=1 Scenario Generation

## Problem: ECT/SIT Parity Failures

### Background

The ToM test uses paired scenarios: Extra=0 (baseline) and Extra=1 (increased epistemic complexity). The goal is for both versions to have:
- **Similar situation tracking demands** (visible event counts within a small range)
- **Different epistemic complexity** (Extra=1 always has more ECT than Extra=0)

### The Filler Approach (Failed)

The initial approach added "filler events" to Extra=0 scenarios to match Extra=1's event count. However, this produced:
- **5 scenarios where Extra=1 ECT <= Extra=0 ECT** (IDs 10, 11, 12, 36, 38)
- **4 scenarios with SIT gap > 3** (IDs 11, 12, 23, 30)

### Root Cause

The CSV file lists all Extra=0 specs first (IDs 1-39), then all Extra=1 specs (IDs 1-39). The generation loop processes specs sequentially, consuming RNG values as it goes. By the time Extra=1 ID=10 is generated, it has a completely different RNG state than Extra=0 ID=10.

This produces **fundamentally different base scenarios** for the same ID:
- Different characters present initially
- Different event orderings
- Different container assignments

No amount of filler can guarantee the ECT relationship when the bases diverge randomly.

## Solution: Shared-Base Generation

### Design Decision

**Chosen approach: Shared-Base Generation**

For each scenario ID that has both Extra=0 and Extra=1 versions:
1. Generate the base scenario once
2. Deep-copy the base before adding filler/extra events
3. Extra=0 gets filler events (situation tracking without ECT)
4. Extra=1 gets extra events (adds ECT by design)

This guarantees:
- **ECT relationship**: Extra=1 = Extra=0 base + ECT-adding events
- **SIT parity**: Same base = same visible events, plus controlled additions

### Alternative Considered: Rejection Sampling

Keep generating until Extra=1 ECT > Extra=0 ECT and SIT within threshold.

**Rejected because:**
- Variable convergence time
- No guarantee of fast convergence for some scenarios
- Fighting randomness rather than controlling it
- Less architecturally clean

### Mitigating "Obviousness"

Concern: Shared-base scenarios may look too similar to players, making it obvious they're paired.

**Mitigation via cosmetic variation:**
- **Container names**: 50% chance to swap bag<->box between versions
- **Object names**: Always use a different item name for the queried object

**NOT varied (to avoid side effects):**
- Character names (could break game logic)
- Event ordering (could have unpredictable ECT effects)

## Processing Order

Output scenarios in paired order for easier comparison:
- ID 1, Extra=0
- ID 1, Extra=1
- ID 2, Extra=0
- ID 2, Extra=1
- ...

This is achieved by grouping specs by ID before processing, rather than processing in CSV order.

## SIT Gap Fix: Rejection Sampling

### Problem: ID 18 SIT Gap

After implementing shared-base generation, ID 18 still had SIT gap > 3 (Extra=1 had 2-5 fewer visible events than Extra=0).

**Root cause**: For scenarios where player A leaves early (e.g., KNOWS_X state), the extra events happen while A is absent. Put/move/remove events are only visible when A is present; leave/enter events are always visible. If the extra events function doesn't add a leave/enter pair (e.g., 67% chance in some functions), Extra=1 may have fewer visible events than Extra=0's filler events.

### Solution: Rejection Sampling

Rather than modifying individual extra events functions (which could have downstream effects), we use **rejection sampling** at the generation level:

```python
MAX_SIT_GAP = 3
MAX_RETRIES = 10

for retry in range(MAX_RETRIES):
    scenario_e1 = generate_extra1_scenario(base_scenario, retry_rng)
    sit_gap = abs(scenario_e1.situation_event_count - e0_sit)
    if sit_gap <= MAX_SIT_GAP:
        break
```

This approach:
- Keeps the core extra events logic unchanged
- Only retries when the SIT gap exceeds the threshold
- Uses a different RNG seed for each retry to get variation
- Warns if max retries reached without satisfying the constraint

## Implementation Details

See `generate_tom_scenarios_new.py`:
- `generate_scenarios_from_tuples()`: Groups specs by ID, generates shared base, uses rejection sampling for SIT parity
- `apply_name_variation()`: Applies cosmetic changes to Extra=1 copy
- `insert_extra_events_believes_true()`: Includes mandatory leave/enter pair to improve SIT parity

## Third Container (Basket)

Added a third container to eliminate nonsensical 'remove' events.

**Problem**: With only 2 containers (bag, box), filler events couldn't use moves without touching the target container. This led to put/remove cycles like "A puts ball in box. A removes ball from box."

**Solution**: Add 'basket' as third container. Filler now uses moves between non-target containers:
- If target is 'bag', filler moves items between 'box' and 'basket'
- No 'remove' events - items are always moved to another container

**Validation**: `_validate_invariants()` raises an error if any 'remove' event is found.

## Item Variation in Revelation Events

For KNOWS_TRUTH scenarios (where the answerer returns and witnesses the final state), the revelation event now varies the item 50% of the time.

**Rationale**: Certainty ECTs (knowledge ↔ belief) depend on presence/absence, not the specific item. When the player:
1. Sees apple in bag
2. Leaves (knowledge → belief)
3. Returns (belief → knowledge)

...the ECT count is the same whether they see apple or banana when they return.

**Implementation** in `insert_extra_events_with_revelation()`:
- 50% of time: Move original item back (standard)
- 50% of time: Put a different item in target container (variation)

This adds variety without affecting ECT metrics.

## Verification

Run `analyze_epistemic_transitions.py` to confirm:
- Extra=1 ECT > Extra=0 ECT for ALL scenarios, ALL seeds
- SIT gap <= 3 for all scenarios (ideally <= 1)
- Zero 'remove' events in generated scenarios

---

## ECT #2 Witnessing Fix

### Problem: Answerer Not Witnessing Events

The `insert_extra_events_believes_*()` functions added answerer enter/leave events, but placed the enter AFTER the moves on the target container. This meant the answerer never witnessed any events and gained zero ECTs.

**EPISTEMIC_METRICS.md defines ECT #2 as:**
> Character (who previously had belief) **sees** a put/move on target container → belief → knowledge

When A is inside observing, entering a room does NOT give knowledge. You must WITNESS an event.
(See EPISTEMIC_METRICS.md for the full rule including when A is outside.)

### Solution

Moved the answerer's enter event BEFORE the moves so they witness the changes:

```python
# Before (wrong):
events_to_insert.append(move_away)
events_to_insert.append(move_back)
events_to_insert.append(Event('enter', answerer))  # After moves - sees nothing!
events_to_insert.append(Event('leave', answerer))

# After (correct):
events_to_insert.append(Event('enter', answerer))  # Before moves
events_to_insert.append(move_away)                 # Answerer WITNESSES this → ECT #2
events_to_insert.append(move_back)                 # Answerer WITNESSES this
events_to_insert.append(Event('leave', answerer))  # ECT #1 when they leave
```

Fixed in: `insert_extra_events_believes_false()`, `insert_extra_events_believes_true()`, `insert_extra_events_believes_x()`

---

## Teammate Belief Integrity

### Problem: Blue Team Invalidation

Player A's actions could invalidate teammate B's belief about the target container (or vice versa). This violates the test design where the player should never undermine their own teammate.

### Solution

1. **Helper function** `_get_teammates_to_exclude_for_target_moves()`: Tracks which blue team members left with beliefs about the target and returns their teammates for exclusion from target modifications.

2. **Validation function** `_validate_teammate_belief_integrity()`: Checks for violations and logs warnings (some specs have inherent conflicts that cannot be resolved).

3. **Generation constraints**: Updated `build_scenario()`, `insert_extra_puts()`, and `insert_extra_events_*` functions to exclude blue team members from modifying the target when their teammate left with a belief.

### Design Decision: Team-Based Protection

- **Blue team (A/B) protected**: A cannot invalidate B's belief, B cannot invalidate A's belief
- **Red team (C/D) NOT protected**: Opponents can invalidate each other's beliefs
- **3 unavoidable violations**: Some specs (e.g., ID 12) have inherent conflicts where only one character can act and they must modify the target. These are logged as warnings.

---

## Event Count Variance

Event counts **vary** across repetitions due to:

1. **Initial presence**: 50% chance for optional characters to be present
2. **Belief group assignment**: 33/33/33% for exclude/exclude_true/leave_immediately
3. **Container/item selection**: Random choice affects ECT relevance
4. **Insert position randomness**: Where extra events go affects visibility

**Recommendation**: Use empirical measurement (generate N scenarios per spec, compute min/max/mean/stddev) rather than analytical enumeration.
