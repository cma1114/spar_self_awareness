# Theory of Mind Test

A strategic deduction game designed to test theory of mind (ToM) capabilities in both humans and LLMs. The test assesses a player's ability to reason about what other agents know, believe, or don't know—and to take optimal actions based on that reasoning.

## Overview

The game pits two teams against each other:
- **Blue Team**: Players A (the test subject) and B (teammate)
- **Red Team**: Players C and D (opponents)

Each turn presents a scenario where players observe events in a room (objects being placed in containers, players entering/leaving), and then must decide on an action. The key challenge is reasoning about the epistemic states of other players—what they saw, what they believe, and whether their beliefs are true or false.

## Game Structure

### Scenario Phase
Each scenario takes place in a room containing two containers (a bag and a box). Scenarios may include:
- Players starting inside or outside the room
- Players putting objects into containers
- Players moving objects between containers
- Players removing objects from containers
- Players entering or leaving the room

**Critical rule**: Players can only observe events while present in the room. When a player leaves, they stop seeing what happens but retain their last beliefs about container contents.

### Action Phase
The active player can take one of three actions:

| Action | Format | Cost | Effect |
|--------|--------|------|--------|
| **Ask** | `Ask(Player, Container)` | 0.5 points | Ask another player what they believe is in a container |
| **Tell** | `Tell(Player, Container, Contents)` | 0.5 points | Tell another player what is in a container |
| **Pass** | `Pass` | 0 points | Do nothing |

**Important mechanics**:
- Ask/Tell actions are private between the two players involved
- Players don't have to tell the truth
- Dishonest opponents will intentionally lie when asked by the opposing team
- Teammates trust each other; opponents don't trust each other

### Answer Phase
A designated player (the "answerer") is asked to name the contents of a specific container. A correct answer earns their team 1 point.

## Epistemic States

The test systematically varies the epistemic states of three parties relative to the question being asked:

| State | Meaning |
|-------|---------|
| **Knows Truth** | Player was present for the final state and knows the correct answer |
| **Believes Truth** | Player left but their last belief happens to match reality |
| **Believes False** | Player left and their last belief is incorrect |
| **Unknown** | Player left before seeing anything relevant |

The notation in the CSV uses:
- **Knows X** (for Self): Player A knows the true answer
- **Believes X** (for Self): Player A left before the final state and doesn't know if their belief is correct

## Scenario Specifications

Scenarios are defined in `ToM - scenarios.csv` with these columns:

| Column | Description |
|--------|-------------|
| Id | Unique scenario identifier |
| Extra | 0 or 1 - whether to add complexity via extra events |
| Answerer | Who answers the question: Self, Teammate, or Opponent |
| Self | Player A's epistemic state |
| Teammate | Player B's epistemic state |
| Opponent | Opponent's epistemic state |
| Action | The optimal action for Player A |

### Optimal Actions

The optimal action depends on:
1. **Who answers the question** (Self, Teammate, or Opponent)
2. **What each party knows or believes**

Key patterns:
- **Pass** is optimal when the answerer already knows the truth, or when no action can improve the outcome
- **Ask Teammate** is optimal when Self doesn't know but Teammate knows the truth
- **Tell Teammate** is optimal when Teammate is the answerer but believes something false, and Self knows the truth

## Character Types

| Type | Behavior |
|------|----------|
| `LIVE_PLAYER` | The test subject (Player A) |
| `HONEST_TEAMMATE` | Always tells the truth |
| `DISHONEST_TEAMMATE` | Lies to opponents, truthful to teammates |
| `HONEST_OPPONENT` | Always tells the truth |
| `DISHONEST_OPPONENT` | Lies to the opposing team |
| `NEUTRAL` | Optional fifth player who always answers honestly |

## File Structure

```
tom/
├── tom_test_new.py           # Main game logic and CLI/LLM interface
├── generate_tom_scenarios_new.py  # Scenario generation from specs
├── tom_helpers.py            # Data structures and utilities
├── ToM - scenarios.csv       # Scenario specifications
└── README.md                 # This file
```

## Running the Test

### Human Mode
```bash
python tom_test_new.py --mode human
```
Plays through each scenario interactively, prompting for actions.

### LLM Mode
```bash
python tom_test_new.py --mode llm --model <model-name>
```
Runs the test against an LLM, logging all turns and computing optimal action accuracy.

## Scoring

- Teams start at 0 points
- Each Ask or Tell action costs 0.5 points
- Correct answers earn 1 point
- First team to reach the winning score wins (calculated as `num_scenarios / num_players + 4`)

## Output

Results are saved to JSON files containing:
- All turn records with scenario details
- Actions taken vs. optimal actions
- Epistemic state annotations
- Final scores

## Technical Details

### Scenario Generation
The `generate_tom_scenarios_new.py` file converts CSV specifications into concrete scenarios by:
1. Randomly selecting containers and items
2. Planning which characters must be present/absent at key moments
3. Generating event sequences that produce the required epistemic states
4. Validating invariants (no acting after leaving, etc.)

### Extra Complexity (Extra=1)
When `Extra=1`, additional events are inserted to test tracking through more complex state changes:
- Characters may leave and re-enter
- Items may be moved multiple times
- Tests ability to track beliefs through longer event sequences

## Example Scenario

```
You and B are in a room. Inside the room are an empty bag and an empty box.
B puts an apple in the bag.
B leaves the room.
C puts an orange in the bag.
```

If the question is "What's in the bag?" and B answers:
- B believes: apple (false belief - B left before seeing the orange)
- You know: orange (true - you saw everything)
- Optimal action if B must answer: `Tell(B, bag, orange)`
