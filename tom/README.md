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
| **Unknown** | A doesn't know whether this player's belief is true or false (player is out, and A cannot verify their belief accuracy) |

The notation in the CSV uses:
- **Knows X** (for Self): Player A knows the true answer
- **Believes X** (for Self): Player A left before the final state and doesn't know if their belief is correct

## Scenario Specifications

Scenarios are defined in `ToM - scenarios.csv` with these columns:

| Column | Description |
|--------|-------------|
| Id | Unique scenario identifier |
| Extra | 0A/B or 1A/B - whether to add complexity via extra events/transitions |
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
├── tom_test_new.py               # Main game logic and CLI/LLM interface
├── generate_tom_scenarios_new.py # Scenario generation from specs
├── tom_helpers.py                # Data structures and utilities
├── ToM - scenarios.csv           # Scenario specifications
├── analyze_results.py            # ToM mastery category scoring
├── analyze_errors.py             # Error analysis (action confusion, features)
├── analyze_reasoning.py          # LLM reasoning analysis
├── compare_old_new.py            # Pre/post bug-fix comparison
├── README.md                     # This file
├── OBSERVATIONS.md               # Analysis observations and insights
├── CHANGELOG.md                  # Change history and bug fixes
└── tom_llm_logs/                 # Test results and analysis outputs
    ├── *_game_data.json          # Per-model test results
    ├── *.log                     # Raw LLM interaction logs
    ├── error_analysis.txt        # Combined error analysis
    ├── error_analysis_thinking.txt    # Thinking models only
    └── error_analysis_nonthinking.txt # Non-thinking models only
```

## Running the Test

### Human Mode
```bash
python tom_test_new.py --mode human
```
Plays through each scenario interactively, prompting for actions.

### LLM Mode
```bash
python tom_test_new.py --mode llm --model <model-name> [--free_response] [--history_mode none|full|actions] [--reps N]
```

**Options:**
- `--free_response`: Allow open-ended text responses (default: multiple choice format)
- `--history_mode`: How much game history to include in prompts (`none`, `full`, `actions`)
- `--reps`: Number of repetitions per scenario (default: 1)

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

### Extra Categories

The `Extra` field controls scenario complexity with four categories:

| Code | Name | Description |
|------|------|-------------|
| **0A** | Minimal Events | No filler events, no Epistemic Category Transitions (ECT) |
| **0B** | Event Load | Higher event load (+3 filler events), no ECT |
| **1A** | Minimal ECT | Filler events for SIT parity, includes ECT |
| **1B** | ECT Load | Extra events with ECT and more complexity |

**ECT (Epistemic Category Transitions)**: Events that change a character's epistemic state category (e.g., from "Knows Truth" to "Believes False" when they leave before a move).

Legacy data may use `Extra=0` (equivalent to 1A) or `Extra=1` (equivalent to 1B).

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

## Analysis Tools

### Results Analysis (`analyze_results.py`)

```bash
python analyze_results.py [--lies_okay]
```

**Options:**
- `--lies_okay`: Count lying to opponent answerer as success (strategic deception)

**Output:**
- **Tables**: Summary by model, Extra category, and free_response mode (saved to `summary_table.txt`)
- **Charts**: Performance comparison, scenario difficulty, Extra category comparison
  - Generates separate charts for Free Response (`_fr_true.png`) and Multiple Choice (`_fr_false.png`) modes
- **Mastery scores**: CSV and text files with ToM category breakdowns

**Free Response Mode:**
When `free_response=True`, the model provides open-ended text responses. When `free_response=False`, the model selects from multiple choice options. Analysis automatically breaks down results by mode when both exist.

**Lies Okay Mode:**
The `--lies_okay` flag treats lying to an opponent answerer as a success. This reflects strategic deception: if an opponent will answer the question and believes something false, telling them a lie (confirming their false belief or adding confusion) is arguably strategic, even if the "optimal" action is Pass. Records where `lied_to_opponent_answerer=TRUE` are counted as successes alongside `was_optimal=TRUE`.

### ToM Mastery Categories

Beyond raw accuracy, the test evaluates performance across six ToM mastery categories that measure distinct cognitive capabilities:

| Category | What It Measures | Key Scenarios |
|----------|------------------|---------------|
| **Self: Knowledge vs Belief** | Distinguishing what you know from what you merely believe | 7-9 (Pass), 12-13 (Ask) |
| **Teammate: Knowledge vs Belief** | Recognizing when teammate knows vs believes | 20-22 (Pass), 17-19 (Tell) |
| **Combined Uncertainty** | Handling situations where both self and teammate are uncertain | 10-11, 23-24 (Pass) |
| **True vs False Belief** | Distinguishing teammate's true belief from false belief | 14-16 (Pass), 17-19 (Tell) |
| **Teammate vs Opponent** | Treating teammate differently from opponent | 12-13 (Ask), 30-32, 37, 39 (Pass), 17-19 (Tell) |
| **Strategic Lies** | Knowing when lying is effective vs. unnecessary | 27-29 (Lie), 30-32 (Pass) |

**Strategic Lies Category:**
This category contrasts scenarios where strategic deception would be effective vs. unnecessary:
- **27-29**: Opponent believes **truth** - lying could deceive them into doubting their correct belief
- **30-32**: Opponent believes **false** - no intervention helps (they'll answer wrong anyway)

Mastery scores use weighted components to reflect the relative difficulty and importance of each sub-skill.

### Error Analysis (`analyze_errors.py`)

Analyzes WHY models fail on specific categories:

1. **Action Confusion Matrix**: What action did failures choose instead of the optimal action?
   ```
   Expected: Ask | Failures chose:
     Pass: 145 (99%)
     Tell: 2 (1%)
   ```

2. **Feature Correlation**: What scenario features predict success vs failure?
   ```
   Feature comparison (% of correct vs incorrect trials with each feature):
     Self never saw a put: 65% vs 13% (predicts success) ***
     Teammate put/moved: 65% vs 44% (predicts success) ***
   ```

3. **Per-Scenario Breakdown**: Success rates with epistemic state context
   ```
   Scenario 12: 38.6% | Self=Believes X, Tm=Knows Truth, Opp=Unknown
   Scenario 13: 15.8% | Self=Believes X, Tm=Knows Truth, Opp=Knows Truth
   ```

4. **Concrete Examples**: Full scenario text for failed trials

The script generates three output files:
- `error_analysis.txt` - All models combined
- `error_analysis_thinking.txt` - Thinking models only (paired)
- `error_analysis_nonthinking.txt` - Non-thinking models only (paired)

"Paired" means only models that have both thinking and non-thinking versions are included in the comparison (e.g., `anthropic-claude-opus-4.5` and `anthropic-claude-opus-4.5_think`).

### Thinking vs Non-Thinking Comparison

Models with extended thinking capabilities (`_think` suffix) are compared against their non-thinking counterparts to measure the impact of deliberative reasoning on ToM tasks.

**Paired models analyzed:**
- anthropic-claude-opus-4.5 / anthropic-claude-opus-4.5_think
- anthropic-claude-sonnet-4.5 / anthropic-claude-sonnet-4.5_think
- openai-gpt-5 / openai-gpt-5_think
- openai-gpt-5.2 / openai-gpt-5.2_think

**Key findings:** Thinking models consistently outperform their non-thinking counterparts, with the largest gains in:
- True vs False Belief (~94% vs ~61%)
- Combined Uncertainty (~82% vs ~58%)
- Teammate vs Opponent (~83% vs ~55%)

### Pre/Post Bug Fix Comparison (`compare_old_new.py`)

Compares results before and after fixing the Extra=1 scenario generation bug (see `CHANGELOG.md`). The bug caused Extra=1 scenarios to sometimes have incorrect optimal actions due to event sequencing issues.

## Key Insights

### Common Failure Patterns

1. **Ask→Pass confusion**: Models often Pass when they should Ask their teammate, even when they only have a belief (not knowledge) and their teammate knows the truth. The model answers correctly but fails to recognize it doesn't *know* it's correct.

2. **Unnecessary Telling**: Models Tell their teammate information the teammate already knows, especially in scenarios where Self knows and Teammate also knows.

3. **Opponent knowledge interference**: When the opponent knows the truth (scenario 13, 19), models perform worse than when the opponent is uncertain—possibly confusing opponent knowledge with teammate knowledge.

### Scenario Difficulty

Hardest scenarios (lowest success rates):
- **Scenario 13** (15.8%): Self Believes X, Teammate Knows Truth, Opponent Knows Truth → should Ask
- **Scenario 12** (38.6%): Self Believes X, Teammate Knows Truth, Opponent Unknown → should Ask
- **Scenario 15** (43.6%): Self Knows X, Teammate Believes Truth → should Pass (not Tell)

### Feature Predictors

- **"Self never saw a put" predicts success** on Ask scenarios: When the model left before any put action, it's more likely to correctly recognize uncertainty and Ask.
- **"Teammate put/moved" predicts success** on Ask scenarios: Salient teammate actions make the model more likely to consider teammate knowledge.
- **"Opponent Knows Truth" predicts failure**: Models struggle when opponents also have knowledge, possibly conflating opponent and teammate epistemic states.
