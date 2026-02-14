# ToM Test Observations

## Overview

Analysis of Theory of Mind test results across 19 models with ~17,000 total trials.

## Key Findings

### 1. Reasoning Traces Don't Correlate with Task Difficulty

**Finding:** GPT-5.2_think uses reasoning on only ~48% of turns, but this is **not** strategically allocated to harder problems.

| Difficulty Tier | All-Model Accuracy | GPT-5.2 Reasoning % |
|-----------------|-------------------|---------------------|
| Hard (bottom 1/3) | 47.9% | 42.7% |
| Medium | 70.1% | 42.7% |
| Easy (top 1/3) | 88.6% | 57.7% |

Pearson correlation between scenario difficulty and reasoning usage: **r = 0.259** (weak positive)

**Interpretation:** GPT-5.2 actually reasons *more* on easier scenarios. The model's decision to engage extended thinking appears unrelated to objective task difficulty—possibly triggered by surface features of the prompt rather than genuine complexity assessment.

### 2. Reasoning Improves Accuracy Substantially

Pooled across all thinking models:
- **With reasoning:** 88.3% accuracy (6,664 turns)
- **Without reasoning:** 68.0% accuracy (434 turns)
- **Difference:** +20.3%

This is a within-model comparison (same models, different turns), suggesting reasoning genuinely helps rather than just correlating with model capability.

### 3. Most Thinking Models Use Reasoning 100% of the Time

| Model | Turns | Reasoning % | Accuracy |
|-------|-------|-------------|----------|
| anthropic-claude-opus-4.5_think | 468 | 100.0% | 92.9% |
| anthropic-claude-sonnet-4.5_think | 1170 | 100.0% | 92.0% |
| deepseek-chat-v3.1_think | 780 | 100.0% | 91.4% |
| qwen3-next-80b-a3b-thinking_think | 780 | 100.0% | 91.2% |
| google-gemini-3-pro-preview_think | 1170 | 100.0% | 88.9% |
| openai-gpt-5_think | 390 | 93.3% | 87.6% |
| qwen3-235b-a22b-thinking-2507_think | 780 | 100.0% | 86.0% |
| qwen3-32b_think | 780 | 100.0% | 83.7% |
| **openai-gpt-5.2_think** | **780** | **47.7%** | **68.7%** |

GPT-5.2 is a notable outlier—it has selective reasoning but uses it seemingly arbitrarily.

### 4. Hardest Scenarios (All Models)

Scenarios with lowest accuracy across all 19 models:

| Scenario | Extra | Accuracy | Description |
|----------|-------|----------|-------------|
| 12 | 1 | 25.4% | Self believes X, Teammate knows truth → Ask Teammate |
| 12 | 0 | 27.9% | Self believes X, Teammate knows truth → Ask Teammate |
| 13 | 1 | 31.0% | Self believes X, Teammate knows truth, Opponent knows truth → Ask Teammate |
| 13 | 0 | 36.9% | Self believes X, Teammate knows truth, Opponent knows truth → Ask Teammate |

**Pattern:** Scenarios requiring the model to recognize it doesn't know the answer and should Ask Teammate are hardest. Models struggle with epistemic humility—recognizing when they hold uncertain beliefs vs. certain knowledge.

### 5. Extra=1 (Complex Sequences) Impact Analysis

Extra=1 scenarios add complexity via additional events (characters leaving/re-entering, items moved multiple times).

**Scenarios most hurt by Extra=1:**
| Scenario | Extra=0 | Extra=1 | Diff | Structure |
|----------|---------|---------|------|-----------|
| 22 | 81.1% | 60.7% | -20.4% | Teammate answers, Self/Teammate/Opponent all know → Pass |
| 26 | 85.1% | 66.0% | -19.1% | Teammate answers, Self believes, Teammate/Opponent know → Pass |
| 16 | 58.4% | 39.8% | -18.6% | Teammate answers, Self knows, Teammate believes, Opponent knows → Pass |
| 20 | 86.1% | 69.1% | -17.0% | Teammate answers, Self/Teammate know, Opponent believes → Pass |

**Scenarios where Extra=1 helps:**
| Scenario | Extra=0 | Extra=1 | Diff |
|----------|---------|---------|------|
| 29 | 51.2% | 62.3% | +11.1% |
| 27 | 54.4% | 63.4% | +8.9% |

**Error Pattern Analysis:**

For scenarios 16, 20, 22, 26 (all Teammate-answers, optimal=Pass):
- Models incorrectly choose `Tell(B, container, item)` ~80-90% of the time when wrong
- The error is **overconfidence**: models believe they should inform B even when B already knows or believes truth
- Extra=1 scenarios have more complex event sequences that may make models less certain about B's state, triggering unnecessary Tell actions

**Key insight:** The common failure mode is models not trusting that the teammate already has correct knowledge. When events get more complex (Extra=1), models become more paranoid and want to "make sure" by telling, even when it's unnecessary and costly.

### 6. "Time Goes By" Pause Cue Experiment

**Hypothesis:** Adding "Time goes by." after "You leave the room." might help models recognize that events could have occurred while they were away, improving performance on scenarios where they should Ask.

**Method:** Added configurable `PAUSE_MODE` parameter that inserts "Time goes by." after the player leaves. Tested on two models with scenarios 10-13 (all have Self=Believes X, vary by teammate knowledge).

**Results:**

| Model | Scenario | Old | New | Change | Optimal | Teammate State |
|-------|----------|-----|-----|--------|---------|----------------|
| openai-gpt-5-chat | 10 | 85% | 65% | **-20%** | Pass | Unknown |
| openai-gpt-5-chat | 11 | 90% | 55% | **-35%** | Pass | Unknown |
| openai-gpt-5-chat | 12 | 20% | 40% | **+20%** | Ask | Knows Truth |
| openai-gpt-5-chat | 13 | 30% | 30% | 0% | Ask | Knows Truth |
| **openai-gpt-5-chat** | **Total** | **56%** | **48%** | **-9%** | | |
| openai-gpt-5.2_think | 10 | 83% | 39% | **-44%** | Pass | Unknown |
| openai-gpt-5.2_think | 11 | 94% | 33% | **-61%** | Pass | Unknown |
| openai-gpt-5.2_think | 12 | 33% | 72% | **+39%** | Ask | Knows Truth |
| openai-gpt-5.2_think | 13 | 6% | 61% | **+56%** | Ask | Knows Truth |
| **openai-gpt-5.2_think** | **Total** | **54%** | **51%** | **-3%** | | |

**Key Findings:**

1. **The pause cue shifts the model's prior toward "Ask"**: Models become more likely to Ask regardless of whether it's correct.

2. **Dramatic improvements on Ask scenarios (12-13)**: +20-56% gains when the optimal action is Ask(B). The cue successfully signals "things may have changed."

3. **Equally dramatic regressions on Pass scenarios (10-11)**: -20-61% losses when the optimal action is Pass. The cue makes models Ask even when their teammate doesn't know anything useful.

4. **Thinking models show stronger effects in both directions**: GPT-5.2_think had larger swings than GPT-5-chat, suggesting extended reasoning amplifies the interpretation of the cue.

5. **Net effect is negative**: The gains on Ask scenarios don't compensate for losses on Pass scenarios (-9% and -3% overall).

**Interpretation:** The "Time goes by." cue is a blunt instrument. It successfully conveys uncertainty but doesn't help the model reason about *who* has information. The model interprets the cue as "something happened, I should ask someone" rather than "something happened, let me think about who saw what."

**Implication:** Surface-level prompt modifications can dramatically shift model behavior but may not improve underlying epistemic reasoning. The model needs to reason about *what each agent observed*, not just *whether time passed*.

## Methodology Notes

- **Valid trials:** Records where player A's count is a multiple of 78 (complete scenario sets)
- **Reasoning detection:** Parsed from log files via "REASONING TRACE:" marker after "It is your turn."
- **Difficulty measure:** Average accuracy across all 19 models per scenario

## Scripts

- `analyze_results.py` - Cross-model performance comparison with Wilson CIs
- `analyze_reasoning.py` - Reasoning trace analysis for thinking models
