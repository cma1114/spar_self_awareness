# ToM Test - Claude Context

## Critical Architecture Rules

### One Spec = One Mini-Game (DO NOT CHANGE)

Each row in `ToM - scenarios.csv` must generate its own separate mini-game. The LLM test loop:
```python
for i, spec in enumerate(self.specs):
    generate_scenarios_from_tuples([spec], outfile=outfile, seed=i, chartypes=chartypes)
    game_state = play_game_cli(scenario_file=outfile, llm_player=self)
```

**WHY**: All scenario specs are designed for player A's perspective. If you consolidate specs into one game file, the turn order cycles A->B->C->D, so A only gets ~20 turns instead of 78.

**NEVER** consolidate all specs into a single game - this breaks the test by reducing data points from 78 to ~20.

### File Management

Use a single temp file that gets overwritten each iteration, not 78 separate files. Clean up when done.

### Extra Field

- `Extra=0`: Base scenario
- `Extra=1`: More complex event sequences
- CSV has 39 base scenarios x 2 (with/without Extra) = 78 total
- Report performance broken down by Extra value
