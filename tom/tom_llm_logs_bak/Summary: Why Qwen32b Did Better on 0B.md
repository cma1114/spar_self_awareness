Summary: Why Qwen32b Did Better on 0B

  The analysis reveals several factors:

  1. Scenario Length Effect

  - 0A scenarios are short (avg 202 chars, 3.4 events)
  - 0B scenarios are longer (avg 341 chars, 6.4 events)
  - Both have the same epistemic complexity (ECT=2.0)

  2. B's Activity Makes a Difference

  - 0B: B performs actions in 50% of scenarios
  - 0A: B performs actions in only 33% of scenarios
  - When B does things (puts/moves), the model better recognizes B has knowledge

  3. Short Scenarios Trigger "Play it Safe" Behavior

  - In brief 0A scenarios like "D puts X in box. B leaves.", the model often unnecessarily Tells B
  - Even though B clearly witnessed the event, the model doesn't trust this
  - In longer 0B scenarios with the same epistemic situation, the model correctly Passes

  4. The Core Issue: Minimal Context = Overcautious Behavior

  The model seems to have a heuristic: "When scenarios are short/simple, better be safe and share information." This backfires because in short scenarios:
  - B often already knows the answer from being present
  - Telling costs 0.5 points unnecessarily
  - The optimal action is usually Pass (87% of the time)

  In 0B's longer scenarios, the additional "filler" events seem to help the model trust that other players have been tracking the situation, leading to more appropriate Passes.
