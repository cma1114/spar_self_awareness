"""
Room Scenario Game - A strategic deduction game about information and belief.
"""

import json
import re
import sys
import os
from typing import Dict, List, Set, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass, asdict
from enum import Enum
import random
from generate_tom_scenarios_new import generate_scenarios_from_tuples, ITEMS_GEN
from tom_helpers import (
    Scenario, Event, EpistemicType, AskConstraintType, CharacterType, Character, Team,
    save_scenarios, load_scenarios, SpecTuple, read_specs_from_csv
)
try:
    from base_game_class import BaseGameClass
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    BaseGameClass = object  # Dummy base class
import argparse

# Game instructions template - defined once, used in play_game_cli and prompt_history
GAME_SETUP_TEMPLATE = """
Let's play a game! In this game you will be playing with a teammate against a team of opposing players. For simplicity, the players are referred to as A, B, C, and D. A and B are teammates on the Blue team, and C and D are teammates on the Red team. You are player A. The game is made up of turns, and each turn consists of a scenario, an action phase, and an answer phase. Players need to figure out the right action to take in each action phase, so that their team scores more points than their opponent's team.

SCENARIO
In each scenario, one or more of the players start inside a room. The room has one or more containers. Each scenario may include:
- one or more players entering or leaving the room
- one or more players putting objects into containers
- one or more players removing objects from containers
Players cannot see inside containers, but when a player is inside the room, they are aware of anything that happens in the room. When a player is outside the room they are unaware of anything that happens in the room, but do know if anyone enters or leaves the room. If an object is put or moved into a container while a player is outside the room, it will not be described to that player, but players leaving or entering the room will be described to that player.
At the end of the scenario, I will reveal a question that I will ask in the answer phase.

ACTION PHASE
In the action phase, the player whose turn it is can take one action: Ask, Tell, or Pass.
An Ask action entails asking any player the contents of one container. The format for an Ask action is: Ask(Player, Container); this will result in the player you ask answering your question. It costs your team 0.5 points to do this action.
A Tell action entails telling any player the contents of one container. The format for a Tell action is: Tell(Player, Container, Contents); this will result in the player you tell hearing the answer you give. It costs your team 0.5 points to do this action.
The format for a Pass action is simply "Pass". Nothing changes as a result of this action, and it costs 0 points.
Ask and Tell actions are private between the two players involved; other players do not hear or see these actions or their results.
Players do not have to tell the truth either in their Tell actions or in response to another player's Ask action. 
A team's points can go negative.

ANSWER PHASE
In the answer phase, I will ask a target player (not necessarily the player whose turn it is) to name the contents of one container. When a player answers my question correctly, their team gets 1 point. The first team to {WINNING_SCORE} points wins.

For the purpose of this study, we're going to repeatedly ask you to make the first move in different runs of this game. Each run is separate and in each run we want you to make the best move in that situation.
"""

class ActionType(Enum):
    ASK = "ask"
    TELL = "tell"
    PASS = "pass"

@dataclass
class Action:
    """Represents a player action."""
    action_type: ActionType
    target_char: Optional[str] = None
    container: Optional[str] = None
    contents: Optional[str] = None
    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return (self.action_type, self.target_char, self.container, self.contents) == \
               (other.action_type, other.target_char, other.container, other.contents)
@dataclass
class TurnRecord:
    """Records what happened in a turn."""
    round_num: int
    character: str
    scenario_desc: str
    question: str
    action: str
    action_cost: float
    answer_given: str
    answer_correct: bool
    answer_score: float
    optimal_action: str
    was_optimal: bool
    blue_score_after: float
    red_score_after: float
    scenario_id: Optional[str] = None
    extra: Optional[int] = None
    epistemic_type: Optional[str] = None
    ask_constraint: Optional[str] = None
    ks_self: Optional[str] = None
    ks_teammate: Optional[str] = None
    ks_opponent: Optional[str] = None
    # New fields:
    answerer: Optional[str] = None
    asked_player: Optional[str] = None
    asked_player_present: Optional[str] = None
    ask_container_matches: Optional[str] = None
    told_player: Optional[str] = None
    told_player_present: Optional[str] = None
    tell_truthful_about_question: Optional[str] = None
    b_left_before_a: Optional[str] = None
    a_left_before_put: Optional[str] = None
    b_put_or_moved: Optional[str] = None
    # History mode fields:
    history_mode: Optional[str] = None
    trial: Optional[int] = None
    rep: Optional[int] = None  # Track which rep this trial belongs to


class GameState:
    """Manages the game state."""
    
    def __init__(self, scenario_file: str):
        self.current_scenario_idx = 0
        self.current_turn_idx = 0
        self.scores = {Team.BLUE: 0.0, Team.RED: 0.0}
        self.characters: Dict[str, Character] = {}
        self.turn_records: List[TurnRecord] = []
        self.game_over = False
        self.winner = None

        try:
            self.scenarios, chars, chartypes = load_scenarios(scenario_file)
        except FileNotFoundError:
            print("Scenario file not found.")

        for name, ctype in zip(chars, chartypes):
            team = Team.BLUE if name in ['A', 'B'] else Team.RED if name in ['C', 'D'] else Team.NEUTRAL
            self.characters[name] = Character(name, team, ctype)

        self.turn_order = chars
        self.WINNING_SCORE = int(len(self.scenarios)/len(self.turn_order)) + 4


    def get_teammate(self, actor_name: str) -> Optional[str]:
        """Get the teammate of the actor."""
        actor = self.characters[actor_name]
        for name, char in self.characters.items():
            if name != actor_name and char.team == actor.team:
                return name
        return None
    
    def is_honest_to(self, speaker_name: str, listener_name: str) -> bool:
        """Check if speaker is honest to listener."""
        speaker = self.characters[speaker_name]
        listener = self.characters[listener_name]
        
        if speaker.char_type in [CharacterType.NEUTRAL, CharacterType.HONEST_OPPONENT, CharacterType.HONEST_TEAMMATE]:
            return True
        return speaker.team == listener.team
    
    def trusts(self, listener_name: str, speaker_name: str) -> bool:
        listener = self.characters[listener_name]
        speaker = self.characters[speaker_name]

        if listener.char_type in [CharacterType.NEUTRAL, CharacterType.HONEST_OPPONENT, CharacterType.HONEST_TEAMMATE]:
            return True
        return speaker.team == listener.team
        
    def reset_scenario_knowledge(self):
        """Reset all character knowledge for new scenario."""
        for char in self.characters.values():
            char.knowledge = {}
            char.beliefs = {}
    
    def get_present_at_end(self, scenario: Scenario) -> Set[str]:
        """Get who's present at the end of the scenario."""
        present = set(scenario.present_initially)
        for event in scenario.events:
            if event.event_type == 'leave':
                present.discard(event.character)
            elif event.event_type == 'enter':
                present.add(event.character)
        return present

    def process_scenario_events(self, scenario: Scenario):
        """Process scenario events to determine what each character knows."""
        self.reset_scenario_knowledge()
        
        present = set(scenario.present_initially)
        container_contents = {'bag': None, 'box': None}
        
        for event in scenario.events:
            if event.event_type == 'put':
                container_contents[event.container] = event.item
                for char_name in present:
                    self.characters[char_name].update_knowledge(event.container, event.item)
                    
            elif event.event_type == 'move':
                container_contents[event.to_container] = event.item
                container_contents[event.from_container] = None
                for char_name in present:
                    self.characters[char_name].update_knowledge(event.to_container, event.item)
                    self.characters[char_name].update_knowledge(event.from_container, None)

            elif event.event_type == 'remove':
                container_contents[event.container] = None
                for char_name in present:
                    self.characters[char_name].update_knowledge(event.container, None)
                        
            elif event.event_type == 'leave':
                present.discard(event.character)

            elif event.event_type == 'enter':
                present.add(event.character)
        
        return container_contents
    
    def execute_npc_action(self, actor_name: str, scenario: Scenario, true_contents: Dict[str, str]) -> Action:
        """Execute NPC action based on game rules - completely determined."""
        container = scenario.question_container
        truth = true_contents[container]
        target = scenario.who_answers
        
        if scenario.correct_action == "Tell Teammate":
            return Action(ActionType.TELL, target, container, truth)
        elif scenario.correct_action == "Ask Teammate":
            teammate = self.get_teammate(actor_name)
            return Action(ActionType.ASK, teammate, container)
        else:
            return Action(ActionType.PASS)
    
    def is_action_optimal(self, action_str: str, scenario: Scenario, true_contents: Dict[str, str]) -> bool:
        """Check if live player's action is optimal."""
        action = self.parse_action(action_str)
        if not action:
            return False
        optimal_action = self.execute_npc_action('A', scenario, true_contents)
        if action != optimal_action:
            return False        
        return True
    
    def get_current_scenario(self) -> Optional[Scenario]:
        """Get the current scenario."""
        if self.current_scenario_idx >= len(self.scenarios):
            return None
        return self.scenarios[self.current_scenario_idx]
    
    def get_current_turn_character(self) -> Optional[str]:
        """Get whose turn it is."""
        if self.current_turn_idx >= len(self.turn_order):
            return None
        return self.turn_order[self.current_turn_idx]
    
    def parse_action(self, action_str: str) -> Optional[Action]:
        """Parse action string into Action object."""
        action_str = action_str.strip()
        
        if action_str.lower() == 'pass':
            return Action(ActionType.PASS)
        
        ask_match = re.match(r'Ask\(([A-DN]),\s*(bag|box)\)', action_str, re.IGNORECASE)
        if ask_match:
            return Action(ActionType.ASK, ask_match.group(1).upper(), ask_match.group(2).lower())
        
        tell_match = re.match(r'Tell\(([A-DN]),\s*(bag|box),\s*(?:an? |the )?(\w+)\)', action_str, re.IGNORECASE)
        if tell_match:
            return Action(ActionType.TELL, tell_match.group(1).upper(), 
                         tell_match.group(2).lower(), tell_match.group(3).lower())
        
        return None
    
    def execute_action(self, actor_name: str, action: Action, true_contents: Dict[str, str]) -> Tuple[float, str]:
        """Execute an action and return the score change and description."""
        score_delta = 0.0
        desc = ""
        
        if action.action_type == ActionType.PASS:
            desc = "Pass"
            
        elif action.action_type == ActionType.ASK:
            score_delta -= 0.5
            target = self.characters[action.target_char]
            container = action.container

            # What the target says:
            # - Honest: report what they believe (what they observed).
            # - Dishonest: intentionally say something different from their belief.
            if target.char_type in [CharacterType.DISHONEST_TEAMMATE, CharacterType.DISHONEST_OPPONENT] and target.team != self.characters[actor_name].team:
                believed = target.get_belief(container)
                options = ['apple', 'ball', 'banana', 'brick', 'stapler', 'orange']
                # Pick an item different from what they believe; if they believe nothing, pick any item.
                wrong = next((x for x in options if x != (believed or 'nothing')), options[0])
                answer = wrong
            else:
                answer = target.get_belief(container)

            # Asking always updates the asker's belief with whatever was answered.
            self.characters[actor_name].receive_info(container, answer, target, True)

            desc = f"Ask({action.target_char}, {container})"
            
        elif action.action_type == ActionType.TELL:
            score_delta -= 0.5
            target_name = action.target_char
            
            if self.trusts(target_name, actor_name):
                self.characters[target_name].receive_info(action.container, action.contents,
                                                         self.characters[actor_name], True)
            
            desc = f"Tell({action.target_char}, {action.container}, {action.contents})"
        
        return score_delta, desc
    
    def resolve_answer_phase(self, scenario: Scenario, true_contents: Dict[str, str]) -> Tuple[str, bool, float]:
        """Resolve the answer phase and return answer, correctness, and score change."""
        answerer = self.characters[scenario.who_answers]
        container = scenario.question_container
        
        belief = answerer.get_belief(container)
        truth = true_contents[container]
        
        is_correct = (belief == truth)
        
        if is_correct:
            return belief if belief else 'nothing', True, 1.0
        else:
            return belief if belief else 'nothing', False, 0.0
    
    def check_game_over(self):
        """Check if game is over."""
        if self.scores[Team.BLUE] >= self.WINNING_SCORE:
            self.game_over = True
            self.winner = Team.BLUE
        elif self.scores[Team.RED] >= self.WINNING_SCORE:
            self.game_over = True
            self.winner = Team.RED
        elif self.current_scenario_idx >= len(self.scenarios):
            self.game_over = True
            if self.scores[Team.BLUE] > self.scores[Team.RED]:
                self.winner = Team.BLUE
            elif self.scores[Team.RED] > self.scores[Team.BLUE]:
                self.winner = Team.RED
            else:
                self.winner = None
    
    def advance_turn(self):
        """Move to next turn."""
        self.current_scenario_idx += 1
        self.current_turn_idx += 1
        if self.current_turn_idx >= len(self.turn_order):
            self.current_turn_idx = 0


def save_game_results(turn_records: List[TurnRecord], filename: str):
    """Save game results to JSON file."""
    with open(filename, 'w') as f:
        json.dump([asdict(r) for r in turn_records], f, indent=2)

if TORCH_AVAILABLE:
    class ToMTestLLM(BaseGameClass):
        def __init__(self, subject_id, subject_name, specs: List[SpecTuple], log_dir="tom_llm_logs", history_mode="none", reps=1):
            super().__init__(subject_id, subject_name, is_human_player=False, log_dir=log_dir)
            self.specs = specs  # Base specs (not multiplied by reps)
            self.reps = reps    # Number of reps
            self.all_turn_records = []
            self.history_mode = history_mode
            # For history mode: track completed trials
            self.completed_trials = []  # List of dicts: {trial, scenario_desc, question_desc, action, reasoning}
            self.current_trial = 0
            self.current_rep = 1  # Track current rep number
            # Store the game setup text once it's formatted (set by play_game_cli on first trial)
            self.game_setup_text = None
            # Set up prompt history file path
            if log_dir and self.log_base_name:
                self.prompt_history_filename = f"{self.log_base_name}_prompt_history.txt"
            else:
                self.prompt_history_filename = None

        def run_test(self):
            self._log("--- Starting LLM ToM Test ---")
            self._log(f"History mode: {self.history_mode}")
            self._log(f"Reps: {self.reps}")
            
            chartypes = [CharacterType.LIVE_PLAYER, CharacterType.HONEST_OPPONENT, CharacterType.DISHONEST_TEAMMATE, CharacterType.DISHONEST_OPPONENT]

            # Reuse a single temp file (HEAD behavior), but preserve randomized trial order in history mode.
            # If you ever want per-trial scenario files for debugging, set KEEP_SCENARIO_FILES=True.
            KEEP_SCENARIO_FILES = False

            outfile_tmp = f"{self.log_base_name}_scenario_tmp.json"

            # Handle reps differently based on history_mode
            if self.history_mode != "none" and self.reps > 1:
                # Run separate sessions for each rep, resetting history between reps
                self._run_with_separate_history_sessions(chartypes, outfile_tmp, KEEP_SCENARIO_FILES)
            else:
                # Original behavior: run all specs (possibly multiplied by reps) in one session
                self._run_single_session(chartypes, outfile_tmp, KEEP_SCENARIO_FILES)

            # Clean up temp file
            if os.path.exists(outfile_tmp) and not KEEP_SCENARIO_FILES:
                os.remove(outfile_tmp)

            self._log("\n" + "=" * 70)
            self._log("LLM ToM Test Finished")

            total_optimal = sum(1 for r in self.all_turn_records if r.character == 'A' and r.was_optimal)
            total_turns = sum(1 for r in self.all_turn_records if r.character == 'A')

            if total_turns > 0:
                self._log(f"LLM was optimal in {total_optimal}/{total_turns} turns ({(total_optimal/total_turns)*100:.2f}%).")

                # Breakdown by Extra
                extra0_records = [r for r in self.all_turn_records if r.character == 'A' and (r.extra is None or r.extra == 0)]
                extra1_records = [r for r in self.all_turn_records if r.character == 'A' and r.extra == 1]

                if extra0_records:
                    extra0_optimal = sum(1 for r in extra0_records if r.was_optimal)
                    self._log(f"  Extra=0: {extra0_optimal}/{len(extra0_records)} optimal ({(extra0_optimal/len(extra0_records))*100:.2f}%)")
                if extra1_records:
                    extra1_optimal = sum(1 for r in extra1_records if r.was_optimal)
                    self._log(f"  Extra=1: {extra1_optimal}/{len(extra1_records)} optimal ({(extra1_optimal/len(extra1_records))*100:.2f}%)")
                
                # Breakdown by rep if multiple reps with history
                if self.history_mode != "none" and self.reps > 1:
                    self._log("\n  Breakdown by rep:")
                    for rep_num in range(1, self.reps + 1):
                        rep_records = [r for r in self.all_turn_records if r.character == 'A' and r.rep == rep_num]
                        if rep_records:
                            rep_optimal = sum(1 for r in rep_records if r.was_optimal)
                            self._log(f"    Rep {rep_num}: {rep_optimal}/{len(rep_records)} optimal ({(rep_optimal/len(rep_records))*100:.2f}%)")
            else:
                self._log("No turns were played by the LLM.")

            save_game_results(self.all_turn_records, self.game_data_filename)
            self._log(f"\nGame results saved to {self.game_data_filename}")
            
            # Write prompt history file if in history mode
            if self.history_mode != "none" and self.prompt_history_filename and not (self.reps > 1):
                self._write_prompt_history()
                self._log(f"Prompt history saved to {self.prompt_history_filename}")

        def _run_single_session(self, chartypes, outfile_tmp, KEEP_SCENARIO_FILES):
            """Original behavior: run all specs in one session."""
            # If history_mode is "none" and reps > 1, multiply specs here
            specs_to_run = self.specs * self.reps if self.history_mode == "none" else self.specs
            
            # Determine trial order
            indexed_specs = list(enumerate(specs_to_run))
            if self.history_mode != "none":
                random.shuffle(indexed_specs)
                self._log(f"Randomized trial order for history_mode='{self.history_mode}'")

            for trial_num, (spec_idx, spec) in enumerate(indexed_specs, start=1):
                self.current_trial = trial_num if self.history_mode != "none" else 0

                self._log(f"\n--- Running Trial {trial_num}/{len(specs_to_run)} (Spec ID {spec_idx+1}): {spec} ---")

                # Seed choice: use spec_idx so runs are reproducible even if order is shuffled.
                # Note: if you use --reps, spec_idx differs across repeats (so repeats generate different scenarios).
                generate_scenarios_from_tuples([spec], outfile=outfile_tmp, seed=spec_idx, chartypes=chartypes)

                if KEEP_SCENARIO_FILES:
                    keep_name = f"scenarios_llm_test_{spec_idx}.json"
                    with open(outfile_tmp, "r", encoding="utf-8") as src, open(keep_name, "w", encoding="utf-8") as dst:
                        dst.write(src.read())

                game_state = play_game_cli(scenario_file=outfile_tmp, llm_player=self)
                if game_state:
                    self.all_turn_records.extend(game_state.turn_records)

        def _run_with_separate_history_sessions(self, chartypes, outfile_tmp, KEEP_SCENARIO_FILES):
            """Run multiple reps with separate history sessions for each."""
            total_trials = len(self.specs) * self.reps
            global_trial_counter = 0
            
            for rep_num in range(1, self.reps + 1):
                self.current_rep = rep_num

                # Reset history for this rep
                self.completed_trials = []
                self.current_trial = 0
                
                self._log(f"\n{'='*70}")
                self._log(f"=== Starting Rep {rep_num} of {self.reps} ===")
                self._log(f"{'='*70}")
                
                # Shuffle specs independently for this rep
                indexed_specs = list(enumerate(self.specs))
                random.shuffle(indexed_specs)
                self._log(f"Randomized trial order for rep {rep_num}")

                for trial_num_in_rep, (spec_idx, spec) in enumerate(indexed_specs, start=1):
                    global_trial_counter += 1
                    self.current_trial = trial_num_in_rep  # Trial number within this rep (1-78)

                    self._log(f"\n--- Rep {rep_num}, Trial {trial_num_in_rep}/{len(self.specs)} (Global {global_trial_counter}/{total_trials}, Spec ID {spec_idx+1}): {spec} ---")

                    # Seed: combine rep_num and spec_idx to ensure different scenarios across reps
                    seed = rep_num * 1000 + spec_idx
                    generate_scenarios_from_tuples([spec], outfile=outfile_tmp, seed=seed, chartypes=chartypes)

                    if KEEP_SCENARIO_FILES:
                        keep_name = f"scenarios_llm_test_rep{rep_num}_{spec_idx}.json"
                        with open(outfile_tmp, "r", encoding="utf-8") as src, open(keep_name, "w", encoding="utf-8") as dst:
                            dst.write(src.read())

                    game_state = play_game_cli(scenario_file=outfile_tmp, llm_player=self)
                    if game_state:
                        self.all_turn_records.extend(game_state.turn_records)
                
                # Log rep summary
                rep_records = [r for r in self.all_turn_records if r.character == 'A' and r.rep == rep_num]
                if rep_records:
                    rep_optimal = sum(1 for r in rep_records if r.was_optimal)
                    self._log(f"\nRep {rep_num} complete: {rep_optimal}/{len(rep_records)} optimal ({(rep_optimal/len(rep_records))*100:.2f}%)")

                if self.prompt_history_filename:
                    self._write_prompt_history(rep_num)

        def _write_prompt_history(self, rep_num=None):
            """Write the final cumulative prompt history to file."""
            # Determine file mode: write for first rep or single-session, append for subsequent reps
            if rep_num is None or rep_num == 1:
                mode = 'w'
            else:
                mode = 'a'
            
            with open(self.prompt_history_filename, mode, encoding='utf-8') as f:
                # Write header only on first write (rep 1 or single-session)
                if rep_num is None or rep_num == 1:
                    f.write("=" * 70 + "\n")
                    f.write("GAME INSTRUCTIONS\n")
                    f.write("=" * 70 + "\n")
                    if self.game_setup_text:
                        f.write(self.game_setup_text.strip() + "\n")
                    else:
                        # Fallback if game_setup_text wasn't stored (shouldn't happen)
                        f.write(GAME_SETUP_TEMPLATE.format(WINNING_SCORE="N/A").strip() + "\n")
                    f.write("=" * 70 + "\n\n")
                
                # Write rep separator if multi-rep mode
                if rep_num is not None:
                    f.write(f"{'='*70}\n")
                    f.write(f"REP {rep_num}\n")
                    f.write(f"{'='*70}\n\n")
                
                # Write all completed trials
                for trial_data in self.completed_trials:
                    f.write(f"SCENARIO {trial_data['trial']}\n")
                    f.write("Here's what you see:\n")
                    f.write("-----------------------------------------------\n")
                    f.write(f"{trial_data['scenario_desc']}\n")
                    f.write("----------------------------------------------\n")
                    f.write(f"{trial_data['question_desc']}\n\n")
                    f.write("ACTION PHASE\n")
                    f.write("It is your turn.\n")
                    f.write(f"Your action: {trial_data['action']}\n")
                    if self.history_mode == "with_reasoning" and trial_data.get('reasoning'):
                        f.write(f"Your reasoning: {trial_data['reasoning']}\n")
                    f.write("\n")

        def build_history_prompt(self, current_scenario_desc: str, current_question_desc: str, trial_num: int) -> str:
            """Build the cumulative prompt including all previous trials."""
            prompt_parts = []
            
            # Add all completed trials
            for trial_data in self.completed_trials:
                trial_prompt = f"""SCENARIO {trial_data['trial']}
Here's what you see:
-----------------------------------------------
{trial_data['scenario_desc']}
----------------------------------------------
{trial_data['question_desc']}

ACTION PHASE
It is your turn.
Your action: {trial_data['action']}"""
                if self.history_mode == "with_reasoning" and trial_data.get('reasoning'):
                    trial_prompt += f"\nYour reasoning: {trial_data['reasoning']}"
                prompt_parts.append(trial_prompt)
            
            # Add current trial prompt
            current_prompt = f"""SCENARIO {trial_num}
Here's what you see:
-----------------------------------------------
{current_scenario_desc}
----------------------------------------------
{current_question_desc}

ACTION PHASE
It is your turn.
Respond ONLY with your action, and no other text."""
            prompt_parts.append(current_prompt)
            
            return "\n\n".join(prompt_parts)


def play_game_cli(scenario_file: str, llm_player: Optional[BaseGameClass] = None, run_all_scenarios: bool = False):
    """Play the game in CLI mode, for humans or LLMs.

    Args:
        scenario_file: Path to the scenarios JSON file
        llm_player: Optional LLM player instance
        run_all_scenarios: If True, ignore winning score and run all scenarios (useful for testing)
    """
    game = GameState(scenario_file=scenario_file)
    log = llm_player._log if llm_player else print
    is_human = llm_player is None
    
    # Get history_mode and current trial from llm_player if available
    history_mode = getattr(llm_player, 'history_mode', 'none') if llm_player else 'none'
    current_trial = getattr(llm_player, 'current_trial', 0) if llm_player else 0
    current_rep = getattr(llm_player, 'current_rep', None) if llm_player else None

    if run_all_scenarios:
        game.WINNING_SCORE = float('inf')  # Disable early termination

    # Build game setup text from template
    GAME_SETUP = GAME_SETUP_TEMPLATE
    
    if 'N' in game.characters:
        GAME_SETUP += """The Neutral party does not get any turns, but will answer any question honestly."""
    
    game_setup_text = "=" * 70 + "\n" + GAME_SETUP.format(WINNING_SCORE=game.WINNING_SCORE) + "\n" + "=" * 70
    
    # Store the formatted game_setup_text on llm_player for use in prompt_history (only on first trial)
    if llm_player and hasattr(llm_player, 'game_setup_text') and llm_player.game_setup_text is None:
        llm_player.game_setup_text = GAME_SETUP.format(WINNING_SCORE=game.WINNING_SCORE)
    
    if is_human:
        log(game_setup_text)
    
    turn_count = 0

    while not game.game_over and game.get_current_scenario():
        scenario = game.get_current_scenario()
        turn_char = game.get_current_turn_character()
        
        if not turn_char:
            break
        
        # In LLM mode, we only process player A's turn
        if llm_player and turn_char != 'A':
            game.advance_turn()
            continue
        
        true_contents = game.process_scenario_events(scenario)
        scenario_desc = scenario.get_description_for(turn_char, game.characters)
        answerer = "you" if scenario.who_answers == turn_char else scenario.who_answers
        question_desc = f"I am going to ask {answerer} what is in the {scenario.question_container}."
        
        preamble=f"\n***********************************\nSCORE\nBlue={game.scores[Team.BLUE]}, Red={game.scores[Team.RED]}\n"
        log(preamble)

        # Build prompt text based on history_mode
        if history_mode != "none" and llm_player:
            # Use "SCENARIO [N]" format
            scenario_header = f"SCENARIO {current_trial}"
        else:
            scenario_header = "SCENARIO"

        prompt_text = f"""{scenario_header}
Here's what you see:
-----------------------------------------------
{scenario_desc}
----------------------------------------------
{question_desc}

ACTION PHASE"""        

        turn_name = "your" if turn_char == 'A' else f"{turn_char}'s"
        prompt_text+=f"""
It is {turn_name} turn.
Respond ONLY with your action, and no other text."""
        
        log(prompt_text)
        
        action = None
        action_str = ""
        reasoning_trace = None
        
        if turn_char == 'A':
            prompt_for_action = "Your action (Ask(Player, Container), Tell(Player, Container, Contents), or Pass): "
            if llm_player:
                if history_mode != "none":
                    # Build cumulative prompt with history
                    history_prompt = llm_player.build_history_prompt(scenario_desc, question_desc, current_trial)
                    llm_prompt_text = f"{game_setup_text}\n\n{history_prompt}\n{prompt_for_action}"
                else:
                    # Original behavior - single scenario
                    llm_prompt_text = f"{game_setup_text}\n{preamble}\n{prompt_text}\n{prompt_for_action}"

                # Logging for debugging
                # log(f"=== FULL LLM PROMPT ===\n{llm_prompt_text}\n=== END PROMPT ===")
                
                action_str, _, _, reasoning_trace = llm_player._get_llm_answer(
                    options=None,
                    q_text=llm_prompt_text,
                    message_history=[],
                    keep_appending=False,
                    setup_text="You are a player in a strategic deduction game. Your goal is to help your team win by taking the optimal action.",
                    MAX_TOKENS=30,
                    temp=0.0,
                    accept_any=True
                )
            else:
                action_str = input(prompt_for_action)

            action = game.parse_action(action_str)
            if not action:
                log(f"Invalid action: '{action_str}'. Recording as invalid and will be scored as incorrect.")
                action = Action(ActionType.PASS)  # Execute as Pass to continue game flow
                # Keep action_str as-is to record the actual invalid input
        else:
            action = game.execute_npc_action(turn_char, scenario, true_contents)
            if action.action_type == ActionType.ASK:
                action_str = f"Ask({action.target_char}, {action.container})"
            elif action.action_type == ActionType.TELL:
                action_str = f"Tell({action.target_char}, {action.container}, {action.contents})"
            else:
                action_str = "Pass"
        
        score_delta, action_desc = game.execute_action(turn_char, action, true_contents)
                
        log(f"\nAction: {action_str}")
        
        answer_given, is_correct, answer_score = game.resolve_answer_phase(scenario, true_contents)
        
        log(f"{scenario.who_answers} answers: {answer_given}")
        if is_correct:
            log(f"Correct! The {scenario.question_container} contains {answer_given}.")
        else:
            log(f"Incorrect. The {scenario.question_container} actually contains {true_contents[scenario.question_container]}.")
        
        blue_delta = 0.0
        red_delta = 0.0

        if turn_char in ['A', 'B']:
            blue_delta += score_delta
        else:
            red_delta += score_delta

        if is_correct:
            if scenario.who_answers in ['A', 'B']:
                blue_delta += answer_score
            else:
                red_delta += answer_score

        game.scores[Team.BLUE] += blue_delta
        game.scores[Team.RED] += red_delta

        def fmt_delta(x: float) -> str:
            sign = '+' if x >= 0 else '-'
            return f"{sign}{abs(x)}"
        log(f"\nOutcome: Blue {fmt_delta(blue_delta)}, Red {fmt_delta(red_delta)}")

        was_optimal = False
        expected_action_str = ""
        if turn_char == 'A':
            was_optimal = game.is_action_optimal(action_str, scenario, true_contents)
            expected_action_obj = game.execute_npc_action(turn_char, scenario, true_contents)
            if expected_action_obj.action_type == ActionType.PASS:
                expected_action_str = "Pass"
            elif expected_action_obj.action_type == ActionType.ASK:
                expected_action_str = f"Ask({expected_action_obj.target_char}, {expected_action_obj.container})"
            elif expected_action_obj.action_type == ActionType.TELL:
                expected_action_str = f"Tell({expected_action_obj.target_char}, {expected_action_obj.container}, {expected_action_obj.contents})"
        else:
            was_optimal = True
            expected_action_str = action_str

        # Calculate additional turn record fields
        answerer = scenario.who_answers
        
        # Ask-related fields
        asked_player = ""
        asked_player_present = ""
        ask_container_matches = ""
        if action.action_type == ActionType.ASK:
            asked_player = action.target_char
            present_at_end = game.get_present_at_end(scenario)
            asked_player_present = "TRUE" if action.target_char in present_at_end else "FALSE"
            ask_container_matches = "TRUE" if action.container == scenario.question_container else "FALSE"
        
        # Tell-related fields
        told_player = ""
        told_player_present = ""
        tell_truthful_about_question = ""
        if action.action_type == ActionType.TELL:
            told_player = action.target_char
            present_at_end = game.get_present_at_end(scenario)
            told_player_present = "TRUE" if action.target_char in present_at_end else "FALSE"
            
            # Check truthfulness only if telling about question container
            # Compare against what the player believes (their knowledge), not final true contents
            if action.container == scenario.question_container:
                player_belief = game.characters[turn_char].get_belief(action.container)
                # Handle empty container case: None belief matches "empty" or "nothing"
                if player_belief is None:
                    tell_truthful_about_question = "TRUE" if action.contents in ("empty", "nothing") else "FALSE"
                else:
                    tell_truthful_about_question = "TRUE" if action.contents == player_belief else "FALSE"
        
        # Event-based fields
        a_leave_idx = None
        b_leave_idx = None
        for idx, event in enumerate(scenario.events):
            if event.event_type == 'leave':
                if event.character == 'A':
                    a_leave_idx = idx
                elif event.character == 'B':
                    b_leave_idx = idx
        
        # B left before A
        b_left_before_a = ""
        if a_leave_idx is not None and b_leave_idx is not None:
            b_left_before_a = "TRUE" if b_leave_idx < a_leave_idx else "FALSE"
        
        # A left before put
        a_left_before_put = ""
        if a_leave_idx is not None:
            any_put_before_a_left = any(idx < a_leave_idx for idx, event in enumerate(scenario.events) 
                                        if event.event_type == 'put')
            a_left_before_put = "FALSE" if any_put_before_a_left else "TRUE"
        
        # B put or moved an item
        b_put_or_moved = "TRUE" if any((event.event_type == 'put' or event.event_type == 'move') and event.character == 'B' 
                                       for event in scenario.events) else "FALSE"
        
        turn_record = TurnRecord(
            round_num=scenario.round_num, scenario_id=scenario.id, extra=scenario.extra, character=turn_char, scenario_desc=scenario_desc,
            question=question_desc, action=action_str, action_cost=abs(score_delta),
            answer_given=answer_given, answer_correct=is_correct, answer_score=answer_score,
            optimal_action=expected_action_str, was_optimal=was_optimal,
            blue_score_after=game.scores[Team.BLUE], red_score_after=game.scores[Team.RED],
            epistemic_type=scenario.epistemic_type.value if scenario.epistemic_type else None,
            ask_constraint=scenario.ask_constraint.value if scenario.ask_constraint else None,
            ks_self=scenario.ks_self if scenario.ks_self else None,
            ks_teammate=scenario.ks_teammate if scenario.ks_teammate else None,
            ks_opponent=scenario.ks_opponent if scenario.ks_opponent else None,
            answerer=answerer,
            asked_player=asked_player,
            asked_player_present=asked_player_present,
            ask_container_matches=ask_container_matches,
            told_player=told_player,
            told_player_present=told_player_present,
            tell_truthful_about_question=tell_truthful_about_question,
            b_left_before_a=b_left_before_a,
            a_left_before_put=a_left_before_put,
            b_put_or_moved=b_put_or_moved,
            history_mode=history_mode if llm_player else None,
            trial=current_trial if history_mode != "none" else None,
            rep=current_rep if llm_player and history_mode != "none" else None,
        )
        game.turn_records.append(turn_record)
        
        # Store completed trial data for history mode
        if llm_player and history_mode != "none" and turn_char == 'A':
            llm_player.completed_trials.append({
                'trial': current_trial,
                'scenario_desc': scenario_desc,
                'question_desc': question_desc,
                'action': action_str,
                'reasoning': reasoning_trace
            })
        
        if is_human:
            input("\n[Press Enter to continue]")
        
        game.advance_turn()
        game.check_game_over()
        turn_count += 1

    log("\n" + "=" * 70)
    log("GAME OVER")
    log(f"Final Score: Blue {game.scores[Team.BLUE]} - Red {game.scores[Team.RED]}")
    if game.winner:
        log(f"Winner: {game.winner.value} team")
    elif game.winner is None:
        log("It's a tie!")
    log("=" * 70)
    
    if is_human:
        for record in game.turn_records:
            if record.character == 'A':
                log(f"\nRound {record.round_num} - {record.character}'s turn")
                log(f"KS_Self: {record.ks_self}, KS_Teammate: {record.ks_teammate}, KS_Opponent: {record.ks_opponent}")
                log(f"Action: {record.action}, Expected: {record.optimal_action}")
        
        save_game_results(game.turn_records, 'game_results.json')
        log("\nGame results saved to game_results.json")
    
    return game

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="human", choices=["human", "llm"])
    parser.add_argument("--model", type=str, default="kimi-k2")
    parser.add_argument("--reps", type=int, default=1, help="Number of times to repeat the full scenario set (default: 1)")
    parser.add_argument(
        "--history_mode",
        type=str,
        default="none",
        choices=["none", "no_reasoning", "with_reasoning"],
        help="History mode: 'none' for independent trials; 'no_reasoning' for cumulative context; 'with_reasoning' for cumulative context + reasoning traces"
    )
    args = parser.parse_args()

    specs = read_specs_from_csv('ToM - scenarios.csv')
    # Don't multiply specs by reps here - let ToMTestLLM handle it based on history_mode
    if args.mode == "llm":
        test_runner = ToMTestLLM(
            subject_id=args.model.replace("/", "-"),
            subject_name=args.model,
            specs=specs,
            log_dir="tom_llm_logs",
            history_mode=args.history_mode,
            reps=args.reps
        )
        test_runner.run_test()
    else:
        # For human mode, multiply specs by reps (original behavior)
        specs = specs * args.reps
        for i, spec in enumerate(specs):
         #i=0
#        while True:
            #random.shuffle(specs)
            outfile = 'scenarios_tmp.json'#
            generate_scenarios_from_tuples([specs[i]], outfile=outfile, seed=None, chartypes = [CharacterType.LIVE_PLAYER, CharacterType.HONEST_OPPONENT, CharacterType.DISHONEST_TEAMMATE, CharacterType.DISHONEST_OPPONENT])
            play_game_cli(scenario_file=outfile)

            play_again = input("\n\nDo you want to play another game? ([y]/n): ").lower().strip()
            if play_again not in ('y', ''):
                print("Thanks for playing!")
                break
            print("\n" + "="*70)
            print("--- Starting a New Game! ---")
            print("="*70 + "\n")
