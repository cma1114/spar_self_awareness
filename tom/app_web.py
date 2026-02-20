#!/usr/bin/env python3
"""
Flask web app for Theory of Mind human baseline study.

Wraps the existing game logic (tom_test_new.py / tom_helpers.py) in a web interface
for use with Prolific participants.

Routes:
    GET  /                  → Landing page, reads Prolific params, assigns rep
    POST /start             → Stores session, redirects to first scenario
    GET  /scenario/<n>      → Displays scenario n
    POST /scenario/<n>      → Processes action, saves result, redirects to next
    GET  /complete           → Completion page with Prolific redirect
"""

import os
import json
import time
import random
import fcntl
from dataclasses import asdict
from flask import Flask, render_template, request, redirect, url_for, session, abort

# Import game logic
from tom_helpers import load_scenarios, save_scenarios, CharacterType, Team
from tom_test_new import (
    GameState, TurnRecord, Action, ActionType,
    save_game_results, GAME_SETUP_TEMPLATE, LOSE_WARNING,
)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-change-in-production')

# --- Configuration ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENARIO_FILE = os.path.join(_SCRIPT_DIR, 'scenarios_standardized.json')
BASE_DATA_DIR = os.path.join(_SCRIPT_DIR, 'tom_human_logs')
DEFAULT_STUDY_ID = 'default'
DROPOUT_TIMEOUT_SECONDS = 2 * 60 * 60  # 2 hours
PROLIFIC_COMPLETION_URL = os.environ.get(
    'PROLIFIC_COMPLETION_URL',
    'https://app.prolific.com/submissions/complete?cc=XXXXXXX'  # Replace with actual code
)

# Precompute rep info at startup
_scenarios, _chars, _chartypes = load_scenarios(SCENARIO_FILE)
TOTAL_REPS = len(set(s.rep for s in _scenarios))


def study_data_dir(study_id):
    """Return the data directory for a given study."""
    safe_id = study_id or DEFAULT_STUDY_ID
    return os.path.join(BASE_DATA_DIR, safe_id)


def study_assignments_file(study_id):
    """Return the rep assignments file path for a given study."""
    return os.path.join(study_data_dir(study_id), 'rep_assignments.json')

# Game rules text (shown on instructions page)
RULES_TEXT = GAME_SETUP_TEMPLATE.format(lose_text="").format(WINNING_SCORE=4)
STUDY_CONTEXT = (
    "For the purpose of this study, we're going to repeatedly ask you to "
    "make the first move in different runs of this game. Each run is separate "
    "and in each run we want you to make the best move in that situation."
)


# --- Rep Assignment ---

def _load_assignments(study_id):
    """Load assignments file, creating it if needed."""
    data_dir = study_data_dir(study_id)
    assignments_file = study_assignments_file(study_id)
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(assignments_file):
        return {"assignments": []}
    with open(assignments_file, 'r') as f:
        return json.load(f)


def _save_assignments(study_id, data):
    """Save assignments file."""
    assignments_file = study_assignments_file(study_id)
    with open(assignments_file, 'w') as f:
        json.dump(data, f, indent=2)


def assign_rep(participant_id, study_id):
    """Assign a rep to a participant using file locking for concurrency safety.

    Priority:
    1. Reps with timed-out in_progress assignments (dropout recovery)
    2. Reps with no assignments yet
    3. Reps with fewest completions (for wrapping around with >10 participants)

    Returns the assigned rep number (1-indexed).
    """
    data_dir = study_data_dir(study_id)
    lock_path = study_assignments_file(study_id) + '.lock'
    os.makedirs(data_dir, exist_ok=True)

    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            data = _load_assignments(study_id)
            now = time.time()

            # Check if this participant already has an active assignment
            for entry in data['assignments']:
                if entry['participant'] == participant_id and entry['status'] == 'in_progress':
                    return entry['rep']

            # Count completions and find timed-out assignments per rep
            rep_completions = {r: 0 for r in range(1, TOTAL_REPS + 1)}
            rep_in_progress = {r: [] for r in range(1, TOTAL_REPS + 1)}

            for entry in data['assignments']:
                rep = entry['rep']
                if entry['status'] == 'complete':
                    rep_completions[rep] += 1
                elif entry['status'] == 'in_progress':
                    rep_in_progress[rep].append(entry)

            # Priority 1: Reps with timed-out in_progress (dropout recovery)
            for rep in range(1, TOTAL_REPS + 1):
                for entry in rep_in_progress[rep]:
                    if now - entry['started_at'] > DROPOUT_TIMEOUT_SECONDS:
                        # Mark old entry as timed_out and reassign this rep
                        entry['status'] = 'timed_out'
                        assigned_rep = rep
                        data['assignments'].append({
                            'participant': participant_id,
                            'rep': assigned_rep,
                            'status': 'in_progress',
                            'started_at': now,
                        })
                        _save_assignments(study_id, data)
                        return assigned_rep

            # Priority 2: Reps with no assignments at all
            assigned_reps = set()
            for entry in data['assignments']:
                if entry['status'] in ('complete', 'in_progress'):
                    assigned_reps.add(entry['rep'])

            for rep in range(1, TOTAL_REPS + 1):
                if rep not in assigned_reps:
                    data['assignments'].append({
                        'participant': participant_id,
                        'rep': rep,
                        'status': 'in_progress',
                        'started_at': now,
                    })
                    _save_assignments(study_id, data)
                    return rep

            # Priority 3: Rep with fewest completions
            min_completions = min(rep_completions.values())
            for rep in range(1, TOTAL_REPS + 1):
                if rep_completions[rep] == min_completions:
                    # Check it doesn't have an active in_progress
                    active = [e for e in rep_in_progress[rep]
                              if now - e['started_at'] <= DROPOUT_TIMEOUT_SECONDS]
                    if not active:
                        data['assignments'].append({
                            'participant': participant_id,
                            'rep': rep,
                            'status': 'in_progress',
                            'started_at': now,
                        })
                        _save_assignments(study_id, data)
                        return rep

            # Fallback: just pick the least-completed rep
            best_rep = min(rep_completions, key=rep_completions.get)
            data['assignments'].append({
                'participant': participant_id,
                'rep': best_rep,
                'status': 'in_progress',
                'started_at': now,
            })
            _save_assignments(study_id, data)
            return best_rep

        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def mark_complete(participant_id, study_id):
    """Mark a participant's assignment as complete."""
    lock_path = study_assignments_file(study_id) + '.lock'

    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            data = _load_assignments(study_id)
            for entry in data['assignments']:
                if entry['participant'] == participant_id and entry['status'] == 'in_progress':
                    entry['status'] = 'complete'
                    entry['completed_at'] = time.time()
                    break
            _save_assignments(study_id, data)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


# --- Scenario Loading ---

def load_participant_scenarios(rep):
    """Load and shuffle scenarios for a given rep."""
    scenarios, chars, chartypes = load_scenarios(SCENARIO_FILE)
    rep_scenarios = [s for s in scenarios if s.rep == rep]
    random.shuffle(rep_scenarios)
    return rep_scenarios, chars, chartypes


# --- Game Processing ---

def process_scenario_for_display(scenario, chars, chartypes, study_id):
    """Process a scenario and return display text and game state needed for action processing.

    Returns dict with scenario_desc, question_desc, and serialized game state.
    """
    # Create a temporary scenario file (GameState requires a file path)
    data_dir = study_data_dir(study_id)
    tmp_file = os.path.join(data_dir, f'_tmp_{os.getpid()}.json')
    save_scenarios([scenario], tmp_file, chars, chartypes)

    try:
        game = GameState(tmp_file)
        scenario_obj = game.get_current_scenario()
        true_contents = game.process_scenario_events(scenario_obj)

        scenario_desc = scenario_obj.get_description_for(
            'A', game.characters, pause_mode="none", ellipsis_mode=True
        )
        answerer = "you" if scenario_obj.who_answers == 'A' else scenario_obj.who_answers
        question_desc = f"I am going to ask {answerer} what is in the {scenario_obj.question_container}."

        return {
            'scenario_desc': scenario_desc,
            'question_desc': question_desc,
            'tmp_file': tmp_file,  # Keep for action processing
        }
    except Exception:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        raise


def process_action(scenario, chars, chartypes, action_str, study_id):
    """Process a player action and return a TurnRecord.

    This replicates the turn processing logic from play_game_cli.
    """
    data_dir = study_data_dir(study_id)
    tmp_file = os.path.join(data_dir, f'_tmp_{os.getpid()}.json')
    save_scenarios([scenario], tmp_file, chars, chartypes)

    try:
        game = GameState(tmp_file)
        scenario_obj = game.get_current_scenario()
        true_contents = game.process_scenario_events(scenario_obj)

        scenario_desc = scenario_obj.get_description_for(
            'A', game.characters, pause_mode="none", ellipsis_mode=True
        )
        answerer_name = "you" if scenario_obj.who_answers == 'A' else scenario_obj.who_answers
        question_desc = f"I am going to ask {answerer_name} what is in the {scenario_obj.question_container}."

        # Parse action
        action = game.parse_action(action_str)
        if not action:
            action = Action(ActionType.PASS)
            # Keep action_str as-is to record invalid input

        # Execute action
        score_delta, action_desc = game.execute_action('A', action, true_contents)

        # Resolve answer
        answer_given, is_correct, answer_score = game.resolve_answer_phase(scenario_obj, true_contents)

        # Score calculation
        blue_delta = score_delta
        if is_correct and scenario_obj.who_answers in ['A', 'B']:
            blue_delta += answer_score
        game.scores[Team.BLUE] += blue_delta

        # Optimality
        was_optimal = game.is_action_optimal(action_str, scenario_obj, true_contents)
        expected_action_obj = game.execute_npc_action('A', scenario_obj, true_contents)
        if expected_action_obj.action_type == ActionType.PASS:
            expected_action_str = "Pass"
        elif expected_action_obj.action_type == ActionType.ASK:
            expected_action_str = f"Ask({expected_action_obj.target_char}, {expected_action_obj.container})"
        elif expected_action_obj.action_type == ActionType.TELL:
            expected_action_str = f"Tell({expected_action_obj.target_char}, {expected_action_obj.container}, {expected_action_obj.contents})"
        else:
            expected_action_str = "Pass"

        # Ask-related fields
        asked_player = ""
        asked_player_present = ""
        ask_container_matches = ""
        if action.action_type == ActionType.ASK:
            asked_player = action.target_char
            present_at_end = game.get_present_at_end(scenario_obj)
            asked_player_present = "TRUE" if action.target_char in present_at_end else "FALSE"
            ask_container_matches = "TRUE" if action.container == scenario_obj.question_container else "FALSE"

        # Tell-related fields
        told_player = ""
        told_player_present = ""
        tell_truthful_about_question = ""
        if action.action_type == ActionType.TELL:
            told_player = action.target_char
            present_at_end = game.get_present_at_end(scenario_obj)
            told_player_present = "TRUE" if action.target_char in present_at_end else "FALSE"
            if action.container == scenario_obj.question_container:
                player_belief = game.characters['A'].get_belief(action.container)
                if player_belief is None:
                    tell_truthful_about_question = "TRUE" if action.contents in ("empty", "nothing") else "FALSE"
                else:
                    tell_truthful_about_question = "TRUE" if action.contents == player_belief else "FALSE"

        # Lie detection
        lied_to_opponent_answerer = ""
        if action.action_type == ActionType.TELL:
            if scenario_obj.who_answers in ['C', 'D'] and action.target_char == scenario_obj.who_answers:
                actual_contents = true_contents.get(action.container)
                lied_to_opponent_answerer = "FALSE" if action.contents == actual_contents else "TRUE"

        # Event-based fields
        a_leave_idx = None
        b_leave_idx = None
        for idx, event in enumerate(scenario_obj.events):
            if event.event_type == 'leave':
                if event.character == 'A':
                    a_leave_idx = idx
                elif event.character == 'B':
                    b_leave_idx = idx

        b_left_before_a = ""
        if a_leave_idx is not None and b_leave_idx is not None:
            b_left_before_a = "TRUE" if b_leave_idx < a_leave_idx else "FALSE"

        a_left_before_put = ""
        if a_leave_idx is not None:
            any_put_before = any(idx < a_leave_idx for idx, event in enumerate(scenario_obj.events)
                                 if event.event_type == 'put')
            a_left_before_put = "FALSE" if any_put_before else "TRUE"

        b_put_or_moved = "TRUE" if any(
            (event.event_type in ('put', 'move')) and event.character == 'B'
            for event in scenario_obj.events
        ) else "FALSE"

        turn_record = TurnRecord(
            round_num=scenario_obj.round_num,
            scenario_id=scenario_obj.id,
            extra=scenario_obj.extra,
            character='A',
            scenario_desc=scenario_desc,
            question=question_desc,
            action=action_str,
            action_cost=abs(score_delta),
            answer_given=answer_given,
            answer_correct=is_correct,
            answer_score=answer_score,
            optimal_action=expected_action_str,
            was_optimal=was_optimal,
            blue_score_after=game.scores[Team.BLUE],
            red_score_after=game.scores[Team.RED],
            epistemic_type=scenario_obj.epistemic_type.value if scenario_obj.epistemic_type else None,
            ask_constraint=scenario_obj.ask_constraint.value if scenario_obj.ask_constraint else None,
            ks_self=scenario_obj.ks_self,
            ks_teammate=scenario_obj.ks_teammate,
            ks_opponent=scenario_obj.ks_opponent,
            answerer=scenario_obj.who_answers,
            asked_player=asked_player,
            asked_player_present=asked_player_present,
            ask_container_matches=ask_container_matches,
            told_player=told_player,
            told_player_present=told_player_present,
            tell_truthful_about_question=tell_truthful_about_question,
            b_left_before_a=b_left_before_a,
            a_left_before_put=a_left_before_put,
            b_put_or_moved=b_put_or_moved,
            history_mode=None,
            trial=None,
            pause_mode="none",
            rep=scenario_obj.rep,
            seed=None,
            situation_event_count=scenario_obj.situation_event_count,
            ect_certainty=scenario_obj.epistemic_transitions.get('certainty') if scenario_obj.epistemic_transitions else None,
            ect_accuracy=scenario_obj.epistemic_transitions.get('accuracy') if scenario_obj.epistemic_transitions else None,
            ect_total=scenario_obj.epistemic_transitions.get('total') if scenario_obj.epistemic_transitions else None,
        )

        return turn_record

    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


# --- Participant Data Persistence ---

def _participant_data_path(participant_id, study_id):
    """Path to a participant's in-progress data file."""
    return os.path.join(study_data_dir(study_id), f'{participant_id}_progress.json')


def _participant_output_path(participant_id, study_id):
    """Path to a participant's final output file."""
    timestamp = int(time.time())
    return os.path.join(study_data_dir(study_id), f'{study_id}_{participant_id}_{timestamp}_game_data.json')


def save_progress(participant_id, study_id, turn_records_dicts, scenario_order):
    """Save in-progress results to disk (called after each scenario)."""
    path = _participant_data_path(participant_id, study_id)
    with open(path, 'w') as f:
        json.dump({
            'turn_records': turn_records_dicts,
            'scenario_order': scenario_order,
        }, f)


def load_progress(participant_id, study_id):
    """Load in-progress results from disk."""
    path = _participant_data_path(participant_id, study_id)
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


# --- Routes ---

@app.route('/')
def landing():
    """Landing page. Reads Prolific params and assigns a rep."""
    prolific_pid = request.args.get('PROLIFIC_PID', '')
    study_id = request.args.get('STUDY_ID', '')
    session_id = request.args.get('SESSION_ID', '')

    if not prolific_pid:
        # Allow manual testing without Prolific
        prolific_pid = f"test_{int(time.time())}"

    if not study_id:
        study_id = DEFAULT_STUDY_ID

    # Ensure study directory exists
    os.makedirs(study_data_dir(study_id), exist_ok=True)

    # Assign rep
    rep = assign_rep(prolific_pid, study_id)

    # Store in session
    session['participant_id'] = prolific_pid
    session['study_id'] = study_id
    session['session_id'] = session_id
    session['rep'] = rep
    session['current_scenario'] = 0

    # Load and shuffle scenarios, save the order
    scenarios, chars, chartypes = load_participant_scenarios(rep)
    scenario_order = [s.id for s in scenarios]  # Track order for reproducibility
    save_progress(prolific_pid, study_id, [], scenario_order)

    # Store scenario order in session (just the IDs - actual scenarios loaded from file each time)
    session['scenario_order'] = scenario_order
    session['total_scenarios'] = len(scenarios)

    return render_template('instructions.html',
                           rules_text=RULES_TEXT,
                           study_context=STUDY_CONTEXT,
                           participant_id=prolific_pid,
                           total_scenarios=len(scenarios))


@app.route('/start', methods=['POST'])
def start():
    """Redirect to first scenario after participant reads instructions."""
    if 'participant_id' not in session:
        return redirect(url_for('landing'))
    return redirect(url_for('scenario', n=1))


@app.route('/scenario/<int:n>', methods=['GET', 'POST'])
def scenario(n):
    """Display or process a scenario."""
    if 'participant_id' not in session:
        return redirect(url_for('landing'))

    participant_id = session['participant_id']
    study_id = session['study_id']
    total = session['total_scenarios']
    rep = session['rep']
    scenario_order = session['scenario_order']

    if n < 1 or n > total:
        return redirect(url_for('complete'))

    # Load the specific scenario
    scenarios, chars, chartypes = load_participant_scenarios(rep)
    # Re-sort to match saved order
    scenario_map = {s.id: s for s in scenarios}
    ordered_scenarios = [scenario_map[sid] for sid in scenario_order]
    current_scenario = ordered_scenarios[n - 1]

    if request.method == 'GET':
        # Display the scenario
        display = process_scenario_for_display(current_scenario, chars, chartypes, study_id)
        # Clean up temp file
        if os.path.exists(display.get('tmp_file', '')):
            os.remove(display['tmp_file'])

        return render_template('scenario.html',
                               scenario_num=n,
                               total_scenarios=total,
                               scenario_desc=display['scenario_desc'],
                               question_desc=display['question_desc'],
                               rules_text=RULES_TEXT)

    else:  # POST
        action_str = request.form.get('action', 'Pass').strip()

        # Process the action
        turn_record = process_action(current_scenario, chars, chartypes, action_str, study_id)

        # Load existing progress and append
        progress = load_progress(participant_id, study_id) or {'turn_records': [], 'scenario_order': scenario_order}
        progress['turn_records'].append(asdict(turn_record))
        save_progress(participant_id, study_id, progress['turn_records'], scenario_order)

        # Advance to next scenario or complete
        if n >= total:
            return redirect(url_for('complete'))
        else:
            return redirect(url_for('scenario', n=n + 1))


@app.route('/complete')
def complete():
    """Completion page. Save final results and redirect to Prolific."""
    if 'participant_id' not in session:
        return redirect(url_for('landing'))

    participant_id = session['participant_id']
    study_id = session['study_id']

    # Load progress and save as final output
    progress = load_progress(participant_id, study_id)
    if progress and progress['turn_records']:
        # Convert dicts back to TurnRecords for save_game_results
        turn_records = []
        for d in progress['turn_records']:
            turn_records.append(TurnRecord(**{
                k: v for k, v in d.items()
                if k in TurnRecord.__dataclass_fields__
            }))

        # Save final output
        output_path = _participant_output_path(participant_id, study_id)
        game_setup_text = RULES_TEXT + "\n\n" + STUDY_CONTEXT
        save_game_results(turn_records, output_path, game_setup_text)

    # Mark assignment as complete
    mark_complete(participant_id, study_id)

    # Clean up progress file
    progress_path = _participant_data_path(participant_id, study_id)
    if os.path.exists(progress_path):
        os.remove(progress_path)

    return render_template('complete.html',
                           prolific_completion_url=PROLIFIC_COMPLETION_URL)


if __name__ == '__main__':
    os.makedirs(BASE_DATA_DIR, exist_ok=True)
    app.run(debug=True, port=5000)
