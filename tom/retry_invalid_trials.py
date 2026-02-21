#!/usr/bin/env python3
"""Retry invalid trials from a token-limited LLM test run.

Usage:
    python retry_invalid_trials.py <log_file> [--max_tokens N] [--dry_run] [--scenario_file FILE]

Example:
    python retry_invalid_trials.py tom_llm_logs/deepseek-chat-v3.1_1769928664.log --max_tokens 4000
"""

import re
import json
import argparse
import os
from typing import List, Dict, Set, Optional, Tuple


def find_invalid_trial_numbers(log_file: str) -> List[int]:
    """Parse log file to find trial numbers that had 'Invalid action'.

    Returns list of trial numbers (1-indexed as in log).
    """
    with open(log_file, 'r') as f:
        content = f.read()

    invalid_trials = []
    trial_pattern = re.compile(r"Running Trial (\d+)/\d+")

    lines = content.split('\n')
    current_trial = None

    for line in lines:
        trial_match = trial_pattern.search(line)
        if trial_match:
            current_trial = int(trial_match.group(1))

        if "Invalid action:" in line and current_trial is not None:
            if current_trial not in invalid_trials:
                invalid_trials.append(current_trial)

    return invalid_trials


def load_game_data(game_data_file: str) -> tuple:
    """Load game_data.json and return (records, wrapper_or_none)."""
    with open(game_data_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data, None
    else:
        return data.get('turn_records', []), data


def get_a_record_indices(records: List[Dict]) -> List[int]:
    """Get indices of player A records in order."""
    return [i for i, r in enumerate(records) if r.get('character') == 'A']


def extract_model_name_from_filename(filename: str) -> str:
    """Extract the API model name from a log filename.

    Filenames are created with subject_id = model.replace("/", "-")
    This reverses that transformation for known provider prefixes.
    """
    basename = os.path.basename(filename)
    # Remove only timestamp and extension: model_timestamp.log -> model
    # Keep everything else including _think, _nothink, etc. as they may be part of model name
    match = re.match(r'^(.+)_\d+\.log$', basename)
    if match:
        model_id = match.group(1)
    else:
        # Fallback: split on underscore and take everything before last numeric part
        model_id = basename.rsplit('.', 1)[0]  # Remove .log
        parts = model_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            model_id = parts[0]

    # Known provider prefixes that should have a / after them
    provider_prefixes = [
        'anthropic-', 'google-', 'openai-', 'x-ai-', 'qwen-',
        'deepseek-', 'meta-', 'mistralai-', 'cohere-', 'kimi-', 'glm-'
    ]

    for prefix in provider_prefixes:
        if model_id.startswith(prefix):
            # Replace first - with / to get provider/model format
            model_name = prefix[:-1] + '/' + model_id[len(prefix):]
            return model_name

    # If no known prefix, return as-is (might work directly)
    return model_id


def parse_log_settings(log_file: str) -> dict:
    """Parse log file header to extract original run settings."""
    settings = {'free_response': True, 'history_mode': 'none', 'lose': False}
    with open(log_file, 'r') as f:
        # Only check first 50 lines for settings
        for i, line in enumerate(f):
            if i > 50:
                break
            if 'Free response:' in line:
                settings['free_response'] = 'True' in line
            if 'History mode:' in line:
                match = re.search(r'History mode:\s*(\w+)', line)
                if match:
                    settings['history_mode'] = match.group(1)
            if 'Lose warning:' in line:
                settings['lose'] = 'True' in line
    return settings


def find_scenario_in_file(scenario_file: str, scenario_id: str, extra: str, rep: int):
    """Find matching scenario by ID/Extra/Rep in pre-generated scenario file.

    Returns (scenario, chars, chartypes) or (None, None, None) if not found.
    """
    from tom_helpers import load_scenarios

    scenarios, chars, chartypes = load_scenarios(scenario_file)
    for s in scenarios:
        if s.id == scenario_id and s.extra == extra and s.rep == rep:
            return s, chars, chartypes
    return None, None, None


def verify_scenario_match(scenario, expected_desc: str, chars: List[str], chartypes) -> bool:
    """Verify scenario produces expected description.

    Must use ellipsis_mode=True and pause_mode='none' to match how scenarios
    are described in tom_test_new.py.
    """
    from tom_helpers import Character, CharacterType, Team

    # Build characters dict (same as in tom_test_new.py)
    characters = {}
    teams = [Team.BLUE, Team.RED, Team.BLUE, Team.RED, Team.NEUTRAL]
    for i, (name, ctype) in enumerate(zip(chars, chartypes)):
        characters[name] = Character(
            name=name,
            team=teams[i] if i < len(teams) else Team.NEUTRAL,
            char_type=ctype
        )

    generated = scenario.get_description_for(
        'A', characters,
        pause_mode="none",
        ellipsis_mode=True
    )
    return generated == expected_desc


def main():
    parser = argparse.ArgumentParser(description='Retry invalid trials')
    parser.add_argument('log_file', help='Log file to parse for invalid trials')
    parser.add_argument('--max_tokens', type=int, default=4000,
                        help='Max tokens for retry (default: 4000)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Just show what would be retried')
    parser.add_argument('--model', type=str,
                        help='Model name (extracted from log filename if not specified)')
    parser.add_argument('--scenario_file', type=str, default='scenarios_standardized.json',
                        help='Pre-generated scenario file (default: scenarios_standardized.json)')
    args = parser.parse_args()

    # Extract model name from log filename if not specified
    if args.model:
        model_name = args.model
    else:
        model_name = extract_model_name_from_filename(args.log_file)

    print(f"Log file: {args.log_file}")
    print(f"Model: {model_name}")
    print(f"Max tokens: {args.max_tokens}")

    # Find invalid trial numbers from log
    invalid_trial_nums = find_invalid_trial_numbers(args.log_file)
    print(f"\nFound {len(invalid_trial_nums)} invalid action entries")

    if not invalid_trial_nums:
        print("No invalid trials to retry")
        return

    # Load game data to get seeds and specs
    game_data_file = args.log_file.replace('.log', '_game_data.json')
    if not os.path.exists(game_data_file):
        print(f"Error: {game_data_file} not found")
        return

    records, wrapper = load_game_data(game_data_file)
    a_indices = get_a_record_indices(records)

    print(f"Total A records in game_data: {len(a_indices)}")

    # Map trial number (1-indexed) to record index
    # Trial 1 = a_indices[0], Trial 2 = a_indices[1], etc.
    trials_to_retry = []
    for trial_num in invalid_trial_nums:
        idx = trial_num - 1  # Convert to 0-indexed
        if idx < len(a_indices):
            record_idx = a_indices[idx]
            record = records[record_idx]

            # Get rep, with fallback to inferring from trial number
            rep = record.get('rep')
            if rep is None:
                # Infer rep from trial number (104 scenarios per rep)
                rep = ((trial_num - 1) // 104) + 1

            trials_to_retry.append({
                'trial': trial_num,
                'record_idx': record_idx,
                'scenario_id': record.get('scenario_id'),
                'extra': record.get('extra'),
                'rep': rep,
                'seed': record.get('seed'),
                'scenario_desc': record.get('scenario_desc'),
            })

    # Show distribution by extra
    by_extra = {}
    for t in trials_to_retry:
        extra = t['extra']
        by_extra[extra] = by_extra.get(extra, 0) + 1
    print(f"Distribution by Extra: {by_extra}")

    if args.dry_run:
        print(f"\nDry run - would retry (using {args.scenario_file}):")
        for t in trials_to_retry:
            print(f"  trial={t['trial']}, scenario_id={t['scenario_id']}, extra={t['extra']}, rep={t['rep']}")
        return

    # Actually retry the trials
    # Lazy imports
    from tom_helpers import load_scenarios, save_scenarios, CharacterType
    from tom_test_new import play_game_cli, BaseGameClass

    # Verify scenario file exists
    if not os.path.exists(args.scenario_file):
        print(f"Error: Scenario file {args.scenario_file} not found")
        return

    # Create minimal LLM player for retries
    base_dir = os.path.dirname(args.log_file) or '.'

    # Parse original settings from log file
    original_settings = parse_log_settings(args.log_file)
    print(f"Original settings: free_response={original_settings['free_response']}, history_mode={original_settings['history_mode']}")
    print(f"Scenario file: {args.scenario_file}")

    class RetryLLMPlayer(BaseGameClass):
        def __init__(self, model_name, log_dir, max_tokens, free_response, history_mode, lose=False):
            # subject_id uses - for filenames, subject_name uses / for API
            subject_id = model_name.replace("/", "-")
            super().__init__(subject_id, model_name, is_human_player=False, log_dir=log_dir)
            self.max_tokens = max_tokens
            self.free_response = free_response
            self.history_mode = history_mode
            self.lose = lose
            self.all_turn_records = []
            self.completed_trials = []
            self.current_trial = 0
            self.current_seed = None
            self.game_setup_text = None

    llm_player = RetryLLMPlayer(
        model_name, base_dir, args.max_tokens,
        original_settings['free_response'], original_settings['history_mode'],
        original_settings['lose']
    )

    tmp_file = 'retry_tmp.json'
    success_count = 0
    fail_count = 0

    # Load scenarios once and build lookup table
    print(f"Loading scenarios from {args.scenario_file}...")
    all_scenarios, chars, chartypes = load_scenarios(args.scenario_file)
    scenario_lookup = {(s.id, s.extra, s.rep): s for s in all_scenarios}
    print(f"Loaded {len(all_scenarios)} scenarios")

    print(f"\nRetrying {len(trials_to_retry)} trials with max_tokens={args.max_tokens}...")

    for i, trial_info in enumerate(trials_to_retry):
        scenario_id = trial_info['scenario_id']
        extra = trial_info['extra']
        rep = trial_info['rep']
        record_idx = trial_info['record_idx']
        expected_desc = trial_info['scenario_desc']

        print(f"  [{i+1}/{len(trials_to_retry)}] trial={trial_info['trial']}, "
              f"scenario_id={scenario_id}, extra={extra}, rep={rep}")

        try:
            # Find scenario in lookup table
            key = (scenario_id, extra, rep)
            matched_scenario = scenario_lookup.get(key)

            if matched_scenario is None:
                print(f"    Error: scenario not found for id={scenario_id}, extra={extra}, rep={rep}")
                fail_count += 1
                continue

            # Verify scenario matches original (if we have the description)
            if expected_desc:
                if not verify_scenario_match(matched_scenario, expected_desc, chars, chartypes):
                    print(f"    Warning: scenario description mismatch, proceeding anyway")

            # Save single scenario to temp file
            save_scenarios([matched_scenario], tmp_file, chars, chartypes)

            # Run the game
            game_state = play_game_cli(
                scenario_file=tmp_file,
                llm_player=llm_player,
                max_tokens_override=args.max_tokens
            )

            # Extract the new record
            if game_state and hasattr(game_state, 'turn_records') and game_state.turn_records:
                from dataclasses import asdict
                for new_record in game_state.turn_records:
                    if new_record.character == 'A':
                        # Convert dataclass to dict for JSON storage
                        new_record_dict = asdict(new_record)
                        # Mark as retry
                        new_record_dict['retry'] = True

                        # Check if valid action now
                        action = new_record_dict.get('action', '')
                        is_valid = action.startswith(('Pass', 'Ask', 'Tell'))

                        print(f"    -> action={action[:50]}{'...' if len(action) > 50 else ''}, "
                              f"valid={is_valid}, was_optimal={new_record_dict.get('was_optimal')}")

                        # Replace the record
                        records[record_idx] = new_record_dict
                        success_count += 1
                        break
            else:
                print(f"    Error: no turn records returned")
                fail_count += 1

        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    # Cleanup temp file
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    print(f"\nResults: {success_count} succeeded, {fail_count} failed")

    if success_count > 0:
        # Backup original
        backup_file = game_data_file.replace('.json', '_backup.json')
        if not os.path.exists(backup_file):
            import shutil
            shutil.copy(game_data_file, backup_file)
            print(f"Backed up original to: {backup_file}")

        # Save updated data
        if wrapper:
            wrapper['turn_records'] = records
            output = wrapper
        else:
            output = records

        with open(game_data_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Updated: {game_data_file}")


if __name__ == '__main__':
    main()
