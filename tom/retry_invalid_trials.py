#!/usr/bin/env python3
"""Retry invalid trials from a token-limited LLM test run.

Usage:
    python retry_invalid_trials.py <log_file> [--max_tokens N] [--dry_run]

Example:
    python retry_invalid_trials.py tom_llm_logs/deepseek-chat-v3.1_1769928664.log --max_tokens 4000
"""

import re
import json
import argparse
import os
from typing import List, Dict, Set


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


def parse_log_settings(log_file: str) -> dict:
    """Parse log file header to extract original run settings."""
    settings = {'free_response': True, 'history_mode': 'none'}
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
    return settings


def main():
    parser = argparse.ArgumentParser(description='Retry invalid trials')
    parser.add_argument('log_file', help='Log file to parse for invalid trials')
    parser.add_argument('--max_tokens', type=int, default=4000,
                        help='Max tokens for retry (default: 4000)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Just show what would be retried')
    parser.add_argument('--model', type=str,
                        help='Model name (extracted from log filename if not specified)')
    args = parser.parse_args()

    # Extract model name from log filename if not specified
    if args.model:
        model_name = args.model
    else:
        basename = os.path.basename(args.log_file)
        model_name = basename.split('_')[0]

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
            trials_to_retry.append({
                'trial': trial_num,
                'record_idx': record_idx,
                'scenario_id': record.get('scenario_id'),
                'extra': record.get('extra'),
                'seed': record.get('seed'),
            })

    # Show distribution by extra
    by_extra = {}
    for t in trials_to_retry:
        extra = t['extra']
        by_extra[extra] = by_extra.get(extra, 0) + 1
    print(f"Distribution by Extra: {by_extra}")

    if args.dry_run:
        print("\nDry run - would retry:")
        for t in trials_to_retry:
            print(f"  trial={t['trial']}, scenario_id={t['scenario_id']}, extra={t['extra']}, seed={t['seed']}")
        return

    # Actually retry the trials
    # Lazy imports
    from tom_helpers import read_specs_from_csv, CharacterType
    from generate_tom_scenarios_new import generate_scenarios_from_tuples
    from tom_test_new import play_game_cli, BaseGameClass

    # Load specs
    all_specs = read_specs_from_csv('ToM - scenarios.csv')
    spec_lookup = {}
    for spec in all_specs:
        key = (spec['Id'], spec['Extra'])
        spec_lookup[key] = spec

    chartypes = [
        CharacterType.LIVE_PLAYER,
        CharacterType.HONEST_OPPONENT,
        CharacterType.DISHONEST_TEAMMATE,
        CharacterType.DISHONEST_OPPONENT,
        CharacterType.NEUTRAL
    ]

    # Create minimal LLM player for retries
    base_dir = os.path.dirname(args.log_file) or '.'

    # Parse original settings from log file
    original_settings = parse_log_settings(args.log_file)
    print(f"Original settings: free_response={original_settings['free_response']}, history_mode={original_settings['history_mode']}")

    class RetryLLMPlayer(BaseGameClass):
        def __init__(self, model_name, log_dir, max_tokens, free_response, history_mode):
            super().__init__(model_name, model_name, is_human_player=False, log_dir=log_dir)
            self.max_tokens = max_tokens
            self.free_response = free_response
            self.history_mode = history_mode
            self.all_turn_records = []
            self.completed_trials = []
            self.current_trial = 0
            self.current_seed = None
            self.game_setup_text = None

    llm_player = RetryLLMPlayer(
        model_name, base_dir, args.max_tokens,
        original_settings['free_response'], original_settings['history_mode']
    )

    tmp_file = 'retry_tmp.json'
    success_count = 0
    fail_count = 0

    print(f"\nRetrying {len(trials_to_retry)} trials with max_tokens={args.max_tokens}...")

    for i, trial_info in enumerate(trials_to_retry):
        scenario_id = trial_info['scenario_id']
        extra = trial_info['extra']
        seed = trial_info['seed']
        record_idx = trial_info['record_idx']

        print(f"  [{i+1}/{len(trials_to_retry)}] trial={trial_info['trial']}, "
              f"scenario_id={scenario_id}, extra={extra}, seed={seed}")

        spec_key = (scenario_id, extra)
        if spec_key not in spec_lookup:
            print(f"    Error: spec not found for {spec_key}")
            fail_count += 1
            continue

        spec = spec_lookup[spec_key]

        try:
            # Generate scenario with EXACT same seed
            generate_scenarios_from_tuples([spec], outfile=tmp_file, seed=seed, chartypes=chartypes)

            llm_player.current_seed = seed

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
                        # Preserve original seed
                        new_record_dict['seed'] = seed
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
