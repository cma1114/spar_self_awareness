#!/usr/bin/env python3
"""Recover turn record data from a crashed LLM test run log file.

Usage:
    python recover_log_data.py <log_file> [--allow-incomplete]

Example:
    python recover_log_data.py tom_llm_logs/anthropic-claude-opus-4.6_1771535483.log
    python recover_log_data.py tom_llm_logs/example.log --allow-incomplete
"""

import argparse
import re
import json
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def parse_epistemic_state(state_str: str) -> str:
    """Extract epistemic state value from log format like <EpistemicState.KNOWS_X: 'Knows X'>"""
    match = re.search(r"'([^']+)'", state_str)
    return match.group(1) if match else state_str


def parse_action(action_text: str) -> Dict:
    """Parse action from log text, returning action type and details."""
    action_text = action_text.strip()

    # Handle Pass
    if action_text.lower() == 'pass' or action_text.lower() == 'pass.':
        return {'type': 'Pass', 'normalized': 'Pass'}

    # Handle Ask(Player, Container)
    ask_match = re.match(r'Ask\s*\(\s*([A-D])\s*,\s*(\w+)\s*\)', action_text, re.IGNORECASE)
    if ask_match:
        player, container = ask_match.groups()
        return {
            'type': 'Ask',
            'player': player.upper(),
            'container': container.lower(),
            'normalized': f'Ask {player.upper()}'
        }

    # Handle Tell(Player, Container, Contents)
    tell_match = re.match(r'Tell\s*\(\s*([A-D])\s*,\s*(\w+)\s*,\s*(\w+)\s*\)', action_text, re.IGNORECASE)
    if tell_match:
        player, container, contents = tell_match.groups()
        return {
            'type': 'Tell',
            'player': player.upper(),
            'container': container.lower(),
            'contents': contents.lower(),
            'normalized': f'Tell {player.upper()}'
        }

    return {'type': 'Unknown', 'raw': action_text, 'normalized': action_text}


def normalize_optimal_action(action: str) -> str:
    """Normalize optimal action from spec for comparison."""
    action = action.strip()
    if action.lower() == 'pass':
        return 'Pass'
    # "Ask Teammate" -> "Ask B", "Tell Teammate" -> "Tell B"
    if 'Teammate' in action:
        return action.replace('Teammate', 'B')
    if 'Opponent' in action:
        return action.replace('Opponent', 'C')  # Default to C
    return action


def is_action_optimal(parsed_action: Dict, optimal_action: str) -> bool:
    """Check if parsed action matches optimal action."""
    normalized_optimal = normalize_optimal_action(optimal_action)
    parsed_normalized = parsed_action.get('normalized', '')

    # Direct match
    if parsed_normalized == normalized_optimal:
        return True

    # Handle "Ask B" matching "Ask Teammate"
    if parsed_action['type'] == 'Ask' and 'Ask' in optimal_action:
        if parsed_action.get('player') == 'B' and 'Teammate' in optimal_action:
            return True

    if parsed_action['type'] == 'Tell' and 'Tell' in optimal_action:
        if parsed_action.get('player') == 'B' and 'Teammate' in optimal_action:
            return True

    return False


def parse_trial_block(block: str, trial_num: int, header_info: Dict) -> Optional[Dict]:
    """Parse a single trial block into a turn record."""

    # Extract spec info from trial header
    # Format: --- Running Trial N/M (Spec ID X): {...} ---
    # Or:     --- Running Trial N/M (Rep R, Spec ID X): {...} ---
    header_match = re.search(
        r"Running Trial (\d+)/\d+ \((?:Rep (\d+), )?Spec ID (\d+)\): \{([^}]+)\}",
        block
    )
    if not header_match:
        return None

    trial_in_block = int(header_match.group(1))
    rep_from_header = header_match.group(2)  # May be None for old format
    spec_id = header_match.group(3)
    spec_dict_str = header_match.group(4)

    # Parse spec fields
    # Extract actual scenario Id (not Spec ID which is just row number)
    id_match = re.search(r"'Id':\s*'(\d+)'", spec_dict_str)
    actual_scenario_id = id_match.group(1) if id_match else spec_id

    extra_match = re.search(r"'Extra':\s*'(\w+)'", spec_dict_str)
    extra = extra_match.group(1) if extra_match else None

    answerer_match = re.search(r"'Answerer':\s*'(\w+)'", spec_dict_str)
    answerer = answerer_match.group(1) if answerer_match else 'Self'

    action_match = re.search(r"'Action':\s*'([^']+)'", spec_dict_str)
    optimal_action = action_match.group(1) if action_match else 'Pass'

    ks_self_match = re.search(r"'KS_Self':\s*<[^>]+:\s*'([^']+)'>", spec_dict_str)
    ks_self = ks_self_match.group(1) if ks_self_match else None

    ks_teammate_match = re.search(r"'KS_Teammate':\s*<[^>]+:\s*'([^']+)'>", spec_dict_str)
    ks_teammate = ks_teammate_match.group(1) if ks_teammate_match else None

    ks_opponent_match = re.search(r"'KS_Opponent':\s*<[^>]+:\s*'([^']+)'>", spec_dict_str)
    ks_opponent = ks_opponent_match.group(1) if ks_opponent_match else None

    # Extract scenario description
    scenario_match = re.search(
        r'-{40,}\s*\n(.+?)\n-{40,}',
        block,
        re.DOTALL
    )
    scenario_desc = scenario_match.group(1).strip() if scenario_match else ""

    # Extract question
    question_match = re.search(r'(I am going to ask .+?\.)', block)
    question = question_match.group(1) if question_match else ""

    # Extract action - try multiple patterns
    action_text = None

    # Pattern 1: "My action: X" or "My action is: X"
    action_match = re.search(r'My action(?:\s+is)?:\s*(.+)', block, re.IGNORECASE)
    if action_match:
        action_text = action_match.group(1).strip()

    # Pattern 2: "Action: X" at end of reasoning (before "A answers:")
    if not action_text:
        # Find the last "Action: X" before "answers:"
        answer_pos = block.find(' answers:')
        if answer_pos > 0:
            section_before = block[:answer_pos]
            # Find all "Action: X" patterns and take the last one
            action_matches = list(re.finditer(r'\nAction:\s*(Pass|Ask\s*\([^)]+\)|Tell\s*\([^)]+\))', section_before, re.IGNORECASE))
            if action_matches:
                action_text = action_matches[-1].group(1).strip()

    # Pattern 3: "I will take the action: X" or "the action: X" or "choose to X"
    if not action_text:
        action_match = re.search(r'(?:take\s+)?the\s+action:\s*(Pass|Ask\s*\([^)]+\)|Tell\s*\([^)]+\))', block, re.IGNORECASE)
        if action_match:
            action_text = action_match.group(1).strip()

    # Pattern 4: "I will choose to X" or "choose to X"
    if not action_text:
        action_match = re.search(r'choose\s+to\s+(Pass|Ask\s*\([^)]+\)|Tell\s*\([^)]+\))', block, re.IGNORECASE)
        if action_match:
            action_text = action_match.group(1).strip()

    # Pattern 5: "my action would be: X"
    if not action_text:
        action_match = re.search(r'my\s+action\s+would\s+be:\s*(Pass|Ask\s*\([^)]+\)|Tell\s*\([^)]+\))', block, re.IGNORECASE)
        if action_match:
            action_text = action_match.group(1).strip()

    # Pattern 6: Markdown bold "**Action:** Pass" or "**Action:** Ask(...)"
    if not action_text:
        action_match = re.search(r'\*\*Action:\*\*\s*(Pass|Ask\s*\([^)]+\)|Tell\s*\([^)]+\))', block, re.IGNORECASE)
        if action_match:
            action_text = action_match.group(1).strip()

    # Pattern 7: "Action: Answer: Pass" or similar
    if not action_text:
        action_match = re.search(r'Action:\s*Answer:\s*(Pass|Ask\s*\([^)]+\)|Tell\s*\([^)]+\))', block, re.IGNORECASE)
        if action_match:
            action_text = action_match.group(1).strip()

    # Pattern 8: Fallback - look for any "Pass", "Ask(...)", "Tell(...)" on its own line in the action section
    if not action_text:
        action_section = re.search(r'ACTION PHASE.*?([A-D]) answers:', block, re.DOTALL)
        if action_section:
            section = action_section.group(0)
            # Look for Pass anywhere (more lenient)
            pass_match = re.search(r'\b(Pass)\b', section, re.IGNORECASE)
            ask_match = re.search(r'(Ask\s*\([^)]+\))', section, re.IGNORECASE)
            tell_match = re.search(r'(Tell\s*\([^)]+\))', section, re.IGNORECASE)
            # Prefer Ask/Tell over Pass if both exist (Pass is more common false positive)
            if ask_match:
                action_text = ask_match.group(1)
            elif tell_match:
                action_text = tell_match.group(1)
            elif pass_match:
                action_text = pass_match.group(1)

    if not action_text:
        # Pattern 2: Look for standalone action lines before "A answers:"
        # Find the section between ACTION PHASE and "A answers:"
        action_section = re.search(r'ACTION PHASE.*?([A-D]) answers:', block, re.DOTALL)
        if action_section:
            section = action_section.group(0)
            # Look for Pass, Ask(...), or Tell(...)
            pass_match = re.search(r'\n(Pass\.?)\s*\n', section, re.IGNORECASE)
            ask_match = re.search(r'\n(Ask\s*\([^)]+\))\s*\n', section, re.IGNORECASE)
            tell_match = re.search(r'\n(Tell\s*\([^)]+\))\s*\n', section, re.IGNORECASE)

            if pass_match:
                action_text = pass_match.group(1)
            elif ask_match:
                action_text = ask_match.group(1)
            elif tell_match:
                action_text = tell_match.group(1)

    if not action_text:
        return None

    parsed_action = parse_action(action_text)

    # Extract answer and correctness
    answer_match = re.search(r'([A-D]) answers:\s*(\w+)', block)
    answer_given = answer_match.group(2) if answer_match else ""
    answerer_char = answer_match.group(1) if answer_match else "A"

    # Check if correct
    answer_correct = 'Correct!' in block and 'Incorrect' not in block

    # Calculate rep (1-indexed from header, or compute from trial number)
    scenarios_per_rep = header_info.get('scenarios_per_rep', 104)
    if rep_from_header:
        rep = int(rep_from_header)
    else:
        rep = ((trial_in_block - 1) // scenarios_per_rep) + 1

    # Determine was_optimal
    was_optimal = is_action_optimal(parsed_action, optimal_action)

    # Build turn record
    record = {
        "round_num": 1,
        "character": "A",
        "scenario_desc": scenario_desc,
        "question": question,
        "action": action_text,
        "action_cost": 0.0 if parsed_action['type'] == 'Pass' else 0.5,
        "answer_given": answer_given,
        "answer_correct": answer_correct,
        "answer_score": 1.0 if answer_correct else 0.0,
        "optimal_action": optimal_action,
        "was_optimal": was_optimal,
        "blue_score_after": None,  # Could parse from Outcome line
        "red_score_after": None,
        "scenario_id": actual_scenario_id,
        "extra": extra,
        "epistemic_type": None,
        "ask_constraint": None,
        "ks_self": ks_self,
        "ks_teammate": ks_teammate,
        "ks_opponent": ks_opponent,
        "answerer": answerer_char,
        "asked_player": parsed_action.get('player', '') if parsed_action['type'] == 'Ask' else '',
        "asked_player_present": "",
        "ask_container_matches": "",
        "told_player": parsed_action.get('player', '') if parsed_action['type'] == 'Tell' else '',
        "told_player_present": "",
        "tell_truthful_about_question": "",
        "b_left_before_a": "",
        "a_left_before_put": "",
        "b_put_or_moved": "",
        "history_mode": header_info.get('history_mode', 'none'),
        "trial": trial_in_block,
        "pause_mode": "none",
        "rep": rep,
        "seed": None,
        "free_response": header_info.get('free_response', True),
        "lied_to_opponent_answerer": "",
        "situation_event_count": None,
        "ect_certainty": None,
        "ect_accuracy": None,
        "ect_total": None
    }

    return record


def recover_log_data(log_file: str) -> Tuple[List[Dict], Dict]:
    """Parse a log file and recover turn records.

    Returns:
        Tuple of (records, header_info)
    """

    with open(log_file, 'r') as f:
        content = f.read()

    # Parse header info
    header_info = {}
    history_match = re.search(r'History mode:\s*(\w+)', content)
    if history_match:
        header_info['history_mode'] = history_match.group(1)

    free_resp_match = re.search(r'Free response:\s*(\w+)', content)
    if free_resp_match:
        header_info['free_response'] = free_resp_match.group(1).lower() == 'true'

    # Parse reps and scenarios to calculate scenarios_per_rep
    # Format: "Reps: 10" and "Running reps 1 to 10 (scenarios 0 to 1039)"
    reps_match = re.search(r'Reps:\s*(\d+)', content)
    scenarios_match = re.search(r'scenarios \d+ to (\d+)', content)
    if reps_match and scenarios_match:
        total_reps = int(reps_match.group(1))
        max_scenario = int(scenarios_match.group(1))
        total_scenarios = max_scenario + 1  # 0-indexed
        header_info['scenarios_per_rep'] = total_scenarios // total_reps
        header_info['total_reps'] = total_reps
    else:
        header_info['scenarios_per_rep'] = 104  # Default
        header_info['total_reps'] = None

    # Split into trial blocks
    blocks = re.split(r'(?=--- Running Trial \d+/\d+)', content)

    records = []
    for i, block in enumerate(blocks):
        if '--- Running Trial' not in block:
            continue

        # Check if this trial completed (has GAME OVER)
        if 'GAME OVER' not in block:
            print(f"Skipping incomplete trial in block {i}")
            continue

        record = parse_trial_block(block, i, header_info)
        if record:
            records.append(record)

    return records, header_info


def filter_complete_reps(records: List[Dict], scenarios_per_rep: int) -> Tuple[List[Dict], Dict[int, int]]:
    """Filter to only include records from complete reps.

    Returns:
        Tuple of (filtered_records, rep_counts) where rep_counts maps rep -> count
    """
    # Group records by rep
    by_rep = defaultdict(list)
    for r in records:
        by_rep[r['rep']].append(r)

    rep_counts = {rep: len(recs) for rep, recs in by_rep.items()}

    # Filter to complete reps only
    filtered = []
    for rep, recs in sorted(by_rep.items()):
        if len(recs) == scenarios_per_rep:
            filtered.extend(recs)

    return filtered, rep_counts


def main():
    parser = argparse.ArgumentParser(
        description='Recover turn record data from a crashed LLM test run log file.'
    )
    parser.add_argument('log_file', help='Path to the log file')
    parser.add_argument(
        '--allow-incomplete',
        action='store_true',
        default=False,
        help='Include data from incomplete reps (default: only complete reps)'
    )
    args = parser.parse_args()

    log_file = args.log_file
    print(f"Recovering data from: {log_file}")

    records, header_info = recover_log_data(log_file)
    print(f"Recovered {len(records)} turn records")

    scenarios_per_rep = header_info.get('scenarios_per_rep', 104)
    print(f"Scenarios per rep: {scenarios_per_rep}")

    # Filter by complete reps unless --allow-incomplete
    if not args.allow_incomplete:
        filtered_records, rep_counts = filter_complete_reps(records, scenarios_per_rep)

        # Report rep status
        complete_reps = [r for r, c in rep_counts.items() if c == scenarios_per_rep]
        incomplete_reps = [(r, c) for r, c in rep_counts.items() if c != scenarios_per_rep]

        if complete_reps:
            print(f"Complete reps: {sorted(complete_reps)}")
        if incomplete_reps:
            print(f"Incomplete reps (excluded): {[(r, f'{c}/{scenarios_per_rep}') for r, c in sorted(incomplete_reps)]}")

        records = filtered_records
        print(f"Using {len(records)} records from {len(complete_reps)} complete rep(s)")

    if not records:
        print("No complete reps found. Use --allow-incomplete to include partial data.")
        sys.exit(1)

    # Generate output filename
    output_file = log_file.replace('.log', '_game_data.json')

    # Write output in the wrapper format
    output = {
        "game_setup": None,  # Not available in log
        "turn_records": records
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved to: {output_file}")

    # Print summary stats
    optimal_count = sum(1 for r in records if r['was_optimal'])
    correct_count = sum(1 for r in records if r['answer_correct'])
    print(f"Optimal actions: {optimal_count}/{len(records)} ({100*optimal_count/len(records):.1f}%)")
    print(f"Correct answers: {correct_count}/{len(records)} ({100*correct_count/len(records):.1f}%)")


if __name__ == '__main__':
    main()
