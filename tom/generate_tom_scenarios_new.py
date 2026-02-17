import random
import copy
from collections import defaultdict
from tom_helpers import (
    Scenario, Event, EpistemicState, CharacterType,
    save_scenarios, load_scenarios, SpecTuple, read_specs_from_csv
)
from typing import List, Optional, Tuple, Set, Dict
from validate_scenarios import validate_scenario
from dataclasses import dataclass

ITEMS_GEN = ['apple', 'ball', 'banana', 'brick', 'stapler', 'orange']
CONTAINERS_GEN = ['bag', 'box', 'basket']

# Configurable filler event counts for Extra=0A/0B (no ECT impact)
# See EXTRA_MAPPING.md for full documentation
EXTRA_0A_FILLER = 0   # Minimal load (no filler)
EXTRA_0B_FILLER = 3   # Higher load (tweak as needed)

# Adaptive filler for Extra=1A (balancing with 1B)
ADAPTIVE_FILLER_ENABLED = True  # Set to False to disable

def _map_chartypes_to_names(chartypes: List[CharacterType]) -> List[str]:
    chars = []
    for ct in chartypes:
        if ct == CharacterType.LIVE_PLAYER: chars.append('A')
        elif ct in [CharacterType.HONEST_TEAMMATE, CharacterType.DISHONEST_TEAMMATE]: chars.append('B')
        elif ct == CharacterType.NEUTRAL: chars.append('N')
        elif ct == CharacterType.HONEST_OPPONENT and 'C' not in chars: chars.append('C')
        elif ct == CharacterType.HONEST_OPPONENT and 'C' in chars: chars.append('D')
        elif ct == CharacterType.DISHONEST_OPPONENT and CharacterType.HONEST_OPPONENT not in chartypes and 'C' not in chars: chars.append('C')
        elif ct == CharacterType.DISHONEST_OPPONENT: chars.append('D')
    return chars

def _teammate_of(name: str) -> str:
    return {'A': 'B', 'B': 'A', 'C': 'D', 'D': 'C'}[name]

def _opponent_of(name: str, rng: random.Random) -> str:
    #Get a random opponent of the given character.
    if name in ['A', 'B']:
        return rng.choice(['C', 'D'])  # Blue team → pick a red opponent
    else:
        return rng.choice(['A', 'B'])  # Red team → pick a blue opponent

def _other_container(c: str) -> str:
    """Return ONE other container (for backward compatibility with 2-container logic)."""
    others = [x for x in CONTAINERS_GEN if x != c]
    return others[0]  # Always return the first non-target container


def _non_target_containers(target: str) -> List[str]:
    """Return list of all containers that are NOT the target."""
    return [c for c in CONTAINERS_GEN if c != target]

def _pick_other_item(rng: random.Random, exclude: str) -> str:
    return rng.choice([x for x in ITEMS_GEN if x != exclude])

def _choose_different_actor(present: set, last_actor: Optional[str], rng: random.Random) -> str:
    """Choose actor, preferring someone different from last_actor if possible."""
    present_list = sorted(present)
    if len(present_list) == 1 or last_actor is None:
        return rng.choice(present_list)
    # Exclude last actor, pick from remaining
    candidates = [c for c in present_list if c != last_actor]
    return rng.choice(candidates) if candidates else rng.choice(present_list)


def _get_containers_used_after(scenario: 'Scenario', insert_idx: int) -> Set[str]:
    """Return set of containers used by events at or after insert_idx.

    Used to determine which containers are safe to use for filler/extra events
    without conflicting with later base scenario events.
    """
    containers = set()
    for event in scenario.events[insert_idx:]:
        if event.event_type == 'put':
            containers.add(event.container)
        elif event.event_type == 'move':
            containers.add(event.to_container)
            containers.add(event.from_container)
    return containers


@dataclass
class ScenarioStateAtLeave:
    """State snapshot when a character leaves the room."""
    contents: Dict[str, Optional[str]]
    present: Set[str]  # Characters present AFTER the leave (leaver removed)
    leave_idx: Optional[int]  # Index of the leave event, or None if character never left


def _find_leave_and_track_state(scenario: 'Scenario', character: str) -> ScenarioStateAtLeave:
    """Find when a character leaves and track container/presence state at that point.

    Tracks container contents and character presence through events until the
    specified character leaves. Returns the state at the moment of leaving.

    Args:
        scenario: The scenario to analyze
        character: The character whose leave event to find

    Returns:
        ScenarioStateAtLeave with:
        - contents: container contents when character left
        - present: characters present AFTER the leave (leaver removed)
        - leave_idx: index of the leave event, or None if never left
    """
    contents = {c: None for c in CONTAINERS_GEN}
    present = set(scenario.present_initially)

    for idx, event in enumerate(scenario.events):
        if event.event_type == 'put':
            contents[event.container] = event.item
        elif event.event_type == 'move':
            contents[event.to_container] = event.item
            contents[event.from_container] = None
        elif event.event_type == 'leave':
            if event.character == character:
                present.discard(character)
                return ScenarioStateAtLeave(contents, present, idx)
            present.discard(event.character)
        elif event.event_type == 'enter':
            present.add(event.character)

    return ScenarioStateAtLeave(contents, present, None)


def _get_teammates_to_exclude_for_target_moves(scenario: 'Scenario', up_to_idx: int) -> Set[str]:
    """
    Find blue team members who should be excluded from doing moves on the target container.

    Only protects blue team (A/B) beliefs from each other. Red team (C/D) opponents
    can invalidate each other's beliefs.

    Looks at events up to (but not including) up_to_idx to find blue team characters
    who left with beliefs about the target. Returns the other blue team member if
    they should be excluded.
    """
    target = scenario.question_container
    BLUE = {'A', 'B'}

    present = set(scenario.present_initially)
    char_observed_target = {c: False for c in ['A', 'B', 'C', 'D']}

    # Track which blue team members left with beliefs about the target
    blue_departed_with_belief = set()

    for idx, event in enumerate(scenario.events):
        if idx >= up_to_idx:
            break

        if event.event_type == 'put' and event.container == target:
            for c in present:
                char_observed_target[c] = True
        elif event.event_type == 'move':
            if event.from_container == target or event.to_container == target:
                for c in present:
                    char_observed_target[c] = True
        elif event.event_type == 'leave':
            char = event.character
            if char in BLUE and char_observed_target.get(char, False):
                blue_departed_with_belief.add(char)
            present.discard(char)
        elif event.event_type == 'enter':
            present.add(event.character)

    # Return blue team teammates of those who departed with beliefs
    exclude = set()
    for departed in blue_departed_with_belief:
        teammate = _teammate_of(departed)
        exclude.add(teammate)

    return exclude


def _get_unknown_characters_from_spec(spec: dict) -> Tuple[Set[str], bool]:
    """
    Identify characters who must remain UNKNOWN per the spec.

    Returns:
        Tuple of (must_be_unknown, opponent_needs_one_unknown):
        - must_be_unknown: Set of character names (e.g., {'B'}) that MUST be UNKNOWN
        - opponent_needs_one_unknown: True if at least ONE of C/D must be UNKNOWN

    For Teammate=UNKNOWN: B must be UNKNOWN
    For Opponent=UNKNOWN: At least ONE of C/D must be UNKNOWN (not both required)
    """
    from tom_helpers import EpistemicState

    must_be_unknown = set()
    opponent_needs_one_unknown = False

    if spec.get('KS_Teammate') == EpistemicState.UNKNOWN:
        must_be_unknown.add('B')
    if spec.get('KS_Opponent') == EpistemicState.UNKNOWN:
        opponent_needs_one_unknown = True  # Only need ONE of C/D to stay UNKNOWN

    return must_be_unknown, opponent_needs_one_unknown


def _ensure_unknown_chars_absent(
    scenario: 'Scenario',
    spec: dict,
    insert_pos: int,
    rng: random.Random,
    exclude_chars: Set[str] = None
) -> Tuple[int, List[str]]:
    """
    Insert leave events for chars who must remain UNKNOWN but are present.

    This function ensures that characters who should remain UNKNOWN per the spec
    are not present when extra events are inserted on the target container.

    For Opponent=UNKNOWN: Only forces ONE opponent to leave if BOTH are present
    and would witness the extra events. Uses minimum necessary removals.

    Args:
        scenario: The scenario being modified
        spec: The scenario specification with KS_Teammate, KS_Opponent
        insert_pos: Position where extra events will be inserted
        rng: Random number generator for choosing which opponent to remove
        exclude_chars: Characters to NOT force to leave (e.g., the answerer whose
            leave is already handled by the calling function)

    Returns:
        Tuple of (new_insert_pos, list_of_chars_who_left)
    """
    if exclude_chars is None:
        exclude_chars = set()
    from tom_helpers import EpistemicState

    must_be_unknown, opponent_needs_one_unknown = _get_unknown_characters_from_spec(spec)

    # Track who is present at insert_pos and who has observed the target
    present = set(scenario.present_initially)
    observed_target = {c: False for c in ['A', 'B', 'C', 'D']}
    target = scenario.question_container

    for idx, event in enumerate(scenario.events):
        if idx >= insert_pos:
            break
        if event.event_type == 'put' and event.container == target:
            for c in present:
                observed_target[c] = True
        elif event.event_type == 'move':
            if event.from_container == target or event.to_container == target:
                for c in present:
                    observed_target[c] = True
        elif event.event_type == 'leave':
            present.discard(event.character)
        elif event.event_type == 'enter':
            present.add(event.character)

    # Determine which characters need to leave
    chars_to_leave = []

    # Check if any characters already have a leave event coming up (before any enter for them)
    # If so, they're already leaving and we don't need to insert another leave
    # We scan from insert_pos onward to find leave events for characters who are present
    chars_already_leaving = set()
    for idx in range(insert_pos, len(scenario.events)):
        event = scenario.events[idx]
        if event.event_type == 'leave' and event.character in present:
            # This character is present and will leave - no need to insert another leave
            chars_already_leaving.add(event.character)
        elif event.event_type == 'enter' and event.character in chars_already_leaving:
            # They re-enter after their leave - they're still "already leaving" but will be back
            # Don't remove from chars_already_leaving since we found their leave event
            pass

    # For must_be_unknown (e.g., B for Teammate=UNKNOWN): must leave if present
    # Skip characters in exclude_chars (their departure is handled elsewhere)
    # Skip characters who are already leaving at this position
    for char in must_be_unknown:
        if char in present and char not in exclude_chars and char not in chars_already_leaving:
            chars_to_leave.append(char)

    # For opponent_needs_one_unknown: check if at least one is already UNKNOWN
    if opponent_needs_one_unknown:
        # A character is UNKNOWN if they're not present OR never observed target
        # Also count as "would be unknown" if they're in exclude_chars (will leave separately)
        # Also count as "would be unknown" if they're already leaving at this position
        c_would_be_unknown = ('C' not in present or not observed_target.get('C', False) or
                              'C' in exclude_chars or 'C' in chars_already_leaving)
        d_would_be_unknown = ('D' not in present or not observed_target.get('D', False) or
                              'D' in exclude_chars or 'D' in chars_already_leaving)

        # Only need to force one to leave if BOTH would become non-UNKNOWN
        if not c_would_be_unknown and not d_would_be_unknown:
            # Both C and D are present, observed target, and not being excluded/already leaving
            # Force ONE to leave (minimum necessary)
            candidates = [c for c in ['C', 'D'] if c not in exclude_chars and c not in chars_already_leaving]
            if candidates:
                opponent_to_remove = rng.choice(candidates)
                if opponent_to_remove not in chars_to_leave:
                    chars_to_leave.append(opponent_to_remove)

    # Insert leave events
    events_inserted = 0
    for char in sorted(chars_to_leave):
        scenario.events.insert(insert_pos + events_inserted, Event('leave', char))
        events_inserted += 1

    return insert_pos + events_inserted, chars_to_leave


@dataclass
class Scenario_Builder:
    rng: random.Random
    queried_container: str            
    queried_item: str    
    available: Set[str]    # who is allowed to be in the room initially

    def __post_init__(self):
        self.present: Set[str] = set(self.available)
        self.events: List[Event] = []
        self.contents = {c: None for c in CONTAINERS_GEN} 
        self.used: Set[str] = set()  # anyone who acts or leaves
        self.exclude: Set[str] = set()      # who must leave (random: may or may not see target)
        self.exclude_true: Set[str] = set() # who must leave believing something that matches the end queried_item/container state
        self.exclude_false: Set[str] = set() # who must leave believing something that does NOT match the end state
        self.exclude_unknown: Set[str] = set()  # who must leave BEFORE seeing target (UNKNOWN state)
        self.exclude_flexible_unknown: Set[str] = set()  # who must leave before end, timing flexible (UNKNOWN from A's perspective)
        self.exclude_after_self: Set[str] = set()  # who must leave after A (self), before final put (Case 3)
        self.include: Set[str] = set()      # who must be present at end
        self.present_initially: Set[str] = set()  # who must be present initially
        self.must_leave_together: Tuple[Optional[str], Optional[str]] = (None, None)  # (char1, char2) must be in same group
        self.leave_before_target_reenter: Set[str] = set()  # who must leave before target put and re-enter after (for filler)
        # Character references (set in plan_availability)
        self.self_char: Optional[str] = None
        self.opponent1: Optional[str] = None
        self.opponent2: Optional[str] = None

    def rand_actor(self, exclude: Optional[Set[str]] = None) -> str:
        pool = [p for p in sorted(self.present) if not exclude or p not in exclude]
        return self.rng.choice(pool)

    def leave(self, who: str):
        # Schedule a leave only if they’re currently present
        if who in self.present:
            self.events.append(Event('leave', who))
            self.present.discard(who)
            self.used.add(who)

    def move_out_if_needed(self, container: str, who: str):
        existing = self.contents[container]
        if existing is None:
            return
        to_cont = _other_container(container)
        # In this generator we keep 'to_cont' empty until needed
        if self.contents[to_cont] is not None:
            # Safety: shouldn't happen in our single-flip patterns
            return
        self.events.append(Event('move', who, from_container=container, to_container=to_cont, item=existing))
        self.used.add(who)
        self.contents[container] = None
        self.contents[to_cont] = existing

    def put(self, container: str, item: str, exclude: Optional[Set[str]] = None):
        # Ensure we never narrate simultaneous items in a container
        who = self.rand_actor(exclude)
        if self.contents[container] is not None and self.contents[container] != item:
            self.move_out_if_needed(container, self.rand_actor(exclude))
        self.contents[container] = item
        self.events.append(Event('put', who, container=container, item=item))
        self.used.add(who)

    def plan_availability(self, spec: dict, answerer: str):

        actor, teammate, opponent1, opponent2 = _map_to_char_names(spec['Actor'])
        # Store character references for use in build_scenario
        self.self_char = actor
        self.opponent1 = opponent1
        self.opponent2 = opponent2
        if spec['KS_Self'] == EpistemicState.BELIEVES_X:
            self.exclude.add(actor) 
            if spec['KS_Teammate'] == EpistemicState.KNOWS_TRUTH and spec['KS_Opponent'] == EpistemicState.KNOWS_TRUTH:
                self.include.add(teammate)
                if spec.get('Answerer') == 'Opponent':
                    # ensure the chosen opponent (the one who must answer) is present
                    self.include.add(answerer)
                else:
                    self.include.add(self.rng.choice([opponent1, opponent2]))

            elif spec['KS_Teammate'] == EpistemicState.KNOWS_TRUTH and spec['KS_Opponent'] == EpistemicState.UNKNOWN:
                self.include.add(teammate)
                if spec['Answerer'] == 'Self':
                    # One opponent leaves (UNKNOWN), the other stays for final put
                    # (B is excluded from target moves because A left with belief)
                    excluded_opp = self.rng.choice([opponent1, opponent2])
                    self.exclude_flexible_unknown.add(excluded_opp)
                    self.leave_before_target_reenter.add(excluded_opp)  # For filler availability
                    staying_opp = opponent1 if excluded_opp == opponent2 else opponent2
                    # staying_opp leaves after final put - still UNKNOWN from A's view
                    # (A left earlier, doesn't know what staying_opp saw)
                    self.exclude_true.add(staying_opp)
                elif spec.get('Answerer') == 'Opponent':
                    # ensure the chosen opponent (the one who must answer) is excluded (UNKNOWN)
                    self.exclude_flexible_unknown.add(answerer)
                    self.leave_before_target_reenter.add(answerer)  # For filler availability
                    # Include non-answerer opponent so someone can do Extra=1B events
                    # (B is excluded from target moves because A left with belief)
                    non_answerer_opp = opponent1 if answerer == opponent2 else opponent2
                    self.include.add(non_answerer_opp)
                else:
                    # One opponent leaves (UNKNOWN), the other stays for Extra=1B events
                    # (B is excluded from target moves because A left with belief)
                    excluded_opp = self.rng.choice([opponent1, opponent2])
                    self.exclude_flexible_unknown.add(excluded_opp)
                    self.leave_before_target_reenter.add(excluded_opp)  # For filler availability
                    staying_opp = opponent1 if excluded_opp == opponent2 else opponent2
                    self.include.add(staying_opp)

            elif spec['KS_Teammate'] == EpistemicState.UNKNOWN and spec['KS_Opponent'] == EpistemicState.KNOWS_TRUTH:
                # Teammate can leave at flexible times (before puts OR after intermediate) - all result in UNKNOWN from A's perspective
                self.exclude_flexible_unknown.add(teammate)
                if spec['Answerer'] == 'Self':
                    self.include.add(opponent1)
                    self.include.add(opponent2)
                elif spec.get('Answerer') == 'Opponent':
                    # ensure the chosen opponent (the one who must answer) is present
                    self.include.add(answerer)
                else:
                    self.include.add(self.rng.choice([opponent1, opponent2])) 

            elif spec['KS_Teammate'] == EpistemicState.UNKNOWN and spec['KS_Opponent'] == EpistemicState.UNKNOWN:
                # Teammate can leave at flexible times (before puts OR after intermediate) - all result in UNKNOWN from A's perspective
                self.exclude_flexible_unknown.add(teammate)
                self.leave_before_target_reenter.add(teammate)  # For filler availability
                # Ensure one opponent stays, one leaves (UNKNOWN)
                if spec.get('Answerer') == 'Opponent':
                    leave_opponent = answerer
                    stay_opponent = opponent1 if leave_opponent == opponent2 else opponent2
                else:
                    stay_opponent = self.rng.choice([opponent1, opponent2])
                    leave_opponent = opponent2 if stay_opponent == opponent1 else opponent1
                self.include.add(stay_opponent)  # This opponent must stay until end
                self.exclude_flexible_unknown.add(leave_opponent)  # This opponent can leave at flexible times (UNKNOWN)
                self.leave_before_target_reenter.add(leave_opponent)  # For filler availability

        else: # spec['KS_Self'] == EpistemicState.KNOWS_X:
            self.include.add(actor) 
            if spec['KS_Teammate'] == EpistemicState.BELIEVES_TRUTH and spec['KS_Opponent'] == EpistemicState.BELIEVES_TRUTH:
                self.exclude_true.add(teammate)
                if spec.get('Answerer') == 'Opponent':
                    self.exclude_true.add(answerer)
                    # Include non-answerer opponent so someone can do Extra=1B events
                    # (A is excluded from target moves because B left with belief)
                    non_answerer_opp = opponent1 if answerer == opponent2 else opponent2
                    self.include.add(non_answerer_opp)
                else:
                    # One opponent leaves with true belief, the other stays
                    # Need to include the staying opponent for Extra=1B events
                    # (A is excluded from target moves because B left with belief)
                    excluded_opp = self.rng.choice([opponent1, opponent2])
                    self.exclude_true.add(excluded_opp)
                    staying_opp = opponent1 if excluded_opp == opponent2 else opponent2
                    self.include.add(staying_opp)
            elif spec['KS_Teammate'] == EpistemicState.BELIEVES_TRUTH and spec['KS_Opponent'] == EpistemicState.BELIEVES_FALSE:
                self.exclude_true.add(teammate)
                if spec['Answerer'] == 'Opponent':
                    self.exclude_false.add(answerer)
                    # Include non-answerer opponent so someone can do Extra=1B events
                    # (A is excluded from target moves because B left with belief)
                    non_answerer_opp = opponent1 if answerer == opponent2 else opponent2
                    self.include.add(non_answerer_opp)
                else:
                    # One opponent leaves with false belief, the other stays
                    # Need to include the staying opponent for Extra=1B events
                    # (A is excluded from target moves because B left with belief)
                    excluded_opp = self.rng.choice([opponent1, opponent2])
                    self.exclude_false.add(excluded_opp)
                    staying_opp = opponent1 if excluded_opp == opponent2 else opponent2
                    self.include.add(staying_opp)
            elif spec['KS_Teammate'] == EpistemicState.BELIEVES_TRUTH and spec['KS_Opponent'] == EpistemicState.KNOWS_TRUTH:
                self.exclude_true.add(teammate)
                if spec.get('Answerer') == 'Opponent':
                    self.include.add(answerer)
                else:
                    self.include.add(self.rng.choice([opponent1, opponent2]))
            elif spec['KS_Teammate'] == EpistemicState.BELIEVES_FALSE and spec['KS_Opponent'] == EpistemicState.BELIEVES_TRUTH:
                self.exclude_false.add(teammate)
                if spec['Answerer'] == 'Teammate':
                    # Both opponents leave with true belief, but need one to stay for Extra=1B events
                    # (A is excluded from target moves because B left with belief)
                    # One leaves with true belief, the other stays (will also get true belief before leaving)
                    excluded_opp = self.rng.choice([opponent1, opponent2])
                    self.exclude_true.add(excluded_opp)
                    staying_opp = opponent1 if excluded_opp == opponent2 else opponent2
                    self.include.add(staying_opp)
                elif spec.get('Answerer') == 'Opponent':
                    self.exclude_true.add(answerer)
                    # Include non-answerer opponent so someone can do Extra=1B events
                    # (A is excluded from target moves because B left with belief)
                    non_answerer_opp = opponent1 if answerer == opponent2 else opponent2
                    self.include.add(non_answerer_opp)
                else:
                    # One opponent leaves with true belief, the other stays
                    # Need to include the staying opponent for Extra=1B events
                    # (A is excluded from target moves because B left with belief)
                    excluded_opp = self.rng.choice([opponent1, opponent2])
                    self.exclude_true.add(excluded_opp)
                    staying_opp = opponent1 if excluded_opp == opponent2 else opponent2
                    self.include.add(staying_opp)
            elif spec['KS_Teammate'] == EpistemicState.BELIEVES_FALSE and spec['KS_Opponent'] == EpistemicState.BELIEVES_FALSE:
                self.exclude_false.add(teammate)
                if spec['Answerer'] == 'Opponent':
                    self.exclude_false.add(answerer)
                    # Include non-answerer opponent so someone can do Extra=1B events
                    # (A is excluded from target moves because B left with belief)
                    non_answerer_opp = opponent1 if answerer == opponent2 else opponent2
                    self.include.add(non_answerer_opp)
                else:
                    # One opponent leaves with false belief, the other stays
                    # Need to include the staying opponent for Extra=1B events
                    # (A is excluded from target moves because B left with belief)
                    excluded_opp = self.rng.choice([opponent1, opponent2])
                    self.exclude_false.add(excluded_opp)
                    staying_opp = opponent1 if excluded_opp == opponent2 else opponent2
                    self.include.add(staying_opp)
            elif spec['KS_Teammate'] == EpistemicState.BELIEVES_FALSE and spec['KS_Opponent'] == EpistemicState.KNOWS_TRUTH:
                self.exclude_false.add(teammate)
                if spec['Answerer'] == 'Teammate':
                    self.include.add(opponent1)
                    self.include.add(opponent2)
                elif spec.get('Answerer') == 'Opponent':
                    self.include.add(answerer)
                else:
                    self.include.add(self.rng.choice([opponent1, opponent2]))
            elif spec['KS_Teammate'] == EpistemicState.KNOWS_TRUTH and spec['KS_Opponent'] == EpistemicState.BELIEVES_TRUTH:
                self.include.add(teammate)
                if spec['Answerer'] == 'Opponent':
                    self.exclude_true.add(answerer)
                else:
                    self.exclude_true.add(self.rng.choice([opponent1, opponent2]))
            elif spec['KS_Teammate'] == EpistemicState.KNOWS_TRUTH and spec['KS_Opponent'] == EpistemicState.BELIEVES_FALSE:
                self.include.add(teammate)
                if spec['Answerer'] == 'Teammate' or spec['Answerer'] == 'Opponent':
                    self.exclude_false.add(opponent1)
                    self.exclude_false.add(opponent2)
                else:
                    self.exclude_false.add(self.rng.choice([opponent1, opponent2]))
            elif spec['KS_Teammate'] == EpistemicState.KNOWS_TRUTH and spec['KS_Opponent'] == EpistemicState.KNOWS_TRUTH:
                self.include.add(teammate)
                if spec.get('Answerer') == 'Opponent':
                    self.include.add(answerer)
                else:
                    self.include.add(self.rng.choice([opponent1, opponent2])) 

        self.present_initially = self.exclude | self.exclude_true | self.exclude_false | self.exclude_unknown | self.exclude_flexible_unknown | self.exclude_after_self | self.include # who must be present initially

    def build_scenario(self, answerer: str):
        #randomly add anyone who is in available but not in present_initially to present_initially
        leave_immediately_group = set()
        for who in sorted(self.available):  # sorted for deterministic ordering
            if who not in self.present_initially:
                if self.rng.random() < 0.5:
                    self.present_initially.add(who)
        for who in sorted(self.exclude):  # sorted for deterministic ordering
            r = self.rng.random()
            if r <= 0.33333:
                self.exclude_false.add(who)
            elif r <= 0.66666:
                self.exclude_true.add(who)
            else:
                leave_immediately_group.add(who)

        # Characters who must be UNKNOWN always go to leave_immediately_group
        leave_immediately_group |= self.exclude_unknown

        # Characters with flexible UNKNOWN timing - 3 valid cases
        # All result in UNKNOWN from A's perspective since A doesn't know the final reality
        BLUE = {'A', 'B'}
        for who in sorted(self.exclude_flexible_unknown):
            # Characters in leave_before_target_reenter are forced to Case 1 (for filler availability)
            if who in self.leave_before_target_reenter:
                leave_immediately_group.add(who)
                continue
            r = self.rng.random()
            if r < 0.33:
                # Case 1: Leave before any puts
                leave_immediately_group.add(who)
            elif r < 0.67:
                # Case 2: Leave after intermediate put, before A leaves
                self.exclude_false.add(who)
            else:
                # Case 3: Leave after A
                if who in BLUE:
                    # Teammate: leave before final put (safe for belief integrity)
                    self.exclude_after_self.add(who)
                else:
                    # Opponent: leave after final put (can do final put without violation)
                    # From A's view, opponent is still UNKNOWN (A doesn't know what they saw)
                    self.exclude_true.add(who)

        # Self with "Believes X" must see a put before leaving to form their belief
        # Move them out of leave_immediately_group to exclude_false
        if self.self_char in leave_immediately_group and self.self_char in self.exclude:
            leave_immediately_group.discard(self.self_char)
            self.exclude_false.add(self.self_char)

        # Enforce constraint: ensure both characters are in the same group
        if self.must_leave_together[0] is not None:
            char1, char2 = self.must_leave_together
            # Find which group char1 is in and move char2 to match
            if char1 in self.exclude_false and char2 not in self.exclude_false:
                self.exclude_true.discard(char2)
                leave_immediately_group.discard(char2)
                self.exclude_false.add(char2)
            elif char1 in self.exclude_true and char2 not in self.exclude_true:
                self.exclude_false.discard(char2)
                leave_immediately_group.discard(char2)
                self.exclude_true.add(char2)
            elif char1 in leave_immediately_group and char2 not in leave_immediately_group:
                self.exclude_false.discard(char2)
                self.exclude_true.discard(char2)
                leave_immediately_group.add(char2)

        # Re-check: if Self was moved back to leave_immediately_group by must_leave_together,
        # move BOTH Self and their partner to exclude_false (Self must see a put)
        if self.self_char in leave_immediately_group and self.self_char in self.exclude:
            leave_immediately_group.discard(self.self_char)
            self.exclude_false.add(self.self_char)
            # Also move partner if they're in must_leave_together
            if self.must_leave_together[0] is not None:
                partner = self.must_leave_together[0] if self.must_leave_together[1] == self.self_char else (
                    self.must_leave_together[1] if self.must_leave_together[0] == self.self_char else None
                )
                if partner and partner in leave_immediately_group:
                    leave_immediately_group.discard(partner)
                    self.exclude_false.add(partner)

        # Ensure at least one opponent is present when Self leaves
        # (Constraint: at least one opponent must leave at same phase as Self or later)
        def get_phase(char):
            if char in leave_immediately_group:
                return 0
            elif char in self.exclude_false:
                return 1
            elif char in self.exclude_true:
                return 2
            else:
                return 3  # Not leaving (stays until end)

        def set_phase(char, phase):
            leave_immediately_group.discard(char)
            self.exclude_false.discard(char)
            self.exclude_true.discard(char)
            if phase == 0:
                leave_immediately_group.add(char)
            elif phase == 1:
                self.exclude_false.add(char)
            elif phase == 2:
                self.exclude_true.add(char)

        if self.self_char is not None and self.opponent1 is not None and self.opponent2 is not None:
            self_phase = get_phase(self.self_char)
            opp1_phase = get_phase(self.opponent1)
            opp2_phase = get_phase(self.opponent2)

            # If both opponents leave before Self, move one to Self's phase
            # Only apply when Self is actually leaving (phase < 3)
            if self_phase < 3 and opp1_phase < self_phase and opp2_phase < self_phase:
                opponent_to_move = self.rng.choice([self.opponent1, self.opponent2])
                set_phase(opponent_to_move, self_phase)

        # Helper to order characters so Self leaves first within a phase
        # (ensures opponents are still present when Self leaves)
        def order_self_first(chars):
            chars_list = sorted(chars)  # sorted for deterministic ordering
            if self.self_char in chars_list:
                chars_list.remove(self.self_char)
                return [self.self_char] + self.rng.sample(chars_list, len(chars_list))
            else:
                return self.rng.sample(chars_list, len(chars_list))

        # Now execute the leave-immediately actions
        for who in order_self_first(leave_immediately_group):
            self.leave(who)

        if len(self.exclude_false) > 0:
            old_item = _pick_other_item(self.rng, self.queried_item)
            self.put(self.queried_container, old_item, exclude=None)
            # Re-enter characters who left before target put (for filler availability)
            # They didn't observe the target put, so they're available for leave/enter filler
            for who in sorted(self.leave_before_target_reenter):
                if who not in self.present:  # Only re-enter if they left
                    self.events.append(Event('enter', who))
                    self.present.add(who)
                    self.used.add(who)
            for who in order_self_first(self.exclude_false):
                self.leave(who)

        # Case 3: Characters who leave after A (self), before final put
        # A has already left (in exclude_false), so we use sorted() not order_self_first()
        for who in sorted(self.exclude_after_self):
            self.leave(who)

        # Exclude blue team members from doing the put if their teammate left with false belief.
        # Only blue team (A/B) is protected - opponents (C/D) can invalidate each other.
        BLUE = {'A', 'B'}
        exclude_set = set()
        for departed in self.exclude_false:
            if departed in BLUE:  # Only protect blue team
                teammate = _teammate_of(departed)
                if teammate in self.present:
                    exclude_set.add(teammate)

        # Check if we can still do the put with these exclusions
        available_for_put = [p for p in sorted(self.present) if p not in exclude_set]
        if len(available_for_put) == 0:
            # Can't exclude all teammates, reduce exclusion set
            # Prioritize excluding answerer's teammate if answerer left with false belief
            if answerer in self.exclude_false:
                potential_exclude = _teammate_of(answerer)
                if potential_exclude in self.present:
                    remaining = [p for p in sorted(self.present) if p != potential_exclude]
                    if len(remaining) > 0:
                        exclude_set = {potential_exclude}
                    else:
                        exclude_set = set()  # No choice, can't exclude anyone
                else:
                    exclude_set = set()
            else:
                exclude_set = set()

        self.put(self.queried_container, self.queried_item, exclude=exclude_set if exclude_set else None)
        for who in order_self_first(self.exclude_true):
            self.leave(who)

        self.present_initially = self.present_initially | self.used

def _map_to_char_names(actor_ct: CharacterType) -> Tuple[str, str, str, str]:
    if actor_ct == CharacterType.LIVE_PLAYER:
        return 'A', 'B', 'C', 'D'
    elif actor_ct in [CharacterType.HONEST_TEAMMATE, CharacterType.DISHONEST_TEAMMATE]:
        return 'B', 'A', 'C', 'D'
    elif actor_ct == CharacterType.HONEST_OPPONENT:
        return 'C', 'D', 'A', 'B'
    elif actor_ct == CharacterType.DISHONEST_OPPONENT:
        return 'D', 'C', 'A', 'B'
    else:
        raise ValueError(f"Unknown actor character type: {actor_ct}")


def _count_visible_events(scenario: 'Scenario') -> int:
    """Count events visible from Self's (A's) perspective.

    Leave/enter events are always visible. Put/move/remove events are only
    visible while A is present in the room (perspective_present=True in
    get_description_for).
    """
    present = set(scenario.present_initially)
    count = 0
    for event in scenario.events:
        if event.event_type in ('leave', 'enter'):
            count += 1  # always visible
            if event.event_type == 'leave':
                present.discard(event.character)
            else:
                present.add(event.character)
        elif event.event_type in ('put', 'move', 'remove'):
            if 'A' in present:
                count += 1
    return count


def count_epistemic_category_transitions(scenario: 'Scenario') -> Dict[str, int]:
    """Count epistemic category transitions (ECTs) for the target (queried) container.

    ECTs measure how many times characters' epistemic states change along two axes:
    - Certainty: knowledge (directly observed) vs belief (stale/uncertain)
    - Accuracy: true belief (matches reality) vs false belief (doesn't match)

    Four transition types:
    #1 (Certainty): Character leaves after observing target → knowledge → belief
    #2 (Certainty): Character with belief sees event on target → belief → knowledge
    #3 (Accuracy): Target changes while character absent with true belief → true → false
    #4 (Accuracy): Target changes while character present after #3 → false → true

    Initial learning (first observation of target) is NOT counted.

    IMPORTANT: ECTs are counted from A's perspective. After A leaves, no more ECTs
    are counted because A cannot observe any further transitions.

    Returns: {'certainty': int, 'accuracy': int, 'total': int}
    """
    target = scenario.question_container
    present = set(scenario.present_initially)

    # Collect all characters
    all_chars = set(scenario.present_initially)
    for event in scenario.events:
        all_chars.add(event.character)

    _UNKNOWN = '__unknown__'

    # Per-character state for the target container
    has_observed = {c: False for c in all_chars}
    has_knowledge = {c: False for c in all_chars}
    belief_content: Dict[str, object] = {c: _UNKNOWN for c in all_chars}
    has_false_belief = {c: False for c in all_chars}

    actual = None  # Current target container contents (None = empty)

    certainty_count = 0
    accuracy_count = 0
    a_left = False  # Track when A leaves - no ECTs counted after this

    for event in scenario.events:
        if event.event_type == 'leave':
            char = event.character
            # ECT #1: knowledge → belief when leaving after seeing target
            # Leave events are visible to A, so this counts even after A leaves
            if has_observed[char] and has_knowledge[char]:
                certainty_count += 1  # #1: knowledge → belief
                has_knowledge[char] = False

            # Track when A leaves (for #2, #3, #4 which require A to see put/move)
            if char == 'A':
                a_left = True

            present.discard(char)

        elif event.event_type == 'enter':
            char = event.character
            a_is_outside = 'A' not in present  # Check BEFORE adding char
            present.add(char)

            # Reset a_left when A re-enters - A can observe ECTs again
            if char == 'A':
                a_left = False

            # ECT #2 on enter only when:
            # - A is outside (can't observe what they actually see inside)
            # - The entering character is not A (A knows their own observations)
            # - Target has contents (something to potentially observe)
            # When A is inside, containers are opaque - must witness put/move for knowledge.
            if a_is_outside and char != 'A' and actual is not None:
                had_belief = has_observed[char] and not has_knowledge[char]
                has_observed[char] = True
                has_knowledge[char] = True
                belief_content[char] = actual
                has_false_belief[char] = False
                if had_belief:
                    certainty_count += 1

        else:
            # Determine if this event affects the target container
            affects_target = False
            old_actual = actual

            if event.event_type == 'put' and event.container == target:
                actual = event.item
                affects_target = True
            elif event.event_type == 'move':
                if event.from_container == target:
                    actual = None
                    affects_target = True
                elif event.to_container == target:
                    actual = event.item
                    affects_target = True
            elif event.event_type == 'remove' and event.container == target:
                actual = None
                affects_target = True

            if affects_target and old_actual != actual:
                # Target container contents changed

                # If A has left, still update state tracking but don't count ECTs
                if a_left:
                    for char in list(present):
                        has_observed[char] = True
                        has_knowledge[char] = True
                        belief_content[char] = actual
                        has_false_belief[char] = False
                    for char in all_chars:
                        if char not in present and belief_content[char] != _UNKNOWN:
                            if belief_content[char] != actual:
                                has_false_belief[char] = True
                    continue

                # Present characters see the change
                for char in list(present):
                    was_observed = has_observed[char]
                    was_knowledge = has_knowledge[char]
                    was_false = has_false_belief[char]

                    # Update state: character sees the change
                    has_observed[char] = True
                    has_knowledge[char] = True
                    belief_content[char] = actual
                    has_false_belief[char] = False

                    # #2: belief → knowledge (previously observed but lost knowledge)
                    if was_observed and not was_knowledge:
                        certainty_count += 1

                    # #4: false → true (had false belief from #3)
                    if was_false:
                        accuracy_count += 1

                # Absent characters with beliefs
                for char in all_chars:
                    if char not in present and belief_content[char] != _UNKNOWN:
                        was_true = (belief_content[char] == old_actual)
                        is_now_false = (belief_content[char] != actual)
                        if was_true and is_now_false:
                            accuracy_count += 1  # #3: true → false
                            has_false_belief[char] = True

    return {
        'certainty': certainty_count,
        'accuracy': accuracy_count,
        'total': certainty_count + accuracy_count,
    }


def _validate_invariants(s: 'Scenario') -> None:
    """
    Defensive checks to catch logical errors early:
      - No one acts after leaving.
      - No put into a non-empty container.
      - Move item matches actual container contents.
    """
    present = set(s.present_initially)
    contents = {c: None for c in CONTAINERS_GEN}
    for idx, e in enumerate(s.events):
        if e.event_type == 'leave':
            if e.character not in present:
                raise ValueError(f"Event {idx}: {e.character} left but was not present.")
            present.discard(e.character)
        elif e.event_type == 'put':
            if e.character not in present:
                raise ValueError(f"Event {idx}: {e.character} acted after leaving.")
            if contents[e.container] is not None:
                raise ValueError(f"Event {idx}: put into non-empty {e.container}.")
            contents[e.container] = e.item
        elif e.event_type == 'move':
            if e.character not in present:
                raise ValueError(f"Event {idx}: {e.character} acted after leaving.")
            if contents[e.from_container] is None:
                raise ValueError(f"Event {idx}: move from empty {e.from_container}.")
            if contents[e.from_container] != e.item:
                raise ValueError(f"Event {idx}: move claims item '{e.item}' but {e.from_container} contains '{contents[e.from_container]}'.")
            if contents[e.to_container] is not None:
                raise ValueError(f"Event {idx}: move to non-empty {e.to_container} (contains {contents[e.to_container]}).")
            contents[e.to_container] = e.item
            contents[e.from_container] = None
        elif e.event_type == 'enter':
            if e.character in present:
                raise ValueError(f"Event {idx}: {e.character} entered but was already present.")
            present.add(e.character)
        elif e.event_type == 'remove':
            # 'remove' events should never be generated - use moves to third container instead
            raise ValueError(f"Event {idx}: 'remove' event found - these should be eliminated. Use moves instead.")


def _validate_teammate_belief_integrity(s: 'Scenario') -> int:
    """
    Ensure blue team members (A/B) don't invalidate each other's beliefs about the target.

    Rule: After A or B leaves with a belief about the target container,
    the other blue team member cannot modify the target container.

    Red team (C/D) opponents CAN invalidate each other's beliefs - this constraint
    only applies to the player's team (A/B) since that's what we're testing.

    Returns: Number of violations found.
    """
    target = s.question_container
    BLUE = {'A', 'B'}
    violation_count = 0

    def is_blue_team(char: str) -> bool:
        return char in BLUE

    present = set(s.present_initially)
    target_contents = None  # Current actual contents of target container

    # Track each character's belief about target when they depart
    # Format: {char: (believed_contents, has_observed_target)}
    char_beliefs = {c: (None, False) for c in ['A', 'B', 'C', 'D']}
    departed_with_belief = {}  # {char: believed_contents} for chars who left after observing

    for idx, event in enumerate(s.events):
        # Track target container contents
        if event.event_type == 'put' and event.container == target:
            target_contents = event.item
            # All present characters now know the target contents
            for c in present:
                char_beliefs[c] = (target_contents, True)

        elif event.event_type == 'move':
            if event.from_container == target:
                target_contents = None
                for c in present:
                    char_beliefs[c] = (None, True)
            if event.to_container == target:
                target_contents = event.item
                for c in present:
                    char_beliefs[c] = (target_contents, True)

        elif event.event_type == 'leave':
            char = event.character
            believed, has_observed = char_beliefs.get(char, (None, False))
            if has_observed:
                # This character leaves with a belief about the target
                departed_with_belief[char] = believed
            present.discard(char)

        elif event.event_type == 'enter':
            char = event.character
            present.add(char)
            # When re-entering, they don't automatically know current contents
            # They still have their old belief until they witness an event
            # (belief is preserved from when they left)

        # After any put/move that changes target, check for teammate violations
        if event.event_type in ('put', 'move'):
            affects_target = (
                (event.event_type == 'put' and event.container == target) or
                (event.event_type == 'move' and (event.from_container == target or event.to_container == target))
            )

            if affects_target:
                actor = event.character

                # Only check blue team (A/B) - opponents can invalidate each other
                if not is_blue_team(actor):
                    continue

                for departed_char, believed_contents in list(departed_with_belief.items()):
                    if departed_char not in present:  # Still departed
                        # Only check if departed char is also blue team
                        if is_blue_team(departed_char):
                            # Check if the action invalidated the belief
                            if believed_contents != target_contents:
                                violation_count += 1

    return violation_count


def _get_answerer_state(spec: dict) -> EpistemicState:
    """Get the answerer's final epistemic state from the spec."""
    if spec['Answerer'] == 'Self':
        if spec['KS_Self'] == EpistemicState.KNOWS_X:
            return EpistemicState.KNOWS_TRUTH
        else:  # BELIEVES_X
            return EpistemicState.BELIEVES_X
    elif spec['Answerer'] == 'Teammate':
        return spec['KS_Teammate']
    else:  # 'Opponent'
        return spec['KS_Opponent']


def _epistemic_matches(actual: str, expected: EpistemicState) -> bool:
    """Check if actual epistemic state matches expected, handling X variants.

    Args:
        actual: Actual state string from compute_actual_epistemic_state()
        expected: Expected state from spec

    Returns:
        True if states match (accounting for X variants)
    """
    actual_normalized = actual.upper().replace(' ', '_')
    expected_normalized = expected.value.upper().replace(' ', '_')

    # Handle KNOWS_X matching KNOWS_TRUTH
    if expected_normalized == 'KNOWS_X':
        return actual_normalized in ('KNOWS_TRUTH', 'KNOWS_X')
    if expected_normalized == 'BELIEVES_X':
        return actual_normalized in ('BELIEVES_TRUTH', 'BELIEVES_FALSE', 'BELIEVES_X')

    return actual_normalized == expected_normalized


def _validate_epistemic_states_preserved(scenario: 'Scenario', spec: dict) -> List[str]:
    """
    Verify ALL character epistemic states match spec after extra events.

    This is a comprehensive validation that checks:
    - UNKNOWN characters remain UNKNOWN
    - KNOWS_TRUTH characters still know truth
    - BELIEVES_TRUTH characters still believe truth
    - BELIEVES_FALSE characters still believe false
    - KNOWS_X/BELIEVES_X for Self characters

    Args:
        scenario: The generated scenario
        spec: The scenario specification

    Returns:
        List of error messages (empty if valid)
    """
    from validate_scenarios import compute_actual_epistemic_state, a_can_determine_state

    errors = []

    # Validate Self (KS_Self)
    expected_self = spec.get('KS_Self')
    if expected_self:
        actual = compute_actual_epistemic_state(scenario, 'A', is_self=True)
        if not _epistemic_matches(actual, expected_self):
            errors.append(f"Self (A): expected {expected_self.value}, got {actual}")

    # Validate Teammate (KS_Teammate)
    expected_teammate = spec.get('KS_Teammate')
    if expected_teammate:
        if expected_teammate == EpistemicState.UNKNOWN:
            # For UNKNOWN, check A cannot determine B's state
            if a_can_determine_state(scenario, 'B'):
                errors.append(f"Teammate (B): should be UNKNOWN but A can determine state")
        else:
            actual = compute_actual_epistemic_state(scenario, 'B')
            if not _epistemic_matches(actual, expected_teammate):
                errors.append(f"Teammate (B): expected {expected_teammate.value}, got {actual}")

    # Validate Opponent (KS_Opponent) - at least one must match
    expected_opponent = spec.get('KS_Opponent')
    if expected_opponent:
        if expected_opponent == EpistemicState.UNKNOWN:
            c_unknown = not a_can_determine_state(scenario, 'C')
            d_unknown = not a_can_determine_state(scenario, 'D')
            if not (c_unknown or d_unknown):
                errors.append(f"Opponent: at least one of C/D should be UNKNOWN")
        else:
            actual_c = compute_actual_epistemic_state(scenario, 'C')
            actual_d = compute_actual_epistemic_state(scenario, 'D')
            c_matches = _epistemic_matches(actual_c, expected_opponent)
            d_matches = _epistemic_matches(actual_d, expected_opponent)
            if not (c_matches or d_matches):
                errors.append(f"Opponent: expected {expected_opponent.value}, got C={actual_c}, D={actual_d}")

    return errors


def _find_extra_events_constraint(events: List[Event], answerer: str, player: str,
                                   answerer_state: EpistemicState, queried_container: str) -> int:
    """
    Find the index before which Extra Leave and Enter events must be inserted.
    Returns the constraint index (events must be inserted before this index).
    """
    first_put_idx = None
    last_put_idx = None
    move_idx = None
    player_leave_idx = None
    answerer_leave_idx = None
    
    for idx, event in enumerate(events):
        if event.event_type == 'put':
            if event.container == queried_container:
                if first_put_idx is None:
                    first_put_idx = idx
                last_put_idx = idx
        elif event.event_type == 'move':
            if event.to_container == queried_container or event.from_container == queried_container:
                move_idx = idx
        elif event.event_type == 'leave':
            if event.character == player:
                player_leave_idx = idx
            if event.character == answerer:
                answerer_leave_idx = idx
    
    if answerer_state in (EpistemicState.KNOWS_TRUTH, EpistemicState.BELIEVES_TRUTH):
        # If player leaves, constrain to before player leaves and before first put
        if player_leave_idx is not None:
            return min(first_put_idx, player_leave_idx)
        elif move_idx is not None:
            return move_idx
        elif last_put_idx is not None:
            return last_put_idx
        else:
            return len(events)
    elif answerer_state == EpistemicState.BELIEVES_FALSE:
        return first_put_idx if first_put_idx is not None else len(events)
    elif answerer_state == EpistemicState.UNKNOWN:
        constraints = []
        if player_leave_idx is not None:
            constraints.append(player_leave_idx)
        if answerer_leave_idx is not None:
            constraints.append(answerer_leave_idx)
        if first_put_idx is not None:
            constraints.append(first_put_idx)
        return min(constraints) if constraints else len(events)
    else:
        return len(events)


def insert_extra_events(scenario: Scenario, answerer: str, player: str,
                        answerer_state: EpistemicState, spec: dict, rng: random.Random) -> None:
    """
    Insert extra Leave and Enter events for the answerer to create an additional epistemic state.
    Modifies scenario.events in place.

    Constraints:
    1. If leaver has "Believes" or "Knows" state, a put must happen before their leave
    2. At least one event must occur between leave and enter
    3. If Self leaves, gap events must be leave/enter (not put/move, since Self wouldn't see them)
    """
    constraint_idx = _find_extra_events_constraint(
        scenario.events, answerer, player, answerer_state, scenario.question_container
    )

    # If answerer does any put/move, constrain to before their first action
    first_answerer_action_idx = None
    for idx, event in enumerate(scenario.events):
        if event.event_type in ('put', 'move') and event.character == answerer:
            first_answerer_action_idx = idx
            break

    if first_answerer_action_idx is not None:
        constraint_idx = min(constraint_idx, first_answerer_action_idx)

    # Skip if no room to insert
    if constraint_idx < 0:
        return

    # Determine leaver's epistemic state from CSV column
    if spec['Answerer'] == 'Self':
        leaver_csv_state = spec['KS_Self']
    elif spec['Answerer'] == 'Teammate':
        leaver_csv_state = spec['KS_Teammate']
    else:  # 'Opponent'
        leaver_csv_state = spec['KS_Opponent']

    # Constraint 1: Put before leave required for Believes/Knows (not Unknown)
    requires_put_before_leave = leaver_csv_state != EpistemicState.UNKNOWN

    if requires_put_before_leave:
        # Find first put to queried container
        first_put_idx = None
        for idx, event in enumerate(scenario.events):
            if event.event_type == 'put' and event.container == scenario.question_container:
                first_put_idx = idx
                break

        if first_put_idx is None:
            return  # Can't insert meaningful leave/enter

        min_leave_pos = first_put_idx + 1
    else:
        min_leave_pos = 0  # Can leave at any position

    if min_leave_pos > constraint_idx:
        return

    # Constraint 2: Gap events between leave/enter
    self_is_leaver = (answerer == player)

    # Find valid leave positions that have appropriate gap events after them
    valid_leave_positions = []
    for leave_pos in range(min_leave_pos, constraint_idx + 1):
        # Check events from leave_pos to constraint_idx for valid gap events
        has_valid_gap = False
        for gap_pos in range(leave_pos, constraint_idx):
            event = scenario.events[gap_pos]
            if self_is_leaver:
                # Self needs leave/enter events (not put/move) as gap
                if event.event_type in ('leave', 'enter'):
                    has_valid_gap = True
                    break
            else:
                # Others can have any event type as gap
                has_valid_gap = True
                break

        if has_valid_gap:
            valid_leave_positions.append(leave_pos)

    if not valid_leave_positions:
        return

    leave_insert_pos = rng.choice(valid_leave_positions)

    # Find valid enter positions (must have gap event between leave and enter)
    valid_enter_positions = []
    for enter_pos in range(leave_insert_pos + 1, constraint_idx + 1):
        # Check if there's a valid gap event between leave_insert_pos and enter_pos
        for gap_pos in range(leave_insert_pos, enter_pos):
            event = scenario.events[gap_pos]
            if self_is_leaver:
                if event.event_type in ('leave', 'enter'):
                    valid_enter_positions.append(enter_pos)
                    break
            else:
                valid_enter_positions.append(enter_pos)
                break

    if not valid_enter_positions:
        return

    enter_insert_pos = rng.choice(valid_enter_positions)

    # Insert leave first, then enter (adjusting for the index shift)
    scenario.events.insert(leave_insert_pos, Event('leave', answerer))
    scenario.events.insert(enter_insert_pos + 1, Event('enter', answerer))


def insert_extra_events_with_revelation(scenario: Scenario, answerer: str, rng: random.Random, spec: dict = None,
                                        skip_optional_events: bool = False) -> None:
    """
    Insert extra events for KNOWS_TRUTH answerers using the "revelation" pattern:
    1. Answerer sees initial put (already in events)
    2. Answerer leaves
    3. State changes while they're away (move to other container)
    4. Answerer returns
    5. Final move reveals truth to them (they witness the move back)

    Epistemic model: Containers are opaque. You learn contents ONLY by witnessing put/move events.
    """
    queried = scenario.question_container
    other = _other_container(queried)

    # Track container contents through events
    contents = {c: None for c in CONTAINERS_GEN}

    # Find the initial put to queried container
    initial_put_idx = None
    initial_item = None
    for idx, event in enumerate(scenario.events):
        if event.event_type == 'put':
            contents[event.container] = event.item
            if event.container == queried and initial_put_idx is None:
                initial_put_idx = idx
                initial_item = event.item
        elif event.event_type == 'move':
            contents[event.to_container] = event.item
            contents[event.from_container] = None
        elif event.event_type == 'remove':
            contents[event.container] = None

    if initial_put_idx is None:
        return  # Can't work without an initial put

    # Track who is present after the initial put
    present = set(scenario.present_initially)
    for idx, event in enumerate(scenario.events):
        if idx > initial_put_idx:
            break
        if event.event_type == 'leave':
            present.discard(event.character)
        elif event.event_type == 'enter':
            present.add(event.character)

    # Answerer must be present to leave
    if answerer not in present:
        return

    # After answerer leaves, we need someone still present to do the moves
    present_after_leave = present - {answerer}
    if not present_after_leave:
        return

    # Build the extra events
    leave_pos = initial_put_idx + 1

    # Ensure characters who should remain UNKNOWN are not present
    # Exclude the answerer - their leave/enter is handled separately in this function
    if spec:
        leave_pos, chars_who_left = _ensure_unknown_chars_absent(
            scenario, spec, leave_pos, rng, exclude_chars={answerer}
        )
        for char in chars_who_left:
            present_after_leave.discard(char)

    if not present_after_leave:
        return  # Everyone who could act had to leave to preserve UNKNOWN

    # Check which containers are used by events AFTER leave_pos
    non_targets = _non_target_containers(queried)
    third = non_targets[1] if other == non_targets[0] else non_targets[0]
    containers_used_after = _get_containers_used_after(scenario, leave_pos)

    can_use_other = other not in containers_used_after
    can_use_third = third not in containers_used_after

    # 1. Answerer leaves
    leave_event = Event('leave', answerer)

    # Exclude teammates of characters who left with beliefs about the target
    # Check the FULL base scenario to find all teammates who will leave with beliefs
    teammates_to_exclude = _get_teammates_to_exclude_for_target_moves(scenario, len(scenario.events))
    # ALSO exclude the answerer's teammate if the answerer is blue team.
    # The answerer leaves with a belief, so their teammate can't modify the target.
    BLUE = {'A', 'B'}
    if answerer in BLUE:
        teammates_to_exclude = teammates_to_exclude | {_teammate_of(answerer)}

    # Also exclude characters who must remain UNKNOWN
    unknown_chars = set()
    if spec:
        must_be_unknown, _ = _get_unknown_characters_from_spec(spec)
        unknown_chars = must_be_unknown

    valid_movers = [c for c in sorted(present_after_leave) if c not in teammates_to_exclude and c not in unknown_chars]
    if not valid_movers:
        # Fallback: For the revelation pattern, the moves are temporary (item returns to target
        # before any teammates leave). The base scenario's subsequent events will set the final
        # state. So we can use any present character as a mover - the validation will catch
        # any actual violations later.
        if present_after_leave:
            valid_movers = sorted(present_after_leave)
        else:
            return  # No one present to do the moves

    # 2. Someone moves item to other container while answerer is away
    # Track container contents up to leave_pos (where we insert events)
    contents_at_leave = {c: None for c in CONTAINERS_GEN}
    for idx, event in enumerate(scenario.events):
        if idx >= leave_pos:
            break
        if event.event_type == 'put':
            contents_at_leave[event.container] = event.item
        elif event.event_type == 'move':
            contents_at_leave[event.to_container] = event.item
            contents_at_leave[event.from_container] = None

    # Determine which container to use as temp storage
    other_item = contents_at_leave[other]
    third_item = contents_at_leave[third]
    clear_temp_event = None
    temp_container = None

    # Try primary pattern: use 'other' as temp
    if can_use_other and (other_item is None or (can_use_third and third_item is None)):
        temp_container = other
        if other_item is not None:
            # Need to clear other container first by moving to third
            clear_temp_event = Event('move', rng.choice(valid_movers), from_container=other, to_container=third, item=other_item)
    # Fallback: use 'third' as temp (if other is blocked but third is available)
    elif can_use_third and third_item is None:
        temp_container = third
    else:
        return  # No valid temp container available

    mover1 = rng.choice(valid_movers)
    move_away_event = Event('move', mover1, from_container=queried, to_container=temp_container, item=initial_item)

    # 3. Answerer returns
    enter_event = Event('enter', answerer)

    # 4. Revelation: answerer witnesses item being placed in queried container
    # 50% chance: use a DIFFERENT item (adds variety without affecting certainty ECTs)
    # The answerer's certainty transitions (knowledge ↔ belief) depend on presence/absence,
    # not on which specific item is in the container.
    # IMPORTANT: Only use variation if queried container is NOT used by later base events,
    # because those events expect initial_item to be there.
    valid_movers2 = [c for c in valid_movers if c != mover1]
    mover2 = rng.choice(valid_movers2) if valid_movers2 else mover1
    queried_used_after = queried in containers_used_after
    if rng.random() < 0.5 and not queried_used_after:
        # Variation: put a different item in queried container
        # Original item stays in temp_container, new item goes in 'queried'
        # Exclude all items currently in any container
        used_items = {contents[c] for c in CONTAINERS_GEN if contents[c] is not None}
        available_items = [i for i in ITEMS_GEN if i not in used_items]
        if not available_items:
            # Fall back to standard move if no unique items available
            revelation_event = Event('move', mover2, from_container=temp_container, to_container=queried, item=initial_item)
        else:
            new_item = rng.choice(available_items)
            revelation_event = Event('put', mover2, container=queried, item=new_item)
    else:
        # Standard: move original item back
        revelation_event = Event('move', mover2, from_container=temp_container, to_container=queried, item=initial_item)

    # Build list of events to insert
    events_to_insert = [leave_event]

    # 50% chance: have another character leave while answerer is away
    # (must keep at least one person present for the moves)
    # Also exclude any character who has events later in the base scenario
    # AND exclude teammates of characters who modify the target later (to prevent belief violations)
    # Skip if caller indicates 1A can't compensate (to avoid SIT gap violations)
    if not skip_optional_events and rng.random() < 0.5 and len(present_after_leave) > 1:
        # Find characters who act later in the base scenario (can't make them leave)
        chars_acting_later = set()
        # Find characters who modify the target later (their teammates shouldn't leave with beliefs)
        chars_modifying_target_later = set()
        for evt in scenario.events[leave_pos:]:
            chars_acting_later.add(evt.character)
            if evt.event_type == 'put' and evt.container == queried:
                chars_modifying_target_later.add(evt.character)
            elif evt.event_type == 'move' and (evt.from_container == queried or evt.to_container == queried):
                chars_modifying_target_later.add(evt.character)

        # Teammates of target modifiers shouldn't leave (would form beliefs that get violated)
        teammates_of_modifiers = {_teammate_of(c) for c in chars_modifying_target_later if c in {'A', 'B', 'C', 'D'}}

        # Characters who need KNOWS_TRUTH should not leave (they must be present at end)
        must_stay = set()
        if spec:
            if spec.get('KS_Teammate') == EpistemicState.KNOWS_TRUTH:
                must_stay.add('B')  # Teammate must stay
            if spec.get('KS_Self') == EpistemicState.KNOWS_X:
                must_stay.add('A')  # Self must stay (or re-enter, but answerer already handles that)

        potential_leavers = [c for c in sorted(present_after_leave)
                            if c != mover1 and c != mover2 and c not in chars_acting_later
                            and c not in teammates_of_modifiers and c not in must_stay]
        if potential_leavers:
            leaver = rng.choice(potential_leavers)
            events_to_insert.append(Event('leave', leaver))

    # Add clear_temp_event if needed (clears temp container before move_away)
    if clear_temp_event is not None:
        events_to_insert.append(clear_temp_event)

    events_to_insert.append(move_away_event)
    events_to_insert.append(enter_event)
    events_to_insert.append(revelation_event)

    # Insert all events at the correct position
    for i, evt in enumerate(events_to_insert):
        scenario.events.insert(leave_pos + i, evt)


def insert_extra_events_believes_true(scenario: Scenario, answerer: str, rng: random.Random,
                                      spec: dict = None) -> None:
    """
    Insert extra events for BELIEVES_TRUE answerers using ACCURACY load.

    Pattern: While answerer is absent, cycle items through target container.
    The answerer's belief about the target becomes false, then true again.
    Result: Their belief (from before leaving) is still true, but accuracy ECTs added.

    This creates accuracy ECTs:
    - #3 (true→false): When target contents change while answerer absent with true belief
    - #4 (false→true): When target contents revert while answerer still absent

    Args:
        scenario: The scenario to modify
        answerer: The character who answers (has BELIEVES_TRUE state)
        rng: Random number generator
        spec: Optional spec dict with KS_Teammate, KS_Opponent to preserve UNKNOWN states
    """
    queried = scenario.question_container
    other = _other_container(queried)
    # Get the third container (not queried, not other)
    non_targets = _non_target_containers(queried)
    third = non_targets[1] if other == non_targets[0] else non_targets[0]

    # Find where answerer leaves and track container contents
    state = _find_leave_and_track_state(scenario, answerer)
    if state.leave_idx is None:
        return

    answerer_leave_idx = state.leave_idx
    contents = state.contents
    present = state.present

    # Z = what answerer believes (item in queried container when they left)
    # X = item in other container (might be None)
    Z = contents[queried]
    X = contents[other]

    if Z is None:
        return  # Nothing to work with - answerer has no belief about target

    # Pick a new item Y - exclude ALL items used in ANY event (including future events)
    all_scenario_items = {e.item for e in scenario.events if e.item}
    used_items = all_scenario_items | {contents[c] for c in CONTAINERS_GEN if contents[c] is not None}
    available_items = [item for item in ITEMS_GEN if item not in used_items]
    if not available_items:
        return  # No unique items available
    Y = rng.choice(available_items)

    # After answerer leaves, we need someone to do the moves
    present.discard(answerer)
    if not present:
        return

    # Exclude teammates of characters who left with beliefs about the target
    # Check the FULL base scenario to find all teammates who will leave with beliefs
    insert_pos = answerer_leave_idx + 1

    # Ensure characters who should remain UNKNOWN are not present
    # Exclude the answerer - their presence/absence is determined by the base scenario
    if spec:
        insert_pos, chars_who_left = _ensure_unknown_chars_absent(
            scenario, spec, insert_pos, rng, exclude_chars={answerer}
        )
        for char in chars_who_left:
            present.discard(char)

    if not present:
        return  # Everyone who could act had to leave to preserve UNKNOWN

    teammates_to_exclude = _get_teammates_to_exclude_for_target_moves(scenario, len(scenario.events))

    # Also exclude characters who must remain UNKNOWN
    unknown_chars = set()
    if spec:
        must_be_unknown, _ = _get_unknown_characters_from_spec(spec)
        unknown_chars = must_be_unknown

    valid_actors = [c for c in sorted(present) if c not in teammates_to_exclude and c not in unknown_chars]
    if not valid_actors:
        return  # Cannot insert extra events without violating constraints

    # Check if containers are used by events AFTER insert_pos - if so, we can't safely add extra events
    containers_used_after = _get_containers_used_after(scenario, insert_pos)
    # Check container availability
    third_contents = contents[third]

    # Determine which containers we can use based on what's used after insert_pos
    can_use_other = other not in containers_used_after
    can_use_third = third not in containers_used_after
    queried_used_after = queried in containers_used_after

    # If queried is used after, we can't safely do round-trip through it
    if queried_used_after:
        return  # Can't add extra events without conflicting with later base events

    # Build the extra events, alternating actors where possible
    extra_events = []
    last_actor = None

    def choose_actor():
        nonlocal last_actor
        candidates = [c for c in valid_actors if c != last_actor]
        actor = rng.choice(candidates) if candidates else rng.choice(valid_actors)
        last_actor = actor
        return actor

    # Try primary pattern: Z round-trip through 'other'
    # Requires: other is empty OR we can move X from other to third
    if can_use_other and (X is None or (can_use_third and third_contents is None)):
        # a. If X exists in other container: MOVE X to third container (to make room)
        third_occupied = False
        if X is not None:
            extra_events.append(Event('move', choose_actor(), from_container=other, to_container=third, item=X))
            third_occupied = True

        # b. MOVE Z from queried to other container (answerer's belief now FALSE - accuracy ECT #3)
        extra_events.append(Event('move', choose_actor(), from_container=queried, to_container=other, item=Z))

        # c. PUT Y in queried container (target now has Y, not Z)
        # d. MOVE Y from queried to third container (target now empty)
        # Skip c and d if third is occupied or third already has item or third used after
        if not third_occupied and third_contents is None and can_use_third:
            extra_events.append(Event('put', choose_actor(), container=queried, item=Y))
            extra_events.append(Event('move', choose_actor(), from_container=queried, to_container=third, item=Y))

        # e. MOVE Z from other back to queried container (answerer's belief now TRUE - accuracy ECT #4)
        extra_events.append(Event('move', choose_actor(), from_container=other, to_container=queried, item=Z))

    # Fallback pattern: Z round-trip through 'third' (if other is blocked but third is available)
    elif can_use_third and third_contents is None:
        # MOVE Z from queried to third (belief becomes FALSE)
        extra_events.append(Event('move', choose_actor(), from_container=queried, to_container=third, item=Z))
        # MOVE Z from third back to queried (belief becomes TRUE)
        extra_events.append(Event('move', choose_actor(), from_container=third, to_container=queried, item=Z))

    else:
        return  # No valid pattern available

    # Insert all extra events right after answerer leaves
    for i, event in enumerate(extra_events):
        scenario.events.insert(insert_pos + i, event)


def insert_extra_events_believes_false(scenario: Scenario, answerer: str, rng: random.Random,
                                       spec: dict = None) -> None:
    """
    Insert extra events for BELIEVES_FALSE answerers using ACCURACY load.

    Pattern: While answerer is absent, cycle items through target container.
    This adds accuracy transitions by making the answerer's belief temporarily
    false, then true, then false again.
    Result: Their belief is still false, but accuracy ECTs added.

    For BELIEVES_FALSE:
    - Answerer believes W is in target (what they last saw)
    - After subsequent base events, truth becomes X (different item) → false belief
    - We insert extra item cycling right after answerer leaves

    Args:
        scenario: The scenario to modify
        answerer: The character who answers (has BELIEVES_FALSE state)
        rng: Random number generator
        spec: Optional spec dict with KS_Teammate, KS_Opponent to preserve UNKNOWN states
    """
    queried = scenario.question_container
    other = _other_container(queried)
    # Get the third container (not queried, not other)
    non_targets = _non_target_containers(queried)
    third = non_targets[1] if other == non_targets[0] else non_targets[0]

    # Find where answerer leaves and track container contents at that point
    state = _find_leave_and_track_state(scenario, answerer)
    if state.leave_idx is None:
        return

    answerer_leave_idx = state.leave_idx
    contents = state.contents
    present = state.present

    # W = what answerer believes (item in queried container when they left)
    # X = item in other container (might be None)
    W = contents[queried]  # Answerer's belief
    X = contents[other]

    if W is None:
        return  # Answerer has no belief about target

    # Pick a new item Y - exclude ALL items used in ANY event (including future events)
    all_scenario_items = {e.item for e in scenario.events if e.item}
    used_items = all_scenario_items | {contents[c] for c in CONTAINERS_GEN if contents[c] is not None}
    available_items = [item for item in ITEMS_GEN if item not in used_items]
    if not available_items:
        return  # No unique items available
    Y = rng.choice(available_items)

    # After answerer leaves, we need someone to do the moves
    present.discard(answerer)
    if not present:
        return

    # Insert events right after answerer leaves
    insert_pos = answerer_leave_idx + 1

    # Ensure characters who should remain UNKNOWN are not present
    if spec:
        insert_pos, chars_who_left = _ensure_unknown_chars_absent(
            scenario, spec, insert_pos, rng, exclude_chars={answerer}
        )
        for char in chars_who_left:
            present.discard(char)

    if not present:
        return  # Everyone who could act had to leave to preserve UNKNOWN

    # Check which containers are used by events AFTER insert_pos
    # Note: For BELIEVES_FALSE, queried WILL be used after (to change belief to false)
    # Our round-trip puts W back, so later events can still change it
    containers_used_after = _get_containers_used_after(scenario, insert_pos)

    can_use_other = other not in containers_used_after
    can_use_third = third not in containers_used_after

    # Exclude teammates of characters who left with beliefs about the target
    # Check the FULL base scenario to find all teammates who will leave with beliefs
    teammates_to_exclude = _get_teammates_to_exclude_for_target_moves(scenario, len(scenario.events))

    # Also exclude characters who must remain UNKNOWN
    unknown_chars = set()
    if spec:
        must_be_unknown, _ = _get_unknown_characters_from_spec(spec)
        unknown_chars = must_be_unknown

    valid_actors = [c for c in sorted(present) if c not in teammates_to_exclude and c not in unknown_chars]
    if not valid_actors:
        return  # Cannot insert extra events without violating constraints

    # Build the extra events, alternating actors where possible
    extra_events = []
    last_actor = None

    def choose_actor():
        nonlocal last_actor
        candidates = [c for c in valid_actors if c != last_actor]
        actor = rng.choice(candidates) if candidates else rng.choice(valid_actors)
        last_actor = actor
        return actor

    # ACCURACY LOAD: Cycle items through target while answerer is absent
    # This happens right after answerer leaves, before base scenario events continue.
    # We add a round-trip: W→out, W→back (and optionally Y→in, Y→out)
    # This adds accuracy transitions, then base events will change to final false state.
    #
    # Check container availability
    third_contents = contents[third]

    # Primary pattern: W round-trip through 'other'
    if can_use_other and (X is None or (can_use_third and third_contents is None)):
        if X is not None:
            # Other has X, move it to third first
            extra_events.append(Event('move', choose_actor(), from_container=other, to_container=third, item=X))
            # Now other is empty but third has X - can't use third for Y
            # Simplified pattern: just do W round-trip for accuracy ECTs
            extra_events.append(Event('move', choose_actor(), from_container=queried, to_container=other, item=W))
            extra_events.append(Event('move', choose_actor(), from_container=other, to_container=queried, item=W))
        else:
            # X is None, other is empty - full pattern with Y
            extra_events.append(Event('move', choose_actor(), from_container=queried, to_container=other, item=W))

            if third_contents is None and can_use_third:
                extra_events.append(Event('put', choose_actor(), container=queried, item=Y))
                extra_events.append(Event('move', choose_actor(), from_container=queried, to_container=third, item=Y))

            extra_events.append(Event('move', choose_actor(), from_container=other, to_container=queried, item=W))

    # Fallback pattern: W round-trip through 'third'
    elif can_use_third and third_contents is None:
        extra_events.append(Event('move', choose_actor(), from_container=queried, to_container=third, item=W))
        extra_events.append(Event('move', choose_actor(), from_container=third, to_container=queried, item=W))

    else:
        return  # No valid pattern available

    # Now the base scenario events will continue and eventually change W to something else,
    # making the final belief false as required.

    # Insert all extra events right after answerer leaves
    for i, event in enumerate(extra_events):
        scenario.events.insert(insert_pos + i, event)


def insert_extra_events_believes_x(scenario: Scenario, answerer: str, rng: random.Random,
                                   skip_optional_events: bool = False,
                                   spec: dict = None) -> None:
    """
    Insert extra events for BELIEVES_X answerers (Self who believes something uncertain).
    Pattern: Answerer sees events, leaves, then more events happen after they leave.
    Result: Their belief is unchanged (they left), but scenario has more complexity.

    Args:
        scenario: The scenario to modify
        answerer: The character who answers (has BELIEVES_X state)
        rng: Random number generator
        skip_optional_events: If True, skip the 50% chance leave/enter pair to reduce
            SIT count when 1A can't compensate with adaptive filler.
        spec: Optional spec dict with KS_Teammate, KS_Opponent to preserve UNKNOWN states
    """
    queried = scenario.question_container
    other = _other_container(queried)

    # Find where answerer leaves and track container contents
    state = _find_leave_and_track_state(scenario, answerer)
    if state.leave_idx is None:
        return

    answerer_leave_idx = state.leave_idx
    contents = state.contents
    present = state.present  # Already has answerer removed

    # After answerer leaves, we need someone to do the moves
    if not present:
        return

    # What's in the queried container when answerer left?
    item_in_queried = contents[queried]
    if item_in_queried is None:
        return  # Nothing to move

    # Check container contents
    item_in_other = contents[other]
    # Get the third container
    non_targets = _non_target_containers(queried)
    third = non_targets[1] if other == non_targets[0] else non_targets[0]
    item_in_third = contents[third]

    # Insert events after answerer leaves
    insert_pos = answerer_leave_idx + 1

    # Ensure characters who should remain UNKNOWN are not present
    if spec:
        insert_pos, chars_who_left = _ensure_unknown_chars_absent(
            scenario, spec, insert_pos, rng, exclude_chars={answerer}
        )
        for char in chars_who_left:
            present.discard(char)

    if not present:
        return  # Everyone who could act had to leave to preserve UNKNOWN

    # Check if containers are used by events AFTER insert_pos
    containers_used_after = _get_containers_used_after(scenario, insert_pos)

    can_use_other = other not in containers_used_after and item_in_other is None
    can_use_third = third not in containers_used_after and item_in_third is None

    # Exclude teammates of characters who left with beliefs about the target
    # Check the FULL base scenario to find all teammates who will leave with beliefs
    teammates_to_exclude = _get_teammates_to_exclude_for_target_moves(scenario, len(scenario.events))

    # Also exclude characters who must remain UNKNOWN
    unknown_chars = set()
    if spec:
        must_be_unknown, _ = _get_unknown_characters_from_spec(spec)
        unknown_chars = must_be_unknown

    valid_movers = [c for c in sorted(present) if c not in teammates_to_exclude and c not in unknown_chars]
    if not valid_movers:
        return  # Cannot insert extra events without violating constraints

    # Determine which container to use for round-trip
    if can_use_other:
        temp_container = other
    elif can_use_third:
        temp_container = third
    else:
        return  # No available container for round-trip

    mover1 = rng.choice(valid_movers)
    move_away = Event('move', mover1, from_container=queried, to_container=temp_container, item=item_in_queried)

    # Prefer different mover than mover1, still respecting teammate exclusions
    valid_movers2 = [c for c in valid_movers if c != mover1]
    mover2 = rng.choice(valid_movers2) if valid_movers2 else mover1
    move_back = Event('move', mover2, from_container=temp_container, to_container=queried, item=item_in_queried)

    # Build list of events to insert
    events_to_insert = []

    # CRITICAL: Answerer must WITNESS events on target container to gain knowledge.
    # Per EPISTEMIC_METRICS.md, entering alone does NOT create ECT #2 (belief → knowledge).
    # The answerer must SEE a put/move/remove on the target container.
    #
    # Correct order:
    #   1. Answerer enters
    #   2. Answerer WITNESSES moves on target container → ECT #2 (belief → knowledge)
    #   3. Answerer leaves → ECT #1 (knowledge → belief)
    events_to_insert.append(Event('enter', answerer))
    events_to_insert.append(move_away)
    events_to_insert.append(move_back)

    # Optional: 50% chance for another character's leave/enter pair for visible complexity
    # Skip if caller indicates 1A can't compensate with filler (to avoid SIT gap violations)
    potential = sorted(present)  # sorted for deterministic ordering
    if not skip_optional_events and potential and rng.random() < 0.5:
        opp = rng.choice(potential)
        events_to_insert.append(Event('leave', opp))
        events_to_insert.append(Event('enter', opp))

    events_to_insert.append(Event('leave', answerer))

    # Insert all events at the correct position
    for i, evt in enumerate(events_to_insert):
        scenario.events.insert(insert_pos + i, evt)


def insert_extra_events_unknown(scenario: Scenario, answerer: str, rng: random.Random,
                                spec: dict = None) -> None:
    """
    Insert extra events for UNKNOWN answerers.
    The answerer left without seeing relevant puts, so they have no knowledge.
    Add moves/events at the END of the scenario to increase complexity.

    IMPORTANT: For UNKNOWN answerers, we must insert at the END of the scenario,
    not right after they leave. This is because the put to the target container
    happens AFTER they leave (that's why they're UNKNOWN). Inserting right after
    they leave would be before the put, so no one would have observed the target
    yet and we'd add no ECT.

    By inserting at the end:
    - The put has already happened
    - Characters present have observed the target container
    - Moves on target affect those who observed it (ECT #3 for absent believers)
    - Leave/enter triggers ECT #1 for the leaving character (who observed target)

    Args:
        scenario: The scenario to modify
        answerer: The character who answers (has UNKNOWN state)
        rng: Random number generator
        spec: Optional spec dict with KS_Teammate, KS_Opponent to preserve UNKNOWN states
    """
    queried = scenario.question_container
    other = _other_container(queried)

    # Track state through ALL events to find final state
    contents = {c: None for c in CONTAINERS_GEN}
    present = set(scenario.present_initially)
    answerer_found = False

    for event in scenario.events:
        if event.event_type == 'put':
            contents[event.container] = event.item
        elif event.event_type == 'move':
            contents[event.to_container] = event.item
            contents[event.from_container] = None
        elif event.event_type == 'remove':
            contents[event.container] = None
        elif event.event_type == 'leave':
            if event.character == answerer:
                answerer_found = True
            present.discard(event.character)
        elif event.event_type == 'enter':
            present.add(event.character)

    if not answerer_found:
        return  # Answerer never left

    # We need someone present to do the events
    if not present:
        return

    # Insert at end of scenario
    insert_pos = len(scenario.events)

    # Ensure characters who should remain UNKNOWN are not present
    # This prevents them from witnessing the extra events on the target container
    if spec:
        insert_pos, chars_who_left = _ensure_unknown_chars_absent(
            scenario, spec, insert_pos, rng, exclude_chars={answerer}
        )
        # Update present set after leave events were inserted
        for char in chars_who_left:
            present.discard(char)

    if not present:
        return  # Everyone who could act had to leave to preserve UNKNOWN

    item_in_queried = contents[queried]
    item_in_other = contents[other]

    # Exclude teammates of characters who left with beliefs about the target
    # Check the FULL base scenario to find all teammates who will leave with beliefs
    teammates_to_exclude = _get_teammates_to_exclude_for_target_moves(scenario, len(scenario.events))

    # Also exclude characters who must remain UNKNOWN
    unknown_chars = set()
    if spec:
        must_be_unknown, opponent_needs_one = _get_unknown_characters_from_spec(spec)
        unknown_chars = must_be_unknown
        # For opponent_needs_one, we already handled it in _ensure_unknown_chars_absent

    valid_movers = [c for c in sorted(present) if c not in teammates_to_exclude and c not in unknown_chars]
    if not valid_movers:
        return  # Cannot insert extra events without violating constraints

    events_to_insert = []
    non_targets = _non_target_containers(queried)
    third = non_targets[1] if other == non_targets[0] else non_targets[0]

    if item_in_queried is not None and item_in_other is None:
        # Can safely move from queried to other (other is empty)
        mover1 = rng.choice(valid_movers)
        events_to_insert.append(Event('move', mover1, from_container=queried, to_container=other, item=item_in_queried))
        mover2 = _choose_different_actor(set(valid_movers), mover1, rng)
        events_to_insert.append(Event('move', mover2, from_container=other, to_container=queried, item=item_in_queried))
    elif item_in_other is not None and item_in_queried is None:
        # Can safely move from other to queried (queried is empty)
        mover1 = rng.choice(valid_movers)
        events_to_insert.append(Event('move', mover1, from_container=other, to_container=queried, item=item_in_other))
        mover2 = _choose_different_actor(set(valid_movers), mover1, rng)
        events_to_insert.append(Event('move', mover2, from_container=queried, to_container=other, item=item_in_other))
    elif item_in_queried is None and item_in_other is None:
        # Both containers empty — add put to TARGET, then move to non-target
        # Only if third is empty too
        if contents[third] is not None:
            return  # Can't do this pattern - third container has item
        used_items = {contents[c] for c in CONTAINERS_GEN if contents[c] is not None}
        available_items = [i for i in ITEMS_GEN if i not in used_items]
        if not available_items:
            return  # No unique items available
        item = rng.choice(available_items)
        actor1 = rng.choice(valid_movers)
        events_to_insert.append(Event('put', actor1, container=queried, item=item))
        actor2 = _choose_different_actor(set(valid_movers), actor1, rng)
        events_to_insert.append(Event('move', actor2, from_container=queried, to_container=third, item=item))
    else:
        # Both queried and other have items - use third as temp container if available
        if contents[third] is not None:
            return  # Can't do this pattern - all containers occupied
        # Move item from queried to third, then back
        mover1 = rng.choice(valid_movers)
        events_to_insert.append(Event('move', mover1, from_container=queried, to_container=third, item=item_in_queried))
        mover2 = _choose_different_actor(set(valid_movers), mover1, rng)
        events_to_insert.append(Event('move', mover2, from_container=third, to_container=queried, item=item_in_queried))

    # Mandatory leave/enter pair: the leaving character has observed the target
    # (since puts happened earlier), so this triggers ECT #1 (knowledge → belief).
    potential = sorted(present)  # sorted for deterministic ordering
    if potential:
        opp = rng.choice(potential)
        events_to_insert.append(Event('leave', opp))
        events_to_insert.append(Event('enter', opp))

    for i, evt in enumerate(events_to_insert):
        scenario.events.insert(insert_pos + i, evt)


def insert_teammate_reentry_unknown(scenario: Scenario, teammate: str, rng: random.Random) -> None:
    """
    For scenarios where Self=BELIEVES_X and Teammate=UNKNOWN,
    add teammate re-entry after final target state.

    Teammate enters then leaves - adding 2 SIT events.
    Teammate's state transitions: UNKNOWN → KNOWS_TRUTH → UNKNOWN (from A's view).
    A can't observe any of this (A left earlier), so A's classification of teammate remains UNKNOWN.
    """
    # Verify teammate actually left (not present at end)
    present = set(scenario.present_initially)
    for event in scenario.events:
        if event.event_type == 'leave':
            present.discard(event.character)
        elif event.event_type == 'enter':
            present.add(event.character)

    if teammate in present:
        return  # Teammate is still present, no re-entry needed

    # Insert at end of scenario
    scenario.events.append(Event('enter', teammate))
    scenario.events.append(Event('leave', teammate))


def insert_extra_puts(scenario: Scenario, answerer: str, rng: random.Random,
                      spec: dict = None) -> None:
    """
    Insert extra put/move events for scenarios where Answerer=Teammate,
    Self=Knows X, Teammate=Believes Truth, with Extra=1.
    Creates journey: Believes Truth -> Believes False -> Believes Truth
    Modifies scenario.events in place. Uses moves to third container instead of removes.

    Args:
        scenario: The scenario to modify
        answerer: The character who answers (the Teammate)
        rng: Random number generator
        spec: Optional spec dict with KS_Teammate, KS_Opponent to preserve UNKNOWN states
    """
    queried = scenario.question_container
    other = _other_container(queried)
    # Get the third container (not queried, not other)
    non_targets = _non_target_containers(queried)
    third = non_targets[1] if other == non_targets[0] else non_targets[0]

    # Find where answerer leaves and track container contents
    state = _find_leave_and_track_state(scenario, answerer)
    if state.leave_idx is None:
        return

    answerer_leave_idx = state.leave_idx
    contents = state.contents
    present = state.present  # Already has answerer removed

    # Z = what answerer believes (item in queried container when they left)
    # X = item in other container (might be None)
    Z = contents[queried]
    X = contents[other]

    if Z is None:
        return  # Nothing to work with

    # Pick a new item Y - exclude ALL items used in ANY event (including future events)
    all_scenario_items = {e.item for e in scenario.events if e.item}
    used_items = all_scenario_items | {contents[c] for c in CONTAINERS_GEN if contents[c] is not None}
    available_items = [item for item in ITEMS_GEN if item not in used_items]
    if not available_items:
        return  # No unique items available
    Y = rng.choice(available_items)

    # Pick random present characters to perform actions
    if not present:
        return

    # Exclude teammates of blue team members who left with beliefs about the target
    # Check the FULL base scenario to find all teammates who will leave with beliefs
    insert_pos = answerer_leave_idx + 1

    # Ensure characters who should remain UNKNOWN are not present
    if spec:
        insert_pos, chars_who_left = _ensure_unknown_chars_absent(
            scenario, spec, insert_pos, rng, exclude_chars={answerer}
        )
        for char in chars_who_left:
            present.discard(char)

    if not present:
        return  # Everyone who could act had to leave to preserve UNKNOWN

    teammates_to_exclude = _get_teammates_to_exclude_for_target_moves(scenario, len(scenario.events))

    # Also exclude characters who must remain UNKNOWN
    unknown_chars = set()
    if spec:
        must_be_unknown, _ = _get_unknown_characters_from_spec(spec)
        unknown_chars = must_be_unknown

    valid_actors = [c for c in sorted(present) if c not in teammates_to_exclude and c not in unknown_chars]
    if not valid_actors:
        return  # Cannot insert extra events without violating constraints

    # Check which containers are used by events AFTER insert_pos
    containers_used_after = _get_containers_used_after(scenario, insert_pos)

    # Check container availability
    third_contents = contents[third]
    can_use_other = other not in containers_used_after
    can_use_third = third not in containers_used_after

    # Build the extra events, alternating actors where possible
    extra_events = []
    last_actor = None

    def choose_actor():
        nonlocal last_actor
        candidates = [c for c in valid_actors if c != last_actor]
        actor = rng.choice(candidates) if candidates else rng.choice(valid_actors)
        last_actor = actor
        return actor

    # Primary pattern: Z round-trip through 'other'
    if can_use_other and (X is None or (can_use_third and third_contents is None)):
        # a. If X exists in other container: MOVE X to third container
        third_occupied = False
        if X is not None:
            extra_events.append(Event('move', choose_actor(), from_container=other, to_container=third, item=X))
            third_occupied = True

        # b. MOVE Z from queried to other container
        extra_events.append(Event('move', choose_actor(), from_container=queried, to_container=other, item=Z))

        # c. PUT Y in queried container
        # d. MOVE Y from queried to third container
        # Skip c and d if third is occupied or third already has item or third used after
        if not third_occupied and third_contents is None and can_use_third:
            extra_events.append(Event('put', choose_actor(), container=queried, item=Y))
            extra_events.append(Event('move', choose_actor(), from_container=queried, to_container=third, item=Y))

        # e. MOVE Z from other back to queried container
        extra_events.append(Event('move', choose_actor(), from_container=other, to_container=queried, item=Z))

    # Fallback pattern: Z round-trip through 'third' (if other is blocked but third is available)
    elif can_use_third and third_contents is None:
        extra_events.append(Event('move', choose_actor(), from_container=queried, to_container=third, item=Z))
        extra_events.append(Event('move', choose_actor(), from_container=third, to_container=queried, item=Z))

    else:
        return  # No valid pattern available

    # Insert all extra events right after answerer leaves
    insert_pos = answerer_leave_idx + 1
    for i, event in enumerate(extra_events):
        scenario.events.insert(insert_pos + i, event)


def insert_filler_events(scenario: Scenario, rng: random.Random) -> None:
    """Insert neutral filler events into Extra=0 scenarios to add situation
    tracking demands without adding epistemic category transitions (ECTs).

    Uses MOVES between the two NON-TARGET containers so that no character's
    epistemic state about the target container changes. No 'remove' events.

    Pattern: Move items between non-target containers (2-4 moves).
    No leave/enter pair — that would trigger ECT #1 (knowledge → belief)
    for the leaving character, narrowing the ECT gap with Extra=1.

    Insertion point: the last position where A (the player) is still present.
    Move events are only visible to A while A is in the room, so
    inserting after A leaves would add invisible (wasted) filler.
    """
    target = scenario.question_container
    non_targets = _non_target_containers(target)  # e.g., ['box', 'basket']

    # Find the last position where A is present, and track state at that point
    a_present = 'A' in scenario.present_initially
    insert_idx = len(scenario.events)  # default: end of events
    present = set(scenario.present_initially)
    contents = {c: None for c in CONTAINERS_GEN}

    for idx, event in enumerate(scenario.events):
        if event.event_type == 'put':
            contents[event.container] = event.item
        elif event.event_type == 'move':
            contents[event.to_container] = event.item
            contents[event.from_container] = None
        elif event.event_type == 'leave':
            if event.character == 'A' and a_present:
                # Insert right before A leaves
                insert_idx = idx
                break
            present.discard(event.character)
        elif event.event_type == 'enter':
            present.add(event.character)
            if event.character == 'A':
                a_present = True

    if not present:
        return  # No one present to do filler events

    # Find containers used by events AFTER insert_idx - we must avoid these
    containers_used_after = _get_containers_used_after(scenario, insert_idx)

    # Filter non-targets to only those not used after insert_idx
    safe_non_targets = [c for c in non_targets if c not in containers_used_after]
    if len(safe_non_targets) < 2:
        return  # Need at least 2 safe containers for put+move pattern

    events_to_add = []
    present_list = sorted(present)  # sorted for deterministic ordering

    # Get contents of safe non-target containers
    nt0, nt1 = safe_non_targets[0], safe_non_targets[1]
    item0 = contents[nt0]
    item1 = contents[nt1]

    # Pick two different items for variety - exclude ALL items used in ANY event
    all_scenario_items = {e.item for e in scenario.events if e.item}
    used_items = all_scenario_items | {contents[c] for c in CONTAINERS_GEN if contents[c] is not None}
    available_items = [i for i in ITEMS_GEN if i not in used_items]
    if len(available_items) < 2:
        return  # Not enough unique items available
    new_item1 = rng.choice(available_items)
    new_item2 = rng.choice([i for i in available_items if i != new_item1])

    if item0 is not None and item1 is not None:
        # Both have items: can't add filler without overflow - skip
        pass
    elif item0 is not None:
        # First non-target has item: move it to keep nt0 empty
        a1 = rng.choice(present_list)
        events_to_add.append(Event('move', a1, from_container=nt0, to_container=nt1, item=item0))
    elif item1 is not None:
        # Second non-target has item: move it to keep nt1 empty
        a1 = rng.choice(present_list)
        events_to_add.append(Event('move', a1, from_container=nt1, to_container=nt0, item=item1))
    else:
        # Both empty: put then move, keeps nt0 empty for subsequent operations
        a1 = rng.choice(present_list)
        events_to_add.append(Event('put', a1, container=nt0, item=new_item1))
        a2 = _choose_different_actor(present, a1, rng)
        events_to_add.append(Event('move', a2, from_container=nt0, to_container=nt1, item=new_item1))

    for i, evt in enumerate(events_to_add):
        scenario.events.insert(insert_idx + i, evt)


def insert_n_filler_events(scenario: Scenario, rng: random.Random, n: int) -> None:
    """Insert exactly n neutral filler events for Extra=0A/0B scenarios.

    If n=0, does nothing (minimal scenario for 0A).
    If n>0, generates n moves/puts on non-target containers, falling back to
    leave/enter pairs when container capacity is exhausted.

    Unlike insert_filler_events() which aims for SIT parity with 1B,
    this function adds a fixed number of filler events.

    Uses same insertion logic as insert_filler_events() - events are
    inserted before player A leaves (last position where A is present).
    """
    if n <= 0:
        return  # Extra=0A: minimal scenario with no filler

    target = scenario.question_container
    non_targets = _non_target_containers(target)

    # Find the last position where A is present
    a_present = 'A' in scenario.present_initially
    insert_idx = len(scenario.events)
    present = set(scenario.present_initially)
    contents = {c: None for c in CONTAINERS_GEN}

    for idx, event in enumerate(scenario.events):
        if event.event_type == 'put':
            contents[event.container] = event.item
        elif event.event_type == 'move':
            contents[event.to_container] = event.item
            contents[event.from_container] = None
        elif event.event_type == 'leave':
            if event.character == 'A' and a_present:
                insert_idx = idx
                break
            present.discard(event.character)
        elif event.event_type == 'enter':
            present.add(event.character)
            if event.character == 'A':
                a_present = True

    if not present:
        return  # No one present to do filler events

    # Track who has observed the target (for ECT-safe leave/enter)
    chars_observed_target = set()
    obs_present = set(scenario.present_initially)
    for idx, event in enumerate(scenario.events):
        if idx >= insert_idx:
            break
        if event.event_type == 'put' and event.container == target:
            chars_observed_target.update(obs_present)
        elif event.event_type == 'move' and (event.from_container == target or event.to_container == target):
            chars_observed_target.update(obs_present)
        if event.event_type == 'leave':
            obs_present.discard(event.character)
        elif event.event_type == 'enter':
            obs_present.add(event.character)

    # Compute characters who are OUT at insert_idx and haven't observed target (ECT-safe for enter/leave pairs)
    all_chars = {'A', 'B', 'C', 'D'}
    chars_out_without_observing = [c for c in sorted(all_chars - present)
                                   if c not in chars_observed_target and c != 'A']

    # Find containers used by events AFTER insert_idx - we must avoid these
    # because our filler events will be inserted BEFORE those events
    containers_used_after = _get_containers_used_after(scenario, insert_idx)

    # Filter non-targets to only those not used after insert_idx
    safe_non_targets = [c for c in non_targets if c not in containers_used_after]

    events_to_add = []
    present_list = sorted(present)
    # Use only safe containers (may be 0, 1, or 2)
    nt0 = safe_non_targets[0] if len(safe_non_targets) > 0 else None
    nt1 = safe_non_targets[1] if len(safe_non_targets) > 1 else None

    # Generate n filler events (puts/moves on non-target containers)
    # Exclude ALL items used in ANY event (including future events in base scenario)
    all_scenario_items = {e.item for e in scenario.events if e.item}
    used_items = all_scenario_items | {contents[c] for c in CONTAINERS_GEN if contents[c] is not None}
    available_items = [i for i in ITEMS_GEN if i not in used_items]
    last_actor = None

    # Track characters who have left WITHOUT observing target (ECT-safe for enter)
    chars_who_left = []

    for _ in range(n):
        actor = _choose_different_actor(present, last_actor, rng)
        last_actor = actor
        added_event = False

        # Try container-based filler first
        if nt0 is not None:
            if nt1 is None:
                # Only one safe container - can only put if empty
                if contents[nt0] is None and available_items:
                    item = available_items.pop(rng.randint(0, len(available_items) - 1))
                    events_to_add.append(Event('put', actor, container=nt0, item=item))
                    contents[nt0] = item
                    added_event = True
            else:
                # Two containers available - alternate between puts and moves
                if contents[nt0] is None and contents[nt1] is None:
                    # Both empty: put an item
                    if available_items:
                        item = available_items.pop(rng.randint(0, len(available_items) - 1))
                        target_container = rng.choice([nt0, nt1])
                        events_to_add.append(Event('put', actor, container=target_container, item=item))
                        contents[target_container] = item
                        added_event = True
                elif contents[nt0] is not None and contents[nt1] is None:
                    # nt0 has item, nt1 empty: move to nt1
                    events_to_add.append(Event('move', actor, from_container=nt0, to_container=nt1, item=contents[nt0]))
                    contents[nt1] = contents[nt0]
                    contents[nt0] = None
                    added_event = True
                elif contents[nt0] is None and contents[nt1] is not None:
                    # nt1 has item, nt0 empty: move to nt0
                    events_to_add.append(Event('move', actor, from_container=nt1, to_container=nt0, item=contents[nt1]))
                    contents[nt0] = contents[nt1]
                    contents[nt1] = None
                    added_event = True

        # Fallback: use leave/enter pairs when container capacity exhausted
        # IMPORTANT: Only use characters who haven't observed target (ECT-safe)
        # to avoid adding ECT #1 (knowledge -> belief) which would cause ECT drift
        if not added_event:
            # Priority 1: bring back someone who left during filler insertion
            # Skip anyone who's already present (may have been brought back via Priority 2)
            while chars_who_left and not added_event:
                char_to_enter = chars_who_left.pop(0)
                if char_to_enter not in present:
                    events_to_add.append(Event('enter', char_to_enter))
                    present.add(char_to_enter)
                    added_event = True
            # Priority 2: bring back someone who is OUT and hasn't observed target
            if not added_event and chars_out_without_observing:
                # Find first character who is NOT currently present
                char_to_enter = None
                for c in list(chars_out_without_observing):
                    if c not in present:
                        char_to_enter = c
                        chars_out_without_observing.remove(c)
                        break
                if char_to_enter:
                    events_to_add.append(Event('enter', char_to_enter))
                    present.add(char_to_enter)
                    chars_who_left.append(char_to_enter)  # Add to chars_who_left so they can leave again
                    added_event = True
            # Priority 3: have someone leave who hasn't observed target
            if not added_event:
                # Have someone leave (not A, must be present, must NOT have observed target)
                leavers = [c for c in sorted(present) if c != 'A' and c not in chars_observed_target]
                if leavers:
                    char_to_leave = rng.choice(leavers)
                    events_to_add.append(Event('leave', char_to_leave))
                    present.discard(char_to_leave)
                    chars_who_left.append(char_to_leave)
                    added_event = True

            # NOTE: We intentionally do NOT use non-ECT-safe fallback here.
            # Using characters who observed the target would add ECT to 1A,
            # potentially violating the ECT(1B) > ECT(1A) constraint.
            # Accept bounded SIT gaps rather than break ECT ordering.

    # If anyone is still out, bring them back before inserting
    # Only add enter for chars who are NOT currently present (they may have been brought back during the loop)
    for char in chars_who_left:
        if char not in present:
            events_to_add.append(Event('enter', char))

    # DEBUG: Validate events before inserting
    debug_present = set(scenario.present_initially)
    for idx, event in enumerate(scenario.events[:insert_idx]):
        if event.event_type == 'leave':
            debug_present.discard(event.character)
        elif event.event_type == 'enter':
            debug_present.add(event.character)

    for evt in events_to_add:
        if evt.event_type == 'enter' and evt.character in debug_present:
            raise ValueError(f"BUG: About to insert enter for {evt.character} but they're in debug_present={debug_present}. "
                           f"chars_who_left={chars_who_left}, chars_out_without_observing={chars_out_without_observing}, "
                           f"present={present}, events_to_add={([(e.event_type, e.character) for e in events_to_add])}")
        elif evt.event_type == 'leave' and evt.character not in debug_present:
            raise ValueError(f"BUG: About to insert leave for {evt.character} but they're not in debug_present={debug_present}")

        if evt.event_type == 'enter':
            debug_present.add(evt.character)
        elif evt.event_type == 'leave':
            debug_present.discard(evt.character)

    for i, evt in enumerate(events_to_add):
        scenario.events.insert(insert_idx + i, evt)


def compute_filler_capacity(scenario: Scenario) -> int:
    """Compute max events that insert_adaptive_filler_events can add.

    Uses same logic as insert_adaptive_filler_events to determine capacity.
    This allows us to predict whether 1A can compensate for 1B's extra events.
    """
    target = scenario.question_container
    non_targets = _non_target_containers(target)

    # Find insert_idx (where A leaves) - same logic as insert_adaptive_filler_events
    a_present = 'A' in scenario.present_initially
    insert_idx = len(scenario.events)
    present = set(scenario.present_initially)
    contents = {c: None for c in CONTAINERS_GEN}

    for idx, event in enumerate(scenario.events):
        if event.event_type == 'put':
            contents[event.container] = event.item
        elif event.event_type == 'move':
            contents[event.to_container] = event.item
            contents[event.from_container] = None
        elif event.event_type == 'leave':
            if event.character == 'A' and a_present:
                insert_idx = idx
                break
            present.discard(event.character)
        elif event.event_type == 'enter':
            present.add(event.character)
            if event.character == 'A':
                a_present = True

    if not present:
        return 0

    # Track who observed target (can't do leave/enter) - same logic
    present = set(scenario.present_initially)
    chars_observed_target = set()
    for idx, event in enumerate(scenario.events):
        if idx >= insert_idx:
            break
        if event.event_type == 'leave':
            present.discard(event.character)
        elif event.event_type == 'enter':
            present.add(event.character)
        if event.event_type == 'put' and event.container == target:
            chars_observed_target.update(present)
        elif event.event_type == 'move' and (event.from_container == target or event.to_container == target):
            chars_observed_target.update(present)

    can_leave_enter = [c for c in present if c not in chars_observed_target and c != 'A']

    # Find safe containers (not used by events after insert_idx)
    containers_used_after = _get_containers_used_after(scenario, insert_idx)
    safe_non_targets = [c for c in non_targets if c not in containers_used_after]

    # Compute capacity based on available resources
    if len(safe_non_targets) >= 2:
        return 99  # Unlimited (put/move cycles between containers)
    elif len(safe_non_targets) == 1:
        return 1 + 2 * len(can_leave_enter)  # 1 put + leave/enter pairs
    else:
        return 2 * len(can_leave_enter)  # Only leave/enter pairs available


def insert_adaptive_filler_events(scenario: Scenario, rng: random.Random, n: int) -> int:
    """Insert additional filler events to balance SIT with paired 1B scenario.

    This function adds N filler events to an already-generated 1A scenario
    (which already has initial filler from insert_filler_events()).

    CRITICAL CONSTRAINT: Only uses non-target containers, no leave/enter pairs.
    This ensures NO ECTs are added. The function validates this invariant.

    Args:
        scenario: The 1A scenario to modify (already has initial filler)
        rng: Random generator for reproducibility
        n: Number of additional filler events to add

    Returns:
        Number of events actually inserted (may be < n if constrained)
    """
    if n <= 0:
        return 0

    target = scenario.question_container
    non_targets = _non_target_containers(target)  # 2 containers

    # Find last position where A is present (same logic as insert_filler_events)
    a_present = 'A' in scenario.present_initially
    insert_idx = len(scenario.events)
    present = set(scenario.present_initially)
    contents = {c: None for c in CONTAINERS_GEN}

    for idx, event in enumerate(scenario.events):
        if event.event_type == 'put':
            contents[event.container] = event.item
        elif event.event_type == 'move':
            contents[event.to_container] = event.item
            contents[event.from_container] = None
        elif event.event_type == 'leave':
            if event.character == 'A' and a_present:
                insert_idx = idx
                break
            present.discard(event.character)
        elif event.event_type == 'enter':
            present.add(event.character)
            if event.character == 'A':
                a_present = True

    if not present:
        return 0  # No actors available

    # Track which characters have observed the target container (up to insert_idx)
    # These characters CANNOT do leave/enter pairs (would trigger ECT #1)
    # Also track characters who left WITHOUT observing - they can do enter/leave pairs
    # CRITICAL: Must reset present to fresh state before this loop
    present = set(scenario.present_initially)
    chars_observed_target = set()
    chars_left_without_observing = []  # Characters who left before seeing target
    for idx, event in enumerate(scenario.events):
        if idx >= insert_idx:
            break
        # Check if target was observed BEFORE updating presence
        if event.event_type == 'put' and event.container == target:
            chars_observed_target.update(present)
        elif event.event_type == 'move' and (event.from_container == target or event.to_container == target):
            chars_observed_target.update(present)
        # Then update presence
        if event.event_type == 'leave':
            if event.character not in chars_observed_target and event.character != 'A':
                chars_left_without_observing.append(event.character)
            present.discard(event.character)
        elif event.event_type == 'enter':
            present.add(event.character)
    # present is now correct at insert_idx

    # Characters who can do leave/enter pairs (present, haven't observed target, not A)
    can_leave_enter = [c for c in present if c not in chars_observed_target and c != 'A']
    # Characters who left without observing AND are not currently present can do enter/leave pairs
    # (they left and didn't come back, so we can bring them back then have them leave)
    can_enter_leave = [c for c in chars_left_without_observing if c not in present]

    # Find containers used by events AFTER insert_idx - we must avoid these
    containers_used_after = _get_containers_used_after(scenario, insert_idx)

    # Filter non-targets to only those not used after insert_idx
    safe_non_targets = [c for c in non_targets if c not in containers_used_after]
    if len(safe_non_targets) < 1:
        return 0  # No safe containers for filler

    # Collect all items already used in scenario (including initial filler)
    all_scenario_items = {e.item for e in scenario.events if e.item}
    used_items = all_scenario_items | {contents[c] for c in CONTAINERS_GEN if contents[c] is not None}
    available_items = [i for i in ITEMS_GEN if i not in used_items]

    events_to_add = []
    present_list = sorted(present)
    # Use only safe containers (may be 1 or 2)
    nt0 = safe_non_targets[0]
    nt1 = safe_non_targets[1] if len(safe_non_targets) > 1 else None

    # Track contents of non-target containers at insertion point
    nt_contents = {nt0: contents[nt0]}
    if nt1 is not None:
        nt_contents[nt1] = contents[nt1]

    last_actor = None
    events_added = 0

    while events_added < n:
        remaining = n - events_added
        # Choose actor (alternate when possible)
        actor = _choose_different_actor(present, last_actor, rng)
        last_actor = actor

        # Handle single-container case (nt1 is None)
        if nt1 is None:
            # Only one safe container - can only put if empty
            if nt_contents[nt0] is None and available_items:
                item = available_items.pop(rng.randint(0, len(available_items) - 1))
                events_to_add.append(Event('put', actor, container=nt0, item=item))
                nt_contents[nt0] = item
                events_added += 1
            elif can_leave_enter and remaining >= 2:
                # Fall back to leave/enter pair (adds 2 events)
                leaver = rng.choice(can_leave_enter)
                events_to_add.append(Event('leave', leaver))
                events_to_add.append(Event('enter', leaver))
                can_leave_enter.remove(leaver)
                events_added += 2
            elif can_enter_leave and remaining >= 2:
                # Bring back someone who left without observing, then have them leave
                enterer = can_enter_leave.pop(0)
                events_to_add.append(Event('enter', enterer))
                events_to_add.append(Event('leave', enterer))
                events_added += 2
            elif can_enter_leave and remaining == 1:
                # Just bring back someone who left without observing (single event)
                enterer = can_enter_leave.pop(0)
                events_to_add.append(Event('enter', enterer))
                events_added += 1
            else:
                break  # Cannot add more events
            continue

        # Two containers available - decide what filler event to add
        if nt_contents[nt0] is None and nt_contents[nt1] is None:
            # Both empty: need to put an item
            if not available_items:
                # No items - try leave/enter pair instead
                if can_leave_enter and remaining >= 2:
                    leaver = rng.choice(can_leave_enter)
                    events_to_add.append(Event('leave', leaver))
                    events_to_add.append(Event('enter', leaver))
                    can_leave_enter.remove(leaver)
                    events_added += 2
                    continue
                elif can_enter_leave and remaining >= 2:
                    enterer = can_enter_leave.pop(0)
                    events_to_add.append(Event('enter', enterer))
                    events_to_add.append(Event('leave', enterer))
                    events_added += 2
                    continue
                elif can_enter_leave and remaining == 1:
                    enterer = can_enter_leave.pop(0)
                    events_to_add.append(Event('enter', enterer))
                    events_added += 1
                    continue
                break  # Cannot add more events
            item = available_items.pop(rng.randint(0, len(available_items) - 1))
            target_container = rng.choice([nt0, nt1])
            events_to_add.append(Event('put', actor, container=target_container, item=item))
            nt_contents[target_container] = item
            events_added += 1
        elif nt_contents[nt0] is not None and nt_contents[nt1] is None:
            # nt0 has item, nt1 empty: move to nt1
            events_to_add.append(Event('move', actor, from_container=nt0, to_container=nt1, item=nt_contents[nt0]))
            nt_contents[nt1] = nt_contents[nt0]
            nt_contents[nt0] = None
            events_added += 1
        elif nt_contents[nt0] is None and nt_contents[nt1] is not None:
            # nt1 has item, nt0 empty: move to nt0
            events_to_add.append(Event('move', actor, from_container=nt1, to_container=nt0, item=nt_contents[nt1]))
            nt_contents[nt0] = nt_contents[nt1]
            nt_contents[nt1] = None
            events_added += 1
        else:
            # Both have items: can't add put/move without overflow
            # Try leave/enter pair for a character who hasn't observed target
            if can_leave_enter and remaining >= 2:
                leaver = rng.choice(can_leave_enter)
                events_to_add.append(Event('leave', leaver))
                events_to_add.append(Event('enter', leaver))
                can_leave_enter.remove(leaver)
                events_added += 2
            elif can_enter_leave and remaining >= 2:
                enterer = can_enter_leave.pop(0)
                events_to_add.append(Event('enter', enterer))
                events_to_add.append(Event('leave', enterer))
                events_added += 2
            elif can_enter_leave and remaining == 1:
                enterer = can_enter_leave.pop(0)
                events_to_add.append(Event('enter', enterer))
                events_added += 1
            else:
                break  # Cannot add more events

    # Insert all events at the insertion point
    for i, evt in enumerate(events_to_add):
        scenario.events.insert(insert_idx + i, evt)

    return len(events_to_add)


def apply_name_variation(scenario: Scenario, rng: random.Random) -> None:
    """Apply cosmetic variations to differentiate Extra=1 from its paired Extra=0.

    - 50% chance: swap all bag<->box references
    - Always: change queried item to a different item name

    This makes paired scenarios look different on the surface while preserving
    the same underlying epistemic structure.
    """
    # 50% chance: swap container names
    if rng.random() < 0.5:
        swap_map = {'bag': 'box', 'box': 'bag'}
        scenario.question_container = swap_map.get(scenario.question_container, scenario.question_container)
        for event in scenario.events:
            if event.container:
                event.container = swap_map.get(event.container, event.container)
            if event.from_container:
                event.from_container = swap_map.get(event.from_container, event.from_container)
            if event.to_container:
                event.to_container = swap_map.get(event.to_container, event.to_container)

    # Always: change the queried item to a different item
    # Find all items currently used in the scenario and the queried item
    all_items_used = set()
    old_item = None
    for event in scenario.events:
        if event.item:
            all_items_used.add(event.item)
        if old_item is None:
            if event.event_type == 'put' and event.container == scenario.question_container:
                old_item = event.item
            elif event.event_type == 'move' and event.to_container == scenario.question_container:
                old_item = event.item

    if old_item is not None:
        # Exclude all items already in use to avoid duplicates
        available_items = [i for i in ITEMS_GEN if i not in all_items_used]
        if available_items:
            new_item = rng.choice(available_items)
            for event in scenario.events:
                if event.item == old_item:
                    event.item = new_item


def generate_scenarios_from_tuples(specs: List[SpecTuple], outfile: str, seed: Optional[int] = None, chartypes: List[CharacterType] = [CharacterType.LIVE_PLAYER, CharacterType.HONEST_OPPONENT, CharacterType.DISHONEST_TEAMMATE, CharacterType.DISHONEST_OPPONENT, CharacterType.NEUTRAL]) -> None:
    """Generate scenarios using shared-base approach.

    For each scenario ID, all Extra variants (0A, 0B, 1A, 1B) are generated from the SAME base:
    1. Generate base scenario from 0A spec (or first available)
    2. Deep-copy base for each Extra variant
    3. Apply appropriate filler/extra events:
       - 0A: insert_n_filler_events(n=EXTRA_0A_FILLER) - minimal
       - 0B: insert_n_filler_events(n=EXTRA_0B_FILLER) - higher load
       - 1A: insert_filler_events() - SIT parity with 1B
       - 1B: apply_name_variation() + insert_extra_events_*() - adds ECT

    See EXTRA_MAPPING.md for full documentation.

    Output order: ID1-0A, ID1-0B, ID1-1A, ID1-1B, ID2-0A, ...
    """
    rng = random.Random(seed)
    scenarios: List[Scenario] = []
    chars = _map_chartypes_to_names(chartypes)
    acting_chars = [c for c in chars if c != 'N']

    # Tracking for violations
    violation_stats = {'total': 0, 'by_id': defaultdict(int), 'by_extra': defaultdict(int)}
    sit_gap_stats = {'total': 0, 'by_id': {}}

    # Group specs by ID and Extra value (string keys: '0A', '0B', '1A', '1B')
    spec_by_id: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for spec in specs:
        spec_by_id[spec['Id']][str(spec['Extra'])] = spec

    # Preserve original ID order (as they first appear in the CSV)
    seen_ids: List[str] = []
    for spec in specs:
        if spec['Id'] not in seen_ids:
            seen_ids.append(spec['Id'])

    scenario_idx = 0
    for scenario_id in seen_ids:
        e0a_spec = spec_by_id[scenario_id].get('0A')
        e0b_spec = spec_by_id[scenario_id].get('0B')
        e1a_spec = spec_by_id[scenario_id].get('1A')
        e1b_spec = spec_by_id[scenario_id].get('1B')

        # --- Case: No specs at all (shouldn't happen) ---
        base_spec = e0a_spec or e0b_spec or e1a_spec or e1b_spec
        if base_spec is None:
            continue

        # --- Case: At least one spec exists (generate via shared base) ---
        row = base_spec
        actor = _map_chartypes_to_names([row['Actor']])[0]
        answerer = actor if row['Answerer'] == 'Self' else (_teammate_of(actor) if row['Answerer'] == 'Teammate' else _opponent_of(actor, rng))
        available: Set[str] = set(acting_chars)
        queried_container = rng.choice(CONTAINERS_GEN)
        queried_item = rng.choice(ITEMS_GEN)

        sb = Scenario_Builder(rng, queried_container, queried_item, available)
        sb.plan_availability(row, answerer)
        sb.build_scenario(answerer)

        present_initially = sorted(list(sb.present_initially))

        # Create base scenario (before filler or extra events)
        base_scenario = Scenario(
            round_num=(scenario_idx // len(acting_chars)) + 1,
            whose_turn=actor,
            who_answers=answerer,
            ks_self=row['KS_Self'].value,
            ks_teammate=row['KS_Teammate'].value,
            ks_opponent=row['KS_Opponent'].value,
            correct_action=row['Action'],
            question_container=queried_container,
            events=sb.events,
            present_initially=present_initially,
            id=row['Id'],
            extra='0A',  # Will be overwritten per-variant
        )

        # Validate base scenario before applying any filler
        _validate_invariants(base_scenario)

        # --- Extra=0A: Minimal filler (n=EXTRA_0A_FILLER) ---
        if e0a_spec is not None:
            scenario_e0a = copy.deepcopy(base_scenario)
            scenario_e0a.extra = '0A'
            filler_seed = int(f"{seed or 0}{scenario_idx:03d}{int(row['Id']):04d}0")
            filler_rng = random.Random(filler_seed)
            insert_n_filler_events(scenario_e0a, filler_rng, EXTRA_0A_FILLER)

            _validate_invariants(scenario_e0a)
            v = _validate_teammate_belief_integrity(scenario_e0a)
            if v > 0:
                violation_stats['total'] += v
                violation_stats['by_id'][row['Id']] += v
                violation_stats['by_extra']['0A'] += v
            scenario_e0a.situation_event_count = _count_visible_events(scenario_e0a)
            scenario_e0a.epistemic_transitions = count_epistemic_category_transitions(scenario_e0a)

            if not scenario_e0a.events:
                raise ValueError(f"Generated scenario {scenario_e0a.id} Extra={scenario_e0a.extra} has no events")

            # Epistemic state validation
            epistemic_errors = validate_scenario(scenario_e0a, e0a_spec)
            if epistemic_errors:
                raise ValueError(f"ID={row['Id']} Extra=0A epistemic mismatch: {epistemic_errors}")

            scenarios.append(scenario_e0a)
            scenario_idx += 1

        # --- Extra=0B: Higher load filler (n=EXTRA_0B_FILLER) ---
        if e0b_spec is not None:
            scenario_e0b = copy.deepcopy(base_scenario)
            scenario_e0b.extra = '0B'
            scenario_e0b.round_num = (scenario_idx // len(acting_chars)) + 1
            filler_seed = int(f"{seed or 0}{scenario_idx:03d}{int(row['Id']):04d}1")
            filler_rng = random.Random(filler_seed)
            insert_n_filler_events(scenario_e0b, filler_rng, EXTRA_0B_FILLER)

            _validate_invariants(scenario_e0b)
            v = _validate_teammate_belief_integrity(scenario_e0b)
            if v > 0:
                violation_stats['total'] += v
                violation_stats['by_id'][row['Id']] += v
                violation_stats['by_extra']['0B'] += v
            scenario_e0b.situation_event_count = _count_visible_events(scenario_e0b)
            scenario_e0b.epistemic_transitions = count_epistemic_category_transitions(scenario_e0b)

            if not scenario_e0b.events:
                raise ValueError(f"Generated scenario {scenario_e0b.id} Extra={scenario_e0b.extra} has no events")

            # Epistemic state validation
            epistemic_errors = validate_scenario(scenario_e0b, e0b_spec)
            if epistemic_errors:
                raise ValueError(f"ID={row['Id']} Extra=0B epistemic mismatch: {epistemic_errors}")

            scenarios.append(scenario_e0b)
            scenario_idx += 1

        # --- Extra=1A: Will be generated AFTER 1B to match its SIT exactly ---
        # Store placeholder values; actual 1A generation happens after 1B
        e1a_scenario_idx = scenario_idx if e1a_spec is not None else None
        e1a_round_num = (scenario_idx // len(acting_chars)) + 1 if e1a_spec is not None else None
        if e1a_spec is not None:
            scenario_idx += 1  # Reserve index for 1A

        # Placeholder for 1A (will be set after 1B generation)
        scenario_e1a = None
        e1a_sit = None
        e1a_ect = None

        # --- Extra=1B: Was Extra=1 - apply variation + extra events (adds ECT) ---
        # Generate 1B first, then generate 1A to match its SIT exactly
        if e1b_spec is not None:
            MAX_RETRIES = 10

            for retry in range(MAX_RETRIES):
                scenario_e1b = copy.deepcopy(base_scenario)
                scenario_e1b.extra = '1B'
                scenario_e1b.round_num = (scenario_idx // len(acting_chars)) + 1

                # Apply cosmetic name variation (container swap, item change)
                variation_seed = int(f"{seed or 0}{scenario_idx:03d}{int(row['Id']):04d}99")
                variation_rng = random.Random(variation_seed)
                apply_name_variation(scenario_e1b, variation_rng)

                # Insert extra events (adds ECT) - unchanged behavior from old Extra=1
                retry_rng = random.Random(rng.randint(0, 2**31) + retry)
                base_event_count = len(scenario_e1b.events)
                answerer_state = _get_answerer_state(e1b_spec)

                # Constrain 1B's SIT based on 1A's filler capacity
                # This ensures 1A can always match 1B within MAX_SIT_GAP
                MAX_SIT_GAP = 3
                skip_opt = False
                skip_teammate_reentry = False

                if e1a_spec is not None:
                    filler_cap = compute_filler_capacity(base_scenario)
                    base_sit = _count_visible_events(base_scenario)
                    max_allowed_1b_sit = base_sit + filler_cap + MAX_SIT_GAP

                    # Check if teammate reentry applies (for any answerer_state)
                    teammate_reentry_applies = (e1b_spec['KS_Self'] == EpistemicState.BELIEVES_X and
                                               e1b_spec['KS_Teammate'] == EpistemicState.UNKNOWN)
                    teammate_reentry_events = 2 if teammate_reentry_applies else 0

                    # Project 1B's SIT and determine if we need to skip optional events
                    current_1b_sit = _count_visible_events(scenario_e1b)

                    if answerer_state == EpistemicState.BELIEVES_X:
                        # BELIEVES_X: mandatory = enter + 2 moves + leave = 4 SIT
                        mandatory_events = 4
                        optional_events = 2
                        projected_with_all = current_1b_sit + mandatory_events + teammate_reentry_events + optional_events
                        skip_opt = projected_with_all > max_allowed_1b_sit
                        projected_without_opt = current_1b_sit + mandatory_events + teammate_reentry_events
                        skip_teammate_reentry = projected_without_opt > max_allowed_1b_sit
                    elif answerer_state == EpistemicState.KNOWS_TRUTH:
                        # KNOWS_TRUTH: mandatory = leave + move_away + enter + revelation = 4 SIT
                        mandatory_events = 4
                        optional_events = 1
                        projected_with_all = current_1b_sit + mandatory_events + teammate_reentry_events + optional_events
                        skip_opt = projected_with_all > max_allowed_1b_sit
                        projected_without_opt = current_1b_sit + mandatory_events + teammate_reentry_events
                        skip_teammate_reentry = projected_without_opt > max_allowed_1b_sit

                try:
                    if (e1b_spec['Answerer'] == 'Teammate' and
                        e1b_spec['KS_Self'] == EpistemicState.KNOWS_X and
                        e1b_spec['KS_Teammate'] == EpistemicState.BELIEVES_TRUTH):
                        insert_extra_puts(scenario_e1b, answerer, retry_rng, spec=e1b_spec)
                        _validate_invariants(scenario_e1b)
                    elif answerer_state == EpistemicState.KNOWS_TRUTH:
                        insert_extra_events_with_revelation(scenario_e1b, answerer, retry_rng, spec=e1b_spec,
                                                           skip_optional_events=skip_opt)
                        _validate_invariants(scenario_e1b)
                    elif answerer_state == EpistemicState.BELIEVES_TRUTH:
                        insert_extra_events_believes_true(scenario_e1b, answerer, retry_rng, spec=e1b_spec)
                        _validate_invariants(scenario_e1b)
                    elif answerer_state == EpistemicState.BELIEVES_FALSE:
                        insert_extra_events_believes_false(scenario_e1b, answerer, retry_rng, spec=e1b_spec)
                        _validate_invariants(scenario_e1b)
                    elif answerer_state == EpistemicState.BELIEVES_X:
                        # skip_opt computed above before try block
                        insert_extra_events_believes_x(scenario_e1b, answerer, retry_rng,
                                                       skip_optional_events=skip_opt, spec=e1b_spec)
                        _validate_invariants(scenario_e1b)
                    elif answerer_state == EpistemicState.UNKNOWN:
                        insert_extra_events_unknown(scenario_e1b, answerer, retry_rng, spec=e1b_spec)
                        _validate_invariants(scenario_e1b)
                    else:
                        insert_extra_events(scenario_e1b, answerer, actor, answerer_state, e1b_spec, retry_rng)
                        _validate_invariants(scenario_e1b)
                except ValueError as e:
                    raise ValueError(f"ID {row['Id']} Extra=1B answerer_state={answerer_state}: {e}")

                # For Self=BELIEVES_X + Teammate=UNKNOWN scenarios, add teammate's re-entry
                # Skip if it would cause SIT gap violation (teammate stays UNKNOWN either way)
                if (e1b_spec['KS_Self'] == EpistemicState.BELIEVES_X and
                    e1b_spec['KS_Teammate'] == EpistemicState.UNKNOWN and
                    not skip_teammate_reentry):
                    teammate = _teammate_of(actor)
                    insert_teammate_reentry_unknown(scenario_e1b, teammate, retry_rng)

                # Check if extra events were actually added
                if len(scenario_e1b.events) <= base_event_count:
                    # No events added - function returned early due to constraints
                    # This is expected when containers are used by later base events
                    # Continue to next retry
                    continue

                _validate_invariants(scenario_e1b)
                v = _validate_teammate_belief_integrity(scenario_e1b)
                if v > 0:
                    violation_stats['total'] += v
                    violation_stats['by_id'][row['Id']] += v
                    violation_stats['by_extra']['1B'] += v

                # Validate epistemic states are preserved after extra events
                epistemic_preservation_errors = _validate_epistemic_states_preserved(scenario_e1b, e1b_spec)
                if epistemic_preservation_errors:
                    # Retry if epistemic states were violated
                    continue

                scenario_e1b.situation_event_count = _count_visible_events(scenario_e1b)
                scenario_e1b.epistemic_transitions = count_epistemic_category_transitions(scenario_e1b)

                # Successfully generated 1B - exit retry loop
                break

            # Check if we successfully added events (scenario has more events than base)
            base_event_count_final = len(base_scenario.events)
            if len(scenario_e1b.events) <= base_event_count_final:
                # All retries failed to add events - skip this 1B scenario
                import warnings
                warnings.warn(
                    f"ID {row['Id']} Extra=1B: Could not add extra events after {MAX_RETRIES} retries "
                    f"(containers used by later events). Skipping 1B variant."
                )
                continue

            # Epistemic state validation (double-check with existing validation)
            epistemic_errors = validate_scenario(scenario_e1b, e1b_spec)
            if epistemic_errors:
                raise ValueError(f"ID={row['Id']} Extra=1B epistemic mismatch: {epistemic_errors}")

            scenarios.append(scenario_e1b)
            scenario_idx += 1

            # --- Generate 1A with exact filler to match 1B's SIT ---
            if e1a_spec is not None and e1a_scenario_idx is not None:
                e1b_sit = scenario_e1b.situation_event_count
                base_sit = _count_visible_events(base_scenario)
                filler_count = e1b_sit - base_sit  # Exact filler needed to match 1B

                scenario_e1a = copy.deepcopy(base_scenario)
                scenario_e1a.extra = '1A'
                scenario_e1a.round_num = e1a_round_num

                filler_seed = int(f"{seed or 0}{e1a_scenario_idx:03d}{int(row['Id']):04d}2")
                filler_rng = random.Random(filler_seed)

                # Use insert_n_filler_events with exact count to match 1B's SIT
                insert_n_filler_events(scenario_e1a, filler_rng, max(0, filler_count))

                _validate_invariants(scenario_e1a)
                v = _validate_teammate_belief_integrity(scenario_e1a)
                if v > 0:
                    violation_stats['total'] += v
                    violation_stats['by_id'][row['Id']] += v
                    violation_stats['by_extra']['1A'] += v
                scenario_e1a.situation_event_count = _count_visible_events(scenario_e1a)
                scenario_e1a.epistemic_transitions = count_epistemic_category_transitions(scenario_e1a)

                if not scenario_e1a.events:
                    raise ValueError(f"Generated scenario {scenario_e1a.id} Extra={scenario_e1a.extra} has no events")

                # Epistemic state validation
                epistemic_errors = validate_scenario(scenario_e1a, e1a_spec)
                if epistemic_errors:
                    raise ValueError(f"ID={row['Id']} Extra=1A epistemic mismatch: {epistemic_errors}")

                # Validate: Extra=1B must have more ECTs than Extra=1A
                e1a_ect = scenario_e1a.epistemic_transitions['total']
                e1b_ect = scenario_e1b.epistemic_transitions['total']
                if e1b_ect <= e1a_ect:
                    raise ValueError(
                        f"ID {row['Id']} Extra=1B has ECT={e1b_ect} which is not greater than "
                        f"Extra=1A ECT={e1a_ect}. Extra=1B must add epistemic complexity."
                    )

                # Insert 1A at its reserved position (before 1B)
                # Find where 1B was inserted and insert 1A before it
                scenarios.insert(len(scenarios) - 1, scenario_e1a)

                # Track SIT gap violations (gap > MAX_SIT_GAP)
                final_1a_sit = scenario_e1a.situation_event_count
                final_1b_sit = scenario_e1b.situation_event_count
                final_gap = abs(final_1b_sit - final_1a_sit)
                if final_gap > MAX_SIT_GAP:  # Only track violations
                    sit_gap_stats['total'] += 1
                    sid = row['Id']
                    if sid not in sit_gap_stats['by_id']:
                        sit_gap_stats['by_id'][sid] = {'count': 0, 'max_gap': 0, 'sits_1a': [], 'sits_1b': []}
                    sit_gap_stats['by_id'][sid]['count'] += 1
                    sit_gap_stats['by_id'][sid]['sits_1a'].append(final_1a_sit)
                    sit_gap_stats['by_id'][sid]['sits_1b'].append(final_1b_sit)
                    if final_gap > sit_gap_stats['by_id'][sid]['max_gap']:
                        sit_gap_stats['by_id'][sid]['max_gap'] = final_gap

        # --- Fallback: Generate 1A if there's no 1B spec (standalone 1A) ---
        if e1a_spec is not None and e1b_spec is None and e1a_scenario_idx is not None:
            scenario_e1a = copy.deepcopy(base_scenario)
            scenario_e1a.extra = '1A'
            scenario_e1a.round_num = e1a_round_num

            filler_seed = int(f"{seed or 0}{e1a_scenario_idx:03d}{int(row['Id']):04d}2")
            filler_rng = random.Random(filler_seed)
            insert_filler_events(scenario_e1a, filler_rng)  # Use standard filler

            _validate_invariants(scenario_e1a)
            v = _validate_teammate_belief_integrity(scenario_e1a)
            if v > 0:
                violation_stats['total'] += v
                violation_stats['by_id'][row['Id']] += v
                violation_stats['by_extra']['1A'] += v
            scenario_e1a.situation_event_count = _count_visible_events(scenario_e1a)
            scenario_e1a.epistemic_transitions = count_epistemic_category_transitions(scenario_e1a)

            if not scenario_e1a.events:
                raise ValueError(f"Generated scenario {scenario_e1a.id} Extra={scenario_e1a.extra} has no events")

            # Epistemic state validation
            epistemic_errors = validate_scenario(scenario_e1a, e1a_spec)
            if epistemic_errors:
                raise ValueError(f"ID={row['Id']} Extra=1A epistemic mismatch: {epistemic_errors}")

            scenarios.append(scenario_e1a)

    # Final safeguard: ensure we generated scenarios if specs were provided
    if specs and not scenarios:
        raise ValueError(
            f"No scenarios generated from {len(specs)} specs. "
            f"Specs: {[{'Id': s['Id'], 'Extra': s['Extra']} for s in specs]}"
        )

    save_scenarios(scenarios, outfile, chars, chartypes)

    return {'violations': violation_stats, 'sit_gaps': sit_gap_stats}


def validate_scenario_file(filepath: str) -> dict:
    """
    Validate all scenarios in a JSON file for item consistency and other invariants.

    Args:
        filepath: Path to the JSON scenario file to validate.

    Returns:
        dict with keys:
            - 'total_scenarios': Number of scenarios in file
            - 'errors': List of error dicts with 'id', 'extra', 'error' keys
            - 'valid': True if no errors found
    """
    scenarios, _, _ = load_scenarios(filepath)
    errors = []
    for s in scenarios:
        try:
            _validate_invariants(s)
        except ValueError as e:
            errors.append({'id': s.id, 'extra': s.extra, 'error': str(e)})

    return {
        'total_scenarios': len(scenarios),
        'errors': errors,
        'valid': len(errors) == 0
    }


if __name__ == "__main__":
    specs = read_specs_from_csv('ToM - scenarios.csv')
    outfile = 'scenarios_generated4.json'
    chartypes = [CharacterType.LIVE_PLAYER, CharacterType.DISHONEST_OPPONENT, CharacterType.DISHONEST_TEAMMATE, CharacterType.DISHONEST_OPPONENT]
    generate_scenarios_from_tuples(specs, outfile, seed=None, chartypes=chartypes)
    print(f"Created {outfile} with auto-generated scenarios")
