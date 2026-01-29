import random
import copy
from collections import defaultdict
from tom_helpers import (
    Scenario, Event, EpistemicState, CharacterType,
    save_scenarios, SpecTuple, read_specs_from_csv
)
from typing import List, Optional, Tuple, Set, Dict
from dataclasses import dataclass

ITEMS_GEN = ['apple', 'ball', 'banana', 'brick', 'stapler', 'orange']
CONTAINERS_GEN = ['bag', 'box', 'basket']

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
        self.exclude: Set[str] = set()      # who must leave
        self.exclude_true: Set[str] = set() # who must leave believing something that matches the end queried_item/container state
        self.exclude_false: Set[str] = set() # who must leave believing something that matches the end queried_item/container state
        self.include: Set[str] = set()      # who must be present at end
        self.present_initially: Set[str] = set()  # who must be present initially
        self.must_leave_together: Tuple[Optional[str], Optional[str]] = (None, None)  # (char1, char2) must be in same group
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
                    self.exclude.add(opponent1)
                    self.exclude.add(opponent2)
                elif spec.get('Answerer') == 'Opponent':
                    # ensure the chosen opponent (the one who must answer) is exclu
                    self.exclude.add(answerer)
                else:
                    self.exclude.add(self.rng.choice([opponent1, opponent2]))

            elif spec['KS_Teammate'] == EpistemicState.UNKNOWN and spec['KS_Opponent'] == EpistemicState.KNOWS_TRUTH:
                self.exclude.add(teammate)
                if spec['Answerer'] == 'Teammate':
                    self.must_leave_together = (teammate, actor)
                if spec['Answerer'] == 'Self':
                    self.include.add(opponent1)
                    self.include.add(opponent2)
                elif spec.get('Answerer') == 'Opponent':
                    # ensure the chosen opponent (the one who must answer) is present
                    self.include.add(answerer)
                else:
                    self.include.add(self.rng.choice([opponent1, opponent2])) 

            elif spec['KS_Teammate'] == EpistemicState.UNKNOWN and spec['KS_Opponent'] == EpistemicState.UNKNOWN:
                self.exclude.add(teammate)
                if spec['Answerer'] == 'Teammate':
                    self.must_leave_together = (teammate, actor)
                # Ensure one opponent stays, one leaves
                if spec.get('Answerer') == 'Opponent':
                    leave_opponent = answerer
                    stay_opponent = opponent1 if leave_opponent == opponent2 else opponent2
                else:
                    stay_opponent = self.rng.choice([opponent1, opponent2])
                    leave_opponent = opponent2 if stay_opponent == opponent1 else opponent1
                self.include.add(stay_opponent)  # This opponent must stay until end
                self.exclude.add(leave_opponent)  # This opponent must leave

        else: # spec['KS_Self'] == EpistemicState.KNOWS_X:
            self.include.add(actor) 
            if spec['KS_Teammate'] == EpistemicState.BELIEVES_TRUTH and spec['KS_Opponent'] == EpistemicState.BELIEVES_TRUTH:
                self.exclude_true.add(teammate)
                if spec.get('Answerer') == 'Opponent':
                    self.exclude_true.add(answerer) 
                else:
                    self.exclude_true.add(self.rng.choice([opponent1, opponent2]))
            elif spec['KS_Teammate'] == EpistemicState.BELIEVES_TRUTH and spec['KS_Opponent'] == EpistemicState.BELIEVES_FALSE:
                self.exclude_true.add(teammate)
                if spec['Answerer'] == 'Opponent':
                    self.exclude_false.add(opponent1)
                    self.exclude_false.add(opponent2)
                else:
                    self.exclude_false.add(self.rng.choice([opponent1, opponent2]))
            elif spec['KS_Teammate'] == EpistemicState.BELIEVES_TRUTH and spec['KS_Opponent'] == EpistemicState.KNOWS_TRUTH:
                self.exclude_true.add(teammate)
                if spec.get('Answerer') == 'Opponent':
                    self.include.add(answerer)
                else:
                    self.include.add(self.rng.choice([opponent1, opponent2]))
            elif spec['KS_Teammate'] == EpistemicState.BELIEVES_FALSE and spec['KS_Opponent'] == EpistemicState.BELIEVES_TRUTH:
                self.exclude_false.add(teammate)
                if spec['Answerer'] == 'Teammate':
                    self.exclude_true.add(opponent1)
                    self.exclude_true.add(opponent2)
                elif spec.get('Answerer') == 'Opponent':
                    self.exclude_true.add(answerer)
                else:
                    self.exclude_true.add(self.rng.choice([opponent1, opponent2]))
            elif spec['KS_Teammate'] == EpistemicState.BELIEVES_FALSE and spec['KS_Opponent'] == EpistemicState.BELIEVES_FALSE:
                self.exclude_false.add(teammate)
                if spec['Answerer'] == 'Opponent':
                    self.exclude_false.add(answerer)
                else:
                    self.exclude_false.add(self.rng.choice([opponent1, opponent2])) #need to keep one around to do the move
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

        self.present_initially = self.exclude | self.exclude_true | self.exclude_false | self.include # who must be present initially

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
            for who in order_self_first(self.exclude_false):
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

    for event in scenario.events:
        if event.event_type == 'leave':
            char = event.character
            if has_observed[char] and has_knowledge[char]:
                certainty_count += 1  # #1: knowledge → belief
                has_knowledge[char] = False
            present.discard(char)

        elif event.event_type == 'enter':
            present.add(event.character)

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
            contents[e.to_container] = e.item
            contents[e.from_container] = None
        elif e.event_type == 'enter':
            if e.character in present:
                raise ValueError(f"Event {idx}: {e.character} entered but was already present.")
            present.add(e.character)
        elif e.event_type == 'remove':
            # 'remove' events should never be generated - use moves to third container instead
            raise ValueError(f"Event {idx}: 'remove' event found - these should be eliminated. Use moves instead.")


def _validate_teammate_belief_integrity(s: 'Scenario') -> None:
    """
    Ensure blue team members (A/B) don't invalidate each other's beliefs about the target.

    Rule: After A or B leaves with a belief about the target container,
    the other blue team member cannot modify the target container.

    Red team (C/D) opponents CAN invalidate each other's beliefs - this constraint
    only applies to the player's team (A/B) since that's what we're testing.
    """
    target = s.question_container
    BLUE = {'A', 'B'}

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
                                # Log warning instead of failing - some specs have inherent conflicts
                                import warnings
                                warnings.warn(
                                    f"Teammate belief violation at event {idx}: "
                                    f"{actor} changed {target} to '{target_contents}', "
                                    f"invalidating teammate {departed_char}'s belief ('{believed_contents}')"
                                )


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


def insert_extra_events_with_revelation(scenario: Scenario, answerer: str, rng: random.Random) -> None:
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

    # Find the initial put to queried container
    initial_put_idx = None
    initial_item = None
    for idx, event in enumerate(scenario.events):
        if event.event_type == 'put' and event.container == queried:
            initial_put_idx = idx
            initial_item = event.item
            break

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

    # 1. Answerer leaves
    leave_event = Event('leave', answerer)

    # Exclude teammates of characters who left with beliefs about the target
    # Note: we use leave_pos + 1 because the answerer's leave will be inserted first
    teammates_to_exclude = _get_teammates_to_exclude_for_target_moves(scenario, leave_pos)
    # Also need to add answerer's teammate since answerer is about to leave with a belief
    teammates_to_exclude.add(_teammate_of(answerer))
    valid_movers = [c for c in sorted(present_after_leave) if c not in teammates_to_exclude]
    if not valid_movers:
        return  # Cannot insert extra events without violating teammate beliefs

    # 2. Someone moves item to other container while answerer is away
    mover1 = rng.choice(valid_movers)
    move_away_event = Event('move', mover1, from_container=queried, to_container=other, item=initial_item)

    # 3. Answerer returns
    enter_event = Event('enter', answerer)

    # 4. Revelation: answerer witnesses item being placed in queried container
    # 50% chance: use a DIFFERENT item (adds variety without affecting certainty ECTs)
    # The answerer's certainty transitions (knowledge ↔ belief) depend on presence/absence,
    # not on which specific item is in the container.
    valid_movers2 = [c for c in valid_movers if c != mover1]
    mover2 = rng.choice(valid_movers2) if valid_movers2 else mover1
    if rng.random() < 0.5:
        # Variation: put a different item in queried container
        # Original item stays in 'other', new item goes in 'queried'
        new_item = rng.choice([i for i in ITEMS_GEN if i != initial_item])
        revelation_event = Event('put', mover2, container=queried, item=new_item)
    else:
        # Standard: move original item back
        revelation_event = Event('move', mover2, from_container=other, to_container=queried, item=initial_item)

    # Build list of events to insert
    events_to_insert = [leave_event]

    # 50% chance: have another character leave while answerer is away
    # (must keep at least one person present for the moves)
    # Also exclude any character who has events later in the base scenario
    # AND exclude teammates of characters who modify the target later (to prevent belief violations)
    if rng.random() < 0.5 and len(present_after_leave) > 1:
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

        potential_leavers = [c for c in sorted(present_after_leave)
                            if c != mover1 and c != mover2 and c not in chars_acting_later
                            and c not in teammates_of_modifiers]
        if potential_leavers:
            leaver = rng.choice(potential_leavers)
            events_to_insert.append(Event('leave', leaver))

    events_to_insert.append(move_away_event)
    events_to_insert.append(enter_event)
    events_to_insert.append(revelation_event)

    # Insert all events at the correct position
    for i, evt in enumerate(events_to_insert):
        scenario.events.insert(leave_pos + i, evt)


def insert_extra_events_believes_true(scenario: Scenario, answerer: str, rng: random.Random) -> None:
    """
    Insert extra events for BELIEVES_TRUE answerers.
    Pattern: Answerer sees put, leaves, then state changes and reverts while they're away.
    Result: Their belief (from before leaving) is still true.
    """
    queried = scenario.question_container
    other = _other_container(queried)

    # Find where answerer leaves
    answerer_leave_idx = None
    present = set(scenario.present_initially)
    queried_item = None

    for idx, event in enumerate(scenario.events):
        if event.event_type == 'put' and event.container == queried:
            queried_item = event.item
        elif event.event_type == 'leave':
            if event.character == answerer:
                answerer_leave_idx = idx
                break
            present.discard(event.character)
        elif event.event_type == 'enter':
            present.add(event.character)

    if answerer_leave_idx is None or queried_item is None:
        return

    # After answerer leaves, we need someone to do the moves
    present.discard(answerer)
    if not present:
        return

    # Insert events right after answerer leaves:
    # 1. Move item to other container
    # 2. Move item back to queried container
    # Result: Truth unchanged, but more complex sequence
    insert_pos = answerer_leave_idx + 1

    # Exclude teammates of characters who left with beliefs about the target
    teammates_to_exclude = _get_teammates_to_exclude_for_target_moves(scenario, insert_pos)
    valid_movers = [c for c in sorted(present) if c not in teammates_to_exclude]
    if not valid_movers:
        return  # Cannot insert extra events without violating teammate beliefs

    mover1 = rng.choice(valid_movers)
    move_away = Event('move', mover1, from_container=queried, to_container=other, item=queried_item)

    # Prefer different mover than mover1, still respecting teammate exclusions
    valid_movers2 = [c for c in valid_movers if c != mover1]
    mover2 = rng.choice(valid_movers2) if valid_movers2 else mover1
    move_back = Event('move', mover2, from_container=other, to_container=queried, item=queried_item)

    # Build list of events to insert
    events_to_insert = []

    # Find characters who act later in the base scenario (can't make them leave)
    # AND find characters who modify the target later (their teammates shouldn't leave with beliefs)
    chars_acting_later = set()
    chars_modifying_target_later = set()
    for evt in scenario.events[insert_pos:]:
        chars_acting_later.add(evt.character)
        if evt.event_type == 'put' and evt.container == queried:
            chars_modifying_target_later.add(evt.character)
        elif evt.event_type == 'move' and (evt.from_container == queried or evt.to_container == queried):
            chars_modifying_target_later.add(evt.character)

    # Teammates of target modifiers shouldn't leave (would form beliefs that get violated)
    teammates_of_modifiers = {_teammate_of(c) for c in chars_modifying_target_later if c in {'A', 'B', 'C', 'D'}}

    # 50% chance: have another character leave while answerer is away
    # (must keep at least one person present for the moves)
    if rng.random() < 0.5 and len(present) > 1:
        potential_leavers = [c for c in sorted(present)
                            if c != mover1 and c != mover2 and c not in chars_acting_later
                            and c not in teammates_of_modifiers]
        if potential_leavers:
            leaver = rng.choice(potential_leavers)
            events_to_insert.append(Event('leave', leaver))
            present.discard(leaver)  # Track that they left for mandatory leave/enter below

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
    potential = [c for c in sorted(present) if c != mover1 and c != mover2 and c not in chars_acting_later]
    if potential and rng.random() < 0.5:
        opp = rng.choice(potential)
        events_to_insert.append(Event('leave', opp))
        events_to_insert.append(Event('enter', opp))

    events_to_insert.append(Event('leave', answerer))

    # Insert all events at the correct position
    for i, evt in enumerate(events_to_insert):
        scenario.events.insert(insert_pos + i, evt)


def insert_extra_events_believes_false(scenario: Scenario, answerer: str, rng: random.Random) -> None:
    """
    Insert extra events for BELIEVES_FALSE answerers.
    Pattern: Answerer sees wrong item, leaves, then more events happen while they're away.
    Result: Their belief is still false, but sequence is more complex.

    We insert events right after the answerer leaves: move item to other container, then back.
    """
    queried = scenario.question_container
    other = _other_container(queried)

    # Find where answerer leaves and track container contents at that point
    answerer_leave_idx = None
    contents = {c: None for c in CONTAINERS_GEN}
    present = set(scenario.present_initially)

    for idx, event in enumerate(scenario.events):
        if event.event_type == 'put':
            contents[event.container] = event.item
        elif event.event_type == 'move':
            contents[event.to_container] = event.item
            contents[event.from_container] = None
        elif event.event_type == 'remove':
            contents[event.container] = None
        elif event.event_type == 'leave':
            if event.character == answerer:
                answerer_leave_idx = idx
                break
            present.discard(event.character)
        elif event.event_type == 'enter':
            present.add(event.character)

    if answerer_leave_idx is None:
        return

    # After answerer leaves, we need someone to do the moves
    present.discard(answerer)
    if not present:
        return

    # What's in the queried container when answerer left?
    item_in_queried = contents[queried]
    if item_in_queried is None:
        return  # Nothing to move

    # Insert events right after answerer leaves:
    # Move item to other, then move back (adds complexity without changing outcome)
    insert_pos = answerer_leave_idx + 1

    # Exclude teammates of characters who left with beliefs about the target
    # to prevent invalidating their beliefs
    teammates_to_exclude = _get_teammates_to_exclude_for_target_moves(scenario, insert_pos)
    valid_movers = [c for c in sorted(present) if c not in teammates_to_exclude]
    if not valid_movers:
        return  # Cannot insert extra events without violating teammate beliefs

    mover1 = rng.choice(valid_movers)
    move_away = Event('move', mover1, from_container=queried, to_container=other, item=item_in_queried)

    # Prefer different mover than mover1, still respecting teammate exclusions
    valid_movers2 = [c for c in valid_movers if c != mover1]
    mover2 = rng.choice(valid_movers2) if valid_movers2 else mover1
    move_back = Event('move', mover2, from_container=other, to_container=queried, item=item_in_queried)

    # Build list of events to insert
    events_to_insert = []

    # 50% chance: have another character leave while answerer is away
    # (must keep at least one person present for the moves)
    # Also exclude any character who has events later in the base scenario
    # AND exclude teammates of characters who modify the target later
    if rng.random() < 0.5 and len(present) > 1:
        # Find characters who act later in the base scenario (can't make them leave)
        # AND find characters who modify the target later (their teammates shouldn't leave with beliefs)
        chars_acting_later = set()
        chars_modifying_target_later = set()
        for evt in scenario.events[insert_pos:]:
            chars_acting_later.add(evt.character)
            if evt.event_type == 'put' and evt.container == queried:
                chars_modifying_target_later.add(evt.character)
            elif evt.event_type == 'move' and (evt.from_container == queried or evt.to_container == queried):
                chars_modifying_target_later.add(evt.character)

        # Teammates of target modifiers shouldn't leave (would form beliefs that get violated)
        teammates_of_modifiers = {_teammate_of(c) for c in chars_modifying_target_later if c in {'A', 'B', 'C', 'D'}}

        potential_leavers = [c for c in sorted(present)
                            if c != mover1 and c != mover2 and c not in chars_acting_later
                            and c not in teammates_of_modifiers]
        if potential_leavers:
            leaver = rng.choice(potential_leavers)
            events_to_insert.append(Event('leave', leaver))
            present.discard(leaver)  # Track that leaver is no longer present

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

    # 67% chance: add opponent leave/enter to break has_enter heuristic
    if rng.random() < 0.67:
        potential = [c for c in sorted(present) if c not in ('A', 'B') and c != mover1 and c != mover2]
        if potential:
            opp = rng.choice(potential)
            events_to_insert.append(Event('leave', opp))
            events_to_insert.append(Event('enter', opp))

    events_to_insert.append(Event('leave', answerer))

    # Insert all events at the correct position
    for i, evt in enumerate(events_to_insert):
        scenario.events.insert(insert_pos + i, evt)


def insert_extra_events_believes_x(scenario: Scenario, answerer: str, rng: random.Random) -> None:
    """
    Insert extra events for BELIEVES_X answerers (Self who believes something uncertain).
    Pattern: Answerer sees events, leaves, then more events happen after they leave.
    Result: Their belief is unchanged (they left), but scenario has more complexity.
    """
    queried = scenario.question_container
    other = _other_container(queried)

    # Find where answerer leaves and track container contents
    answerer_leave_idx = None
    contents = {c: None for c in CONTAINERS_GEN}
    present = set(scenario.present_initially)

    for idx, event in enumerate(scenario.events):
        if event.event_type == 'put':
            contents[event.container] = event.item
        elif event.event_type == 'move':
            contents[event.to_container] = event.item
            contents[event.from_container] = None
        elif event.event_type == 'remove':
            contents[event.container] = None
        elif event.event_type == 'leave':
            if event.character == answerer:
                answerer_leave_idx = idx
                break
            present.discard(event.character)
        elif event.event_type == 'enter':
            present.add(event.character)

    if answerer_leave_idx is None:
        return

    # After answerer leaves, we need someone to do the moves
    present.discard(answerer)
    if not present:
        return

    # What's in the queried container when answerer left?
    item_in_queried = contents[queried]
    if item_in_queried is None:
        return  # Nothing to move

    # Insert events after answerer leaves
    insert_pos = answerer_leave_idx + 1

    # Exclude teammates of characters who left with beliefs about the target
    teammates_to_exclude = _get_teammates_to_exclude_for_target_moves(scenario, insert_pos)
    valid_movers = [c for c in sorted(present) if c not in teammates_to_exclude]
    if not valid_movers:
        return  # Cannot insert extra events without violating teammate beliefs

    mover1 = rng.choice(valid_movers)
    move_away = Event('move', mover1, from_container=queried, to_container=other, item=item_in_queried)

    # Prefer different mover than mover1, still respecting teammate exclusions
    valid_movers2 = [c for c in valid_movers if c != mover1]
    mover2 = rng.choice(valid_movers2) if valid_movers2 else mover1
    move_back = Event('move', mover2, from_container=other, to_container=queried, item=item_in_queried)

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
    potential = sorted(present)  # sorted for deterministic ordering
    if potential and rng.random() < 0.5:
        opp = rng.choice(potential)
        events_to_insert.append(Event('leave', opp))
        events_to_insert.append(Event('enter', opp))

    events_to_insert.append(Event('leave', answerer))

    # Insert all events at the correct position
    for i, evt in enumerate(events_to_insert):
        scenario.events.insert(insert_pos + i, evt)


def insert_extra_events_unknown(scenario: Scenario, answerer: str, rng: random.Random) -> None:
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
    item_in_queried = contents[queried]
    item_in_other = contents[other]

    # Exclude teammates of characters who left with beliefs about the target
    teammates_to_exclude = _get_teammates_to_exclude_for_target_moves(scenario, insert_pos)
    valid_movers = [c for c in sorted(present) if c not in teammates_to_exclude]
    if not valid_movers:
        return  # Cannot insert extra events without violating teammate beliefs

    events_to_insert = []

    if item_in_queried is not None:
        # Standard move_away / move_back pattern on the TARGET container
        mover1 = rng.choice(valid_movers)
        events_to_insert.append(Event('move', mover1, from_container=queried, to_container=other, item=item_in_queried))
        mover2 = _choose_different_actor(set(valid_movers), mover1, rng)
        events_to_insert.append(Event('move', mover2, from_container=other, to_container=queried, item=item_in_queried))
    elif item_in_other is not None:
        # Move from other to queried (affects target!) and back
        mover1 = rng.choice(valid_movers)
        events_to_insert.append(Event('move', mover1, from_container=other, to_container=queried, item=item_in_other))
        mover2 = _choose_different_actor(set(valid_movers), mover1, rng)
        events_to_insert.append(Event('move', mover2, from_container=queried, to_container=other, item=item_in_other))
    else:
        # Both containers empty — add put to TARGET, then move to non-target
        # (using target ensures ECT impact; move instead of remove)
        non_targets = _non_target_containers(queried)
        third = non_targets[1] if other == non_targets[0] else non_targets[0]
        item = rng.choice(ITEMS_GEN)
        actor1 = rng.choice(valid_movers)
        events_to_insert.append(Event('put', actor1, container=queried, item=item))
        actor2 = _choose_different_actor(set(valid_movers), actor1, rng)
        events_to_insert.append(Event('move', actor2, from_container=queried, to_container=third, item=item))

    # Mandatory leave/enter pair: the leaving character has observed the target
    # (since puts happened earlier), so this triggers ECT #1 (knowledge → belief).
    potential = sorted(present)  # sorted for deterministic ordering
    if potential:
        opp = rng.choice(potential)
        events_to_insert.append(Event('leave', opp))
        events_to_insert.append(Event('enter', opp))

    for i, evt in enumerate(events_to_insert):
        scenario.events.insert(insert_pos + i, evt)


def insert_extra_puts(scenario: Scenario, answerer: str, rng: random.Random) -> None:
    """
    Insert extra put/move events for scenarios where Answerer=Teammate,
    Self=Knows X, Teammate=Believes Truth, with Extra=1.
    Creates journey: Believes Truth -> Believes False -> Believes Truth
    Modifies scenario.events in place. Uses moves to third container instead of removes.
    """
    queried = scenario.question_container
    other = _other_container(queried)
    # Get the third container (not queried, not other)
    non_targets = _non_target_containers(queried)
    third = non_targets[1] if other == non_targets[0] else non_targets[0]

    # Find where answerer leaves and track container contents
    answerer_leave_idx = None
    contents = {c: None for c in CONTAINERS_GEN}
    present = set(scenario.present_initially)

    for idx, event in enumerate(scenario.events):
        if event.event_type == 'put':
            contents[event.container] = event.item
        elif event.event_type == 'move':
            contents[event.to_container] = event.item
            contents[event.from_container] = None
        elif event.event_type == 'leave':
            if event.character == answerer:
                answerer_leave_idx = idx
                break
            present.discard(event.character)

    if answerer_leave_idx is None:
        return

    # Z = what answerer believes (item in queried container when they left)
    # X = item in other container (might be None)
    Z = contents[queried]
    X = contents[other]

    if Z is None:
        return  # Nothing to work with

    # Pick a new item Y different from Z and X
    Y = rng.choice([item for item in ITEMS_GEN if item != Z and item != X])

    # Pick random present characters to perform actions
    # (answerer just left, so use remaining present characters)
    present.discard(answerer)
    if not present:
        return

    # Exclude teammates of blue team members who left with beliefs about the target
    insert_pos = answerer_leave_idx + 1
    teammates_to_exclude = _get_teammates_to_exclude_for_target_moves(scenario, insert_pos)
    valid_actors = [c for c in sorted(present) if c not in teammates_to_exclude]
    if not valid_actors:
        return  # Cannot insert extra events without violating teammate beliefs

    # Build the extra events, alternating actors where possible
    extra_events = []
    last_actor = None

    def choose_actor():
        nonlocal last_actor
        candidates = [c for c in valid_actors if c != last_actor]
        actor = rng.choice(candidates) if candidates else rng.choice(valid_actors)
        last_actor = actor
        return actor

    # a. If X exists in other container: MOVE X to third container
    if X is not None:
        extra_events.append(Event('move', choose_actor(), from_container=other, to_container=third, item=X))

    # b. MOVE Z from queried to other container
    extra_events.append(Event('move', choose_actor(), from_container=queried, to_container=other, item=Z))

    # c. PUT Y in queried container
    extra_events.append(Event('put', choose_actor(), container=queried, item=Y))

    # d. MOVE Y from queried to third container
    extra_events.append(Event('move', choose_actor(), from_container=queried, to_container=third, item=Y))

    # e. MOVE Z from other back to queried container
    extra_events.append(Event('move', choose_actor(), from_container=other, to_container=queried, item=Z))

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

    events_to_add = []
    present_list = sorted(present)  # sorted for deterministic ordering

    # Get contents of non-target containers
    nt0, nt1 = non_targets[0], non_targets[1]
    item0 = contents[nt0]
    item1 = contents[nt1]

    # Pick two different items for variety
    available_items = [i for i in ITEMS_GEN if i != item0 and i != item1]
    new_item1 = rng.choice(available_items)
    new_item2 = rng.choice([i for i in available_items if i != new_item1])

    if item0 is not None and item1 is not None:
        # Both have items: just move one to the other (item1 ends up with item0)
        # Can't swap without using target as temp, so keep it simple
        a1 = rng.choice(present_list)
        events_to_add.append(Event('move', a1, from_container=nt0, to_container=nt1, item=item0))
    elif item0 is not None:
        # First non-target has item: move it once, put new item in empty one
        a1 = rng.choice(present_list)
        events_to_add.append(Event('move', a1, from_container=nt0, to_container=nt1, item=item0))
        a2 = _choose_different_actor(present, a1, rng)
        events_to_add.append(Event('put', a2, container=nt0, item=new_item1))
    elif item1 is not None:
        # Second non-target has item: move it once, put new item in empty one
        a1 = rng.choice(present_list)
        events_to_add.append(Event('move', a1, from_container=nt1, to_container=nt0, item=item1))
        a2 = _choose_different_actor(present, a1, rng)
        events_to_add.append(Event('put', a2, container=nt1, item=new_item1))
    else:
        # Both empty: put different items in each
        a1 = rng.choice(present_list)
        events_to_add.append(Event('put', a1, container=nt0, item=new_item1))
        a2 = _choose_different_actor(present, a1, rng)
        events_to_add.append(Event('put', a2, container=nt1, item=new_item2))

    for i, evt in enumerate(events_to_add):
        scenario.events.insert(insert_idx + i, evt)


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
    # Find the item currently in the queried container
    old_item = None
    for event in scenario.events:
        if event.event_type == 'put' and event.container == scenario.question_container:
            old_item = event.item
            break
        elif event.event_type == 'move' and event.to_container == scenario.question_container:
            old_item = event.item
            break

    if old_item is not None:
        new_item = rng.choice([i for i in ITEMS_GEN if i != old_item])
        for event in scenario.events:
            if event.item == old_item:
                event.item = new_item


def generate_scenarios_from_tuples(specs: List[SpecTuple], outfile: str, seed: Optional[int] = None, chartypes: List[CharacterType] = [CharacterType.LIVE_PLAYER, CharacterType.HONEST_OPPONENT, CharacterType.DISHONEST_TEAMMATE, CharacterType.DISHONEST_OPPONENT, CharacterType.NEUTRAL]) -> None:
    """Generate scenarios using shared-base approach for ECT/SIT parity.

    For each scenario ID, Extra=0 and Extra=1 are generated from the SAME base:
    1. Generate base scenario
    2. Deep-copy base for Extra=1
    3. Apply filler to Extra=0 (adds SIT without ECT)
    4. Apply name variation + extra events to Extra=1 (adds ECT)

    This guarantees Extra=1 ECT > Extra=0 ECT for all scenarios.

    Output order: ID1-E0, ID1-E1, ID2-E0, ID2-E1, ...
    """
    rng = random.Random(seed)
    scenarios: List[Scenario] = []
    chars = _map_chartypes_to_names(chartypes)
    acting_chars = [c for c in chars if c != 'N']

    # Group specs by ID (Extra=0 and Extra=1 for same ID should share base)
    spec_by_id: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for spec in specs:
        spec_by_id[spec['Id']][spec['Extra']] = spec

    # Preserve original ID order (as they first appear in the CSV)
    seen_ids: List[str] = []
    for spec in specs:
        if spec['Id'] not in seen_ids:
            seen_ids.append(spec['Id'])

    scenario_idx = 0
    for scenario_id in seen_ids:
        e0_spec = spec_by_id[scenario_id].get(0)
        e1_spec = spec_by_id[scenario_id].get(1)

        # --- Case: Only E1 spec (single-spec mode from test runner) ---
        # Generate E1 independently without shared base
        if e0_spec is None and e1_spec is not None:
            row = e1_spec
            actor = _map_chartypes_to_names([row['Actor']])[0]
            answerer = actor if row['Answerer'] == 'Self' else (_teammate_of(actor) if row['Answerer'] == 'Teammate' else _opponent_of(actor, rng))
            available: Set[str] = set(acting_chars)
            queried_container = rng.choice(CONTAINERS_GEN)
            queried_item = rng.choice(ITEMS_GEN)

            sb = Scenario_Builder(rng, queried_container, queried_item, available)
            sb.plan_availability(row, answerer)
            sb.build_scenario(answerer)

            present_initially = sorted(list(sb.present_initially))

            scenario_e1 = Scenario(
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
                extra=1,
            )

            # Insert extra events
            extra_rng = random.Random(rng.randint(0, 2**31))
            answerer_state = _get_answerer_state(e1_spec)

            if (e1_spec['Answerer'] == 'Teammate' and
                e1_spec['KS_Self'] == EpistemicState.KNOWS_X and
                e1_spec['KS_Teammate'] == EpistemicState.BELIEVES_TRUTH):
                insert_extra_puts(scenario_e1, answerer, extra_rng)
            elif answerer_state == EpistemicState.KNOWS_TRUTH:
                insert_extra_events_with_revelation(scenario_e1, answerer, extra_rng)
            elif answerer_state == EpistemicState.BELIEVES_TRUTH:
                insert_extra_events_believes_true(scenario_e1, answerer, extra_rng)
            elif answerer_state == EpistemicState.BELIEVES_FALSE:
                insert_extra_events_believes_false(scenario_e1, answerer, extra_rng)
            elif answerer_state == EpistemicState.BELIEVES_X:
                insert_extra_events_believes_x(scenario_e1, answerer, extra_rng)
            elif answerer_state == EpistemicState.UNKNOWN:
                insert_extra_events_unknown(scenario_e1, answerer, extra_rng)
            else:
                insert_extra_events(scenario_e1, answerer, actor, answerer_state, e1_spec, extra_rng)

            _validate_invariants(scenario_e1)
            _validate_teammate_belief_integrity(scenario_e1)
            scenario_e1.situation_event_count = _count_visible_events(scenario_e1)
            scenario_e1.epistemic_transitions = count_epistemic_category_transitions(scenario_e1)

            if not scenario_e1.events:
                raise ValueError(f"Generated scenario {scenario_e1.id} Extra={scenario_e1.extra} has no events")

            scenarios.append(scenario_e1)
            scenario_idx += 1
            continue

        # --- Case: No E0 spec and no E1 spec (shouldn't happen) ---
        if e0_spec is None:
            continue

        # --- Case: E0 spec exists (generate E0, optionally E1 via shared base) ---
        row = e0_spec
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
            extra=0,
        )

        # --- Extra=0: Add filler events ---
        scenario_e0 = copy.deepcopy(base_scenario)
        filler_seed = int(f"{seed or 0}{scenario_idx:03d}{int(row['Id']):04d}")
        filler_rng = random.Random(filler_seed)
        insert_filler_events(scenario_e0, filler_rng)

        # Validate and compute metrics for Extra=0
        _validate_invariants(scenario_e0)
        _validate_teammate_belief_integrity(scenario_e0)
        scenario_e0.situation_event_count = _count_visible_events(scenario_e0)
        scenario_e0.epistemic_transitions = count_epistemic_category_transitions(scenario_e0)

        if not scenario_e0.events:
            raise ValueError(f"Generated scenario {scenario_e0.id} Extra={scenario_e0.extra} has no events")

        scenarios.append(scenario_e0)
        scenario_idx += 1

        # --- Extra=1: Deep-copy base, apply variation, add extra events ---
        # Use rejection sampling to ensure SIT gap <= 3
        if e1_spec is not None:
            MAX_SIT_GAP = 3
            MAX_RETRIES = 10
            e0_sit = scenario_e0.situation_event_count

            for retry in range(MAX_RETRIES):
                scenario_e1 = copy.deepcopy(base_scenario)
                scenario_e1.extra = 1
                scenario_e1.round_num = (scenario_idx // len(acting_chars)) + 1

                # Apply cosmetic name variation (container swap, item change)
                variation_seed = int(f"{seed or 0}{scenario_idx:03d}{int(row['Id']):04d}99")
                variation_rng = random.Random(variation_seed)
                apply_name_variation(scenario_e1, variation_rng)

                # Insert extra events (adds ECT)
                # Use retry counter to vary RNG for retries
                retry_rng = random.Random(rng.randint(0, 2**31) + retry)
                base_event_count = len(scenario_e1.events)
                answerer_state = _get_answerer_state(e1_spec)

                if (e1_spec['Answerer'] == 'Teammate' and
                    e1_spec['KS_Self'] == EpistemicState.KNOWS_X and
                    e1_spec['KS_Teammate'] == EpistemicState.BELIEVES_TRUTH):
                    insert_extra_puts(scenario_e1, answerer, retry_rng)
                elif answerer_state == EpistemicState.KNOWS_TRUTH:
                    insert_extra_events_with_revelation(scenario_e1, answerer, retry_rng)
                elif answerer_state == EpistemicState.BELIEVES_TRUTH:
                    insert_extra_events_believes_true(scenario_e1, answerer, retry_rng)
                elif answerer_state == EpistemicState.BELIEVES_FALSE:
                    insert_extra_events_believes_false(scenario_e1, answerer, retry_rng)
                elif answerer_state == EpistemicState.BELIEVES_X:
                    insert_extra_events_believes_x(scenario_e1, answerer, retry_rng)
                elif answerer_state == EpistemicState.UNKNOWN:
                    insert_extra_events_unknown(scenario_e1, answerer, retry_rng)
                else:
                    insert_extra_events(scenario_e1, answerer, actor, answerer_state, e1_spec, retry_rng)

                # Validate: extra events were actually added
                if len(scenario_e1.events) <= base_event_count:
                    import warnings
                    warnings.warn(
                        f"ID {row['Id']} Extra=1: insert function failed to add events "
                        f"(answerer_state={answerer_state}, before={base_event_count}, after={len(scenario_e1.events)})"
                    )

                # Validate and compute metrics for Extra=1
                _validate_invariants(scenario_e1)
                _validate_teammate_belief_integrity(scenario_e1)
                scenario_e1.situation_event_count = _count_visible_events(scenario_e1)
                scenario_e1.epistemic_transitions = count_epistemic_category_transitions(scenario_e1)

                # Check SIT gap - accept if within threshold
                sit_gap = abs(scenario_e1.situation_event_count - e0_sit)
                if sit_gap <= MAX_SIT_GAP:
                    break

            # Warn if max retries reached without satisfying SIT constraint
            if sit_gap > MAX_SIT_GAP:
                import warnings
                warnings.warn(
                    f"ID {row['Id']} Extra=1: SIT gap {sit_gap} exceeds threshold after {MAX_RETRIES} retries"
                )

            scenarios.append(scenario_e1)
            scenario_idx += 1

    # Final safeguard: ensure we generated scenarios if specs were provided
    if specs and not scenarios:
        raise ValueError(
            f"No scenarios generated from {len(specs)} specs. "
            f"Specs: {[{'Id': s['Id'], 'Extra': s['Extra']} for s in specs]}"
        )

    save_scenarios(scenarios, outfile, chars, chartypes)


if __name__ == "__main__":
    specs = read_specs_from_csv('ToM - scenarios.csv')
    outfile = 'scenarios_generated4.json'
    chartypes = [CharacterType.LIVE_PLAYER, CharacterType.DISHONEST_OPPONENT, CharacterType.DISHONEST_TEAMMATE, CharacterType.DISHONEST_OPPONENT]
    generate_scenarios_from_tuples(specs, outfile, seed=None, chartypes=chartypes)
    print(f"Created {outfile} with auto-generated scenarios")
