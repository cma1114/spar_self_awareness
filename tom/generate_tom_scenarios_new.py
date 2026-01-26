import random
from tom_helpers import (
    Scenario, Event, EpistemicState, CharacterType,
    save_scenarios, SpecTuple, read_specs_from_csv
)
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass

ITEMS_GEN = ['apple', 'ball', 'banana', 'brick', 'stapler', 'orange']
CONTAINERS_GEN = ['bag', 'box']

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
    return CONTAINERS_GEN[1] if c == CONTAINERS_GEN[0] else CONTAINERS_GEN[0]

def _pick_other_item(rng: random.Random, exclude: str) -> str:
    return rng.choice([x for x in ITEMS_GEN if x != exclude])

def _choose_different_actor(present: set, last_actor: Optional[str], rng: random.Random) -> str:
    """Choose actor, preferring someone different from last_actor if possible."""
    present_list = list(present)
    if len(present_list) == 1 or last_actor is None:
        return rng.choice(present_list)
    # Exclude last actor, pick from remaining
    candidates = [c for c in present_list if c != last_actor]
    return rng.choice(candidates) if candidates else rng.choice(present_list)

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
        pool = [p for p in self.present if not exclude or p not in exclude]
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
                    self.include.add(random.choice([opponent1, opponent2])) 

            elif spec['KS_Teammate'] == EpistemicState.KNOWS_TRUTH and spec['KS_Opponent'] == EpistemicState.UNKNOWN:
                self.include.add(teammate)
                if spec['Answerer'] == 'Self':
                    self.exclude.add(opponent1)
                    self.exclude.add(opponent2)
                elif spec.get('Answerer') == 'Opponent':
                    # ensure the chosen opponent (the one who must answer) is exclu
                    self.exclude.add(answerer)
                else:
                    self.exclude.add(random.choice([opponent1, opponent2])) 

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
                    self.include.add(random.choice([opponent1, opponent2])) 

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
                    self.exclude_true.add(random.choice([opponent1, opponent2])) 
            elif spec['KS_Teammate'] == EpistemicState.BELIEVES_TRUTH and spec['KS_Opponent'] == EpistemicState.BELIEVES_FALSE:
                self.exclude_true.add(teammate)
                if spec['Answerer'] == 'Opponent':
                    self.exclude_false.add(opponent1) 
                    self.exclude_false.add(opponent2)
                else:
                    self.exclude_false.add(random.choice([opponent1, opponent2])) 
            elif spec['KS_Teammate'] == EpistemicState.BELIEVES_TRUTH and spec['KS_Opponent'] == EpistemicState.KNOWS_TRUTH:
                self.exclude_true.add(teammate)
                if spec.get('Answerer') == 'Opponent':
                    self.include.add(answerer)
                else:
                    self.include.add(random.choice([opponent1, opponent2]))
            elif spec['KS_Teammate'] == EpistemicState.BELIEVES_FALSE and spec['KS_Opponent'] == EpistemicState.BELIEVES_TRUTH:
                self.exclude_false.add(teammate)
                if spec['Answerer'] == 'Teammate':
                    self.exclude_true.add(opponent1) 
                    self.exclude_true.add(opponent2) 
                elif spec.get('Answerer') == 'Opponent':
                    self.exclude_true.add(answerer)
                else:
                    self.exclude_true.add(random.choice([opponent1, opponent2])) 
            elif spec['KS_Teammate'] == EpistemicState.BELIEVES_FALSE and spec['KS_Opponent'] == EpistemicState.BELIEVES_FALSE:
                self.exclude_false.add(teammate)
                if spec['Answerer'] == 'Opponent':
                    self.exclude_false.add(answerer)
                else:
                    self.exclude_false.add(random.choice([opponent1, opponent2])) #need to keep one around to do the move
            elif spec['KS_Teammate'] == EpistemicState.BELIEVES_FALSE and spec['KS_Opponent'] == EpistemicState.KNOWS_TRUTH:
                self.exclude_false.add(teammate)
                if spec['Answerer'] == 'Teammate':
                    self.include.add(opponent1) 
                    self.include.add(opponent2)
                elif spec.get('Answerer') == 'Opponent':
                    self.include.add(answerer) 
                else:
                    self.include.add(random.choice([opponent1, opponent2])) 
            elif spec['KS_Teammate'] == EpistemicState.KNOWS_TRUTH and spec['KS_Opponent'] == EpistemicState.BELIEVES_TRUTH:
                self.include.add(teammate)
                if spec['Answerer'] == 'Opponent':
                    self.exclude_true.add(answerer) 
                else:
                    self.exclude_true.add(random.choice([opponent1, opponent2])) 
            elif spec['KS_Teammate'] == EpistemicState.KNOWS_TRUTH and spec['KS_Opponent'] == EpistemicState.BELIEVES_FALSE:
                self.include.add(teammate)
                if spec['Answerer'] == 'Teammate' or spec['Answerer'] == 'Opponent':
                    self.exclude_false.add(opponent1) 
                    self.exclude_false.add(opponent2) 
                else:
                    self.exclude_false.add(random.choice([opponent1, opponent2])) 
            elif spec['KS_Teammate'] == EpistemicState.KNOWS_TRUTH and spec['KS_Opponent'] == EpistemicState.KNOWS_TRUTH:
                self.include.add(teammate)
                if spec.get('Answerer') == 'Opponent':
                    self.include.add(answerer) 
                else:
                    self.include.add(random.choice([opponent1, opponent2])) 

        self.present_initially = self.exclude | self.exclude_true | self.exclude_false | self.include # who must be present initially

    def build_scenario(self, answerer: str):
        #randomly add anyone who is in available but not in present_initially to present_initially
        leave_immediately_group = set()
        for who in self.available:
            if who not in self.present_initially:
                if self.rng.random() < 0.5:
                    self.present_initially.add(who)
        for who in self.exclude:#unconstrained - can believe truth or falsehood or nothing
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
            chars_list = list(chars)
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

        # Only exclude answerer's teammate if there's someone else available
        exclude_set = None
        if answerer in self.exclude_false:
            potential_exclude = _teammate_of(answerer)
            # Check if excluding would still leave someone to place the item
            available_for_put = [p for p in self.present if p != potential_exclude]
            if len(available_for_put) > 0:
                exclude_set = {potential_exclude}

        #print(f"exclude_true: {self.exclude_true}, exclude_false: {self.exclude_false}, present_initially: {self.present_initially}, present: {self.present}, answerer: {answerer}")
        self.put(self.queried_container, self.queried_item, exclude=exclude_set)
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


def _validate_invariants(s: 'Scenario') -> None:
    """
    Defensive checks to catch logical errors early:
      - No one acts after leaving.
      - No put into a non-empty container.
    """
    present = set(s.present_initially)
    contents = {'bag': None, 'box': None}
    for idx, e in enumerate(s.events):
        if e.event_type == 'leave':
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
            contents[e.to_container] = e.item
            contents[e.from_container] = None
        elif e.event_type == 'enter':
            if e.character in present:
                raise ValueError(f"Event {idx}: {e.character} entered but was already present.")
            present.add(e.character)
        elif e.event_type == 'remove':
            if e.character not in present:
                raise ValueError(f"Event {idx}: {e.character} acted after leaving.")
            contents[e.container] = None


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

    # 2. Someone moves item to other container while answerer is away
    mover1 = rng.choice(list(present_after_leave))
    move_away_event = Event('move', mover1, from_container=queried, to_container=other, item=initial_item)

    # 3. Answerer returns
    enter_event = Event('enter', answerer)

    # 4. Someone moves item back to queried container (revelation!)
    # Prefer different mover than mover1
    mover2 = _choose_different_actor(present_after_leave, mover1, rng)
    move_back_event = Event('move', mover2, from_container=other, to_container=queried, item=initial_item)

    # Build list of events to insert
    events_to_insert = [leave_event]

    # 50% chance: have another character leave while answerer is away
    # (must keep at least one person present for the moves)
    # Also exclude any character who has events later in the base scenario
    if rng.random() < 0.5 and len(present_after_leave) > 1:
        # Find characters who act later in the base scenario (can't make them leave)
        chars_acting_later = set()
        for evt in scenario.events[leave_pos:]:
            chars_acting_later.add(evt.character)

        potential_leavers = [c for c in present_after_leave
                            if c != mover1 and c != mover2 and c not in chars_acting_later]
        if potential_leavers:
            leaver = rng.choice(potential_leavers)
            events_to_insert.append(Event('leave', leaver))

    events_to_insert.append(move_away_event)
    events_to_insert.append(enter_event)
    events_to_insert.append(move_back_event)

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

    mover1 = rng.choice(list(present))
    move_away = Event('move', mover1, from_container=queried, to_container=other, item=queried_item)

    # Prefer different mover than mover1
    mover2 = _choose_different_actor(present, mover1, rng)
    move_back = Event('move', mover2, from_container=other, to_container=queried, item=queried_item)

    # Build list of events to insert
    events_to_insert = []

    # 50% chance: have another character leave while answerer is away
    # (must keep at least one person present for the moves)
    # Also exclude any character who has events later in the base scenario
    if rng.random() < 0.5 and len(present) > 1:
        # Find characters who act later in the base scenario (can't make them leave)
        chars_acting_later = set()
        for evt in scenario.events[insert_pos:]:
            chars_acting_later.add(evt.character)

        potential_leavers = [c for c in present
                            if c != mover1 and c != mover2 and c not in chars_acting_later]
        if potential_leavers:
            leaver = rng.choice(potential_leavers)
            events_to_insert.append(Event('leave', leaver))

    events_to_insert.append(move_away)
    events_to_insert.append(move_back)

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

    mover1 = rng.choice(list(present))
    move_away = Event('move', mover1, from_container=queried, to_container=other, item=item_in_queried)

    # Prefer different mover than mover1
    mover2 = _choose_different_actor(present, mover1, rng)
    move_back = Event('move', mover2, from_container=other, to_container=queried, item=item_in_queried)

    # Build list of events to insert
    events_to_insert = []

    # 50% chance: have another character leave while answerer is away
    # (must keep at least one person present for the moves)
    # Also exclude any character who has events later in the base scenario
    if rng.random() < 0.5 and len(present) > 1:
        # Find characters who act later in the base scenario (can't make them leave)
        chars_acting_later = set()
        for evt in scenario.events[insert_pos:]:
            chars_acting_later.add(evt.character)

        potential_leavers = [c for c in present
                            if c != mover1 and c != mover2 and c not in chars_acting_later]
        if potential_leavers:
            leaver = rng.choice(potential_leavers)
            events_to_insert.append(Event('leave', leaver))

    events_to_insert.append(move_away)
    events_to_insert.append(move_back)

    # 67% chance: add opponent leave/enter to break has_enter heuristic
    if rng.random() < 0.67:
        potential = [c for c in present if c not in ('A', 'B') and c != mover1 and c != mover2]
        if potential:
            opp = rng.choice(potential)
            events_to_insert.append(Event('leave', opp))
            events_to_insert.append(Event('enter', opp))

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

    # Insert move away/back events after answerer leaves
    insert_pos = answerer_leave_idx + 1

    mover1 = rng.choice(list(present))
    move_away = Event('move', mover1, from_container=queried, to_container=other, item=item_in_queried)

    mover2 = _choose_different_actor(present, mover1, rng)
    move_back = Event('move', mover2, from_container=other, to_container=queried, item=item_in_queried)

    # Insert the events
    scenario.events.insert(insert_pos, move_away)
    scenario.events.insert(insert_pos + 1, move_back)

    # 67% chance: add opponent leave/enter to break has_enter heuristic
    if rng.random() < 0.67:
        # Find an opponent who can leave/enter (not A or B, still present)
        potential = [c for c in present if c not in ('A', 'B')]
        if potential:
            opp = rng.choice(potential)
            scenario.events.insert(insert_pos + 2, Event('leave', opp))
            scenario.events.insert(insert_pos + 3, Event('enter', opp))


def insert_extra_puts(scenario: Scenario, answerer: str, rng: random.Random) -> None:
    """
    Insert extra put/move/remove events for scenarios where Answerer=Teammate,
    Self=Knows X, Teammate=Believes Truth, with Extra=1.
    Creates journey: Believes Truth -> Believes False -> Believes Truth
    Modifies scenario.events in place.
    """
    queried = scenario.question_container
    other = _other_container(queried)
    
    # Find where answerer leaves and track container contents
    answerer_leave_idx = None
    contents = {'bag': None, 'box': None}
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
    
    # Build the extra events, alternating actors where possible
    extra_events = []
    last_actor = None

    # a. If X exists in other container: REMOVE X
    if X is not None:
        actor = _choose_different_actor(present, last_actor, rng)
        extra_events.append(Event('remove', actor, container=other, item=X))
        last_actor = actor

    # b. MOVE Z from queried to other container
    actor = _choose_different_actor(present, last_actor, rng)
    extra_events.append(Event('move', actor, from_container=queried, to_container=other, item=Z))
    last_actor = actor

    # c. PUT Y in queried container
    actor = _choose_different_actor(present, last_actor, rng)
    extra_events.append(Event('put', actor, container=queried, item=Y))
    last_actor = actor

    # d. REMOVE Y from queried container
    actor = _choose_different_actor(present, last_actor, rng)
    extra_events.append(Event('remove', actor, container=queried, item=Y))
    last_actor = actor

    # e. MOVE Z from other back to queried container
    actor = _choose_different_actor(present, last_actor, rng)
    extra_events.append(Event('move', actor, from_container=other, to_container=queried, item=Z))
    
    # Insert all extra events right after answerer leaves
    insert_pos = answerer_leave_idx + 1
    for i, event in enumerate(extra_events):
        scenario.events.insert(insert_pos + i, event)


def generate_scenarios_from_tuples(specs: List[SpecTuple], outfile: str, seed: Optional[int] = None, chartypes: List[CharacterType] = [CharacterType.LIVE_PLAYER, CharacterType.HONEST_OPPONENT, CharacterType.DISHONEST_TEAMMATE, CharacterType.DISHONEST_OPPONENT, CharacterType.NEUTRAL]) -> None:
    rng = random.Random(seed)
    scenarios: List[Scenario] = []
    chars = _map_chartypes_to_names(chartypes)
    acting_chars = [c for c in chars if c != 'N']

    for i, row in enumerate(specs):
        #print(f"spec {i}: {row}")
        actor = _map_chartypes_to_names([row['Actor']])[0]
        answerer = actor if row['Answerer'] == 'Self' else (_teammate_of(actor) if row['Answerer'] == 'Teammate' else _opponent_of(actor, rng))
        available: Set[str] = set(chars)
        queried_container = rng.choice(CONTAINERS_GEN)
        queried_item = rng.choice(ITEMS_GEN)

        sb = Scenario_Builder(rng, queried_container, queried_item, available)
        sb.plan_availability(row, answerer)
        sb.build_scenario(answerer)

        present_initially = sorted(list(sb.present_initially))

        scenario = Scenario(
            round_num=(i // len(acting_chars)) + 1,
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
            extra=row.get('Extra', 0),
        )

        # Insert extra events if Extra=1
        if row.get('Extra', 0) == 1:
            answerer_state = _get_answerer_state(row)
            if (row['Answerer'] == 'Teammate' and
                row['KS_Self'] == EpistemicState.KNOWS_X and
                row['KS_Teammate'] == EpistemicState.BELIEVES_TRUTH):
                insert_extra_puts(scenario, answerer, rng)
            elif answerer_state == EpistemicState.KNOWS_TRUTH:
                # Use revelation pattern for KNOWS_TRUTH answerers
                insert_extra_events_with_revelation(scenario, answerer, rng)
            elif answerer_state == EpistemicState.BELIEVES_TRUTH:
                # Answerer believes truth - add events that change and revert while away
                insert_extra_events_believes_true(scenario, answerer, rng)
            elif answerer_state == EpistemicState.BELIEVES_FALSE:
                # Answerer believes false - add events while they're away
                insert_extra_events_believes_false(scenario, answerer, rng)
            elif answerer_state == EpistemicState.BELIEVES_X:
                # Self believes something (uncertain) - add events after they leave
                insert_extra_events_believes_x(scenario, answerer, rng)
            else:
                insert_extra_events(scenario, answerer, actor, answerer_state, row, rng)

        # Validate invariants
        _validate_invariants(scenario)

        scenarios.append(scenario)

    save_scenarios(scenarios, outfile, chars, chartypes)


if __name__ == "__main__":
    specs = read_specs_from_csv('ToM - scenarios.csv')
    outfile = 'scenarios_generated4.json'
    chartypes = [CharacterType.LIVE_PLAYER, CharacterType.DISHONEST_OPPONENT, CharacterType.DISHONEST_TEAMMATE, CharacterType.DISHONEST_OPPONENT]
    generate_scenarios_from_tuples(specs, outfile, seed=None, chartypes=chartypes)
    print(f"Created {outfile} with auto-generated scenarios")
