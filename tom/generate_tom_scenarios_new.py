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

        # Now execute the leave-immediately actions
        for who in leave_immediately_group:
            self.leave(who)

        if len(self.exclude_false) > 0:
            old_item = _pick_other_item(self.rng, self.queried_item)
            self.put(self.queried_container, old_item, exclude=None)
            for who in self.rng.sample(list(self.exclude_false), len(self.exclude_false)):
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
        for who in self.rng.sample(list(self.exclude_true), len(self.exclude_true)):
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
      - Asked container never narrates two different items without an intervening move.
    """
    present = set(s.present_initially)
    contents = {'bag': None, 'box': None}
    for idx, e in enumerate(s.events):
        if e.event_type == 'leave':
            present.discard(e.character)
        elif e.event_type == 'put':
            if e.character not in present:
                raise ValueError(f"Event {idx}: {e.character} acted after leaving.")
            if contents[e.container] is not None and contents[e.container] != e.item:
                # Should be immediately preceded by a move out of that container
                if not (idx > 0 and s.events[idx-1].event_type == 'move'
                        and s.events[idx-1].from_container == e.container):
                    raise ValueError(f"Event {idx}: put into non-empty {e.container} without prior move.")
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


def _get_answerer_state(spec: dict) -> EpistemicState:
    """Get the answerer's final epistemic state from the spec."""
    if spec['Answerer'] == 'Self':
        if spec['KS_Self'] == EpistemicState.KNOWS_X:
            return EpistemicState.KNOWS_TRUTH
        else:  # BELIEVES_X
            return EpistemicState.UNKNOWN
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
                        answerer_state: EpistemicState, rng: random.Random) -> None:
    """
    Insert extra Leave and Enter events for the answerer to create an additional epistemic state.
    Modifies scenario.events in place.
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
    
    # Leave can be inserted at positions 0 through constraint_idx
    leave_insert_pos = rng.randint(0, constraint_idx)
    
    # Enter must be after leave, so positions leave_insert_pos through constraint_idx
    enter_insert_pos = rng.randint(leave_insert_pos, constraint_idx)
    
    # Insert leave first, then enter (adjusting for the index shift)
    scenario.events.insert(leave_insert_pos, Event('leave', answerer))
    scenario.events.insert(enter_insert_pos + 1, Event('enter', answerer))


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
        )

        # Insert extra events if Extra=1
        if row.get('Extra', 0) == 1:
            insert_extra_events(scenario, answerer, actor, _get_answerer_state(row), rng)

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
