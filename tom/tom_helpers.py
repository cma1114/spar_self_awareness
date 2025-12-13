from enum import Enum
import json
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Set, Tuple

class Team(Enum):
    BLUE = "Blue"
    RED = "Red"
    NEUTRAL = "Neutral"

class CharacterType(Enum):
    LIVE_PLAYER = "live_player"
    HONEST_TEAMMATE = "honest_teammate"
    DISHONEST_TEAMMATE = "dishonest_teammate"
    HONEST_OPPONENT = "honest_opponent"
    DISHONEST_OPPONENT = "dishonest_opponent"
    NEUTRAL = "neutral"

class EpistemicState(Enum):
    BELIEVES_TRUTH = "Believes Truth"
    BELIEVES_FALSE = "Believes False"
    KNOWS_TRUTH = "Knows Truth"
    KNOWS_X = "Knows X"
    BELIEVES_X = "Believes X"
    UNKNOWN = "Unknown"

class EpistemicType(Enum):
    TEAMMATE_HAS_FALSE_BELIEF = "teammate_has_false_belief"
    TEAMMATE_HAS_TRUE_BELIEF = "teammate_has_true_belief"
    TEAMMATE_HAS_NO_BELIEF = "teammate_has_no_belief"
    TEAMMATE_HAS_UNKNOWN_BELIEF = "teammate_has_unknown_belief" # Player has left so doesn't know whether whatever teammate believes is true or false
    PLAYER_HAS_CERTAINTY = "player_has_certainty"
    PLAYER_HAS_UNCERTAINTY = "player_has_uncertainty"
    PLAYER_HAS_NO_BELIEF = "player_has_no_belief"
    OPPONENT_HAS_FALSE_BELIEF = "opponent_has_false_belief"
    OPPONENT_HAS_TRUE_BELIEF_WITH_CERTAINTY = "opponent_has_true_belief_with_certainty"
    HONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY = "honest_opponent_has_true_belief_with_uncertainty"
    DISHONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY = "dishonest_opponent_has_true_belief_with_uncertainty"


class AskConstraintType(Enum):
    TEAMMATE_LACKS_KNOWLEDGE = "teammate_lacks_knowledge"
    TEAMMATE_AND_NEUTRAL_LACK_KNOWLEDGE = "teammate_and_neutral_lack_knowledge"
    TEAMMATE_NEUTRAL_AND_HONEST_OPPONENT_LACK_KNOWLEDGE = "teammate_neutral_and_honest_opponent_lack_knowledge"
    NO_CONSTRAINT = "no_constraint"

SpecTuple = Tuple['CharacterType', 'EpistemicState', 'EpistemicState', 'EpistemicState', 'str', 'CharacterType']

@dataclass
class Event:
    """Represents an event in a scenario."""
    event_type: str  # 'put', 'move', 'leave'
    character: str
    container: Optional[str] = None
    item: Optional[str] = None
    from_container: Optional[str] = None
    to_container: Optional[str] = None


@dataclass
class Character:
    name: str
    team: Team
    char_type: CharacterType
    knowledge: Dict[str, Optional[str]] = field(default_factory=dict)
    beliefs: Dict[str, Optional[str]] = field(default_factory=dict)
    
    def get_knowledge(self, container: str) -> Optional[str]:
        """Get what the character actually knows."""
        return self.knowledge.get(container)
    
    def get_belief(self, container: str) -> Optional[str]:
        """Get what the character believes (knowledge or told belief)."""
        if container in self.beliefs:
            return self.beliefs[container]
        return self.knowledge.get(container)
    
    def update_knowledge(self, container: str, contents: Optional[str]):
        """Update character's direct knowledge."""
        self.knowledge[container] = contents
    
    def receive_info(self, container: str, contents: Optional[str], from_char: 'Character', trust: bool):
        """Receive information from another character."""
        if trust:
            self.beliefs[container] = contents

@dataclass
class Scenario:
    """Represents one scenario in the game."""
    round_num: int
    whose_turn: str
    who_answers: str
    question_container: str
    events: List[Event]
    present_initially: List[str]
    id: Optional[str] = None
    ks_self: Optional[str] = None
    ks_teammate: Optional[str] = None
    ks_opponent: Optional[str] = None
    correct_action: Optional[str] = None
    epistemic_type: Optional[EpistemicType] = None
    ask_constraint: Optional[AskConstraintType] = None
    
    def get_description_for(self, character_name: str, characters: Dict[str, Character]) -> str:
        """Generate scenario description from a character's perspective."""
        char_map = {c: c for c in characters.keys()}
        if character_name in characters:
            char_map[character_name] = "You"
        
        lines = []
        present = set(self.present_initially)
        
        # Initial state
        def format_name_list(names: List[str]) -> str:
            """Return 'X', 'X and Y', or 'X, Y, and Z' with an Oxford comma."""
            if not names:
                return ""
            if len(names) == 1:
                return names[0]
            if len(names) == 2:
                return f"{names[0]} and {names[1]}"
            return ", ".join(names[:-1]) + f", and {names[-1]}"
        names = [char_map.get(c, c) for c in sorted(present)]
        for i, name in enumerate(names):
            if i>0 and name == "You": names[i] = "you"
        char_list = format_name_list(names)
        verb = "are" if (len(names) > 1 or names[0].lower() == "you") else "is"
        lines.append(f"{char_list} {verb} in a room. Inside the room are an empty bag and an empty box.")

        # Narrate only while the perspective character is present
        perspective_present = character_name in present

        for event in self.events:
            actor = char_map.get(event.character, event.character)
            you_form = (actor == "You")
            article = "an" if event.item and event.item[0] in 'aeiou' else "a"

            if event.event_type == 'put':
                if perspective_present:
                    verb_put = "put" if you_form else "puts"
                    lines.append(f"{actor} {verb_put} {article} {event.item} in the {event.container}.")

            elif event.event_type == 'move':
                if perspective_present:
                    verb_move = "move" if you_form else "moves"
                    lines.append(f"{actor} {verb_move} the {event.item} to the {event.to_container}.")

            elif event.event_type == 'leave':
                verb_leave = "leave" if you_form else "leaves"
                lines.append(f"{actor} {verb_leave} the room.")
                present.discard(event.character)
                if event.character == character_name:
                    perspective_present = False
                    
            elif event.event_type == 'enter':
                verb_enter = "enter" if you_form else "enters"
                lines.append(f"{actor} {verb_enter} the room.")
                present.add(event.character)
                if event.character == character_name:
                    perspective_present = True

        return " ".join(lines)
    
    def to_dict(self) -> dict:
        """Convert scenario to dictionary for JSON serialization."""
        return {
            'round_num': self.round_num,
            'whose_turn': self.whose_turn,
            'who_answers': self.who_answers,
            'question_container': self.question_container,
            'events': [asdict(e) for e in self.events],
            'present_initially': self.present_initially,
            'id': self.id if self.id else None, 
            'epistemic_type': self.epistemic_type.value if self.epistemic_type else None,
            'ask_constraint': self.ask_constraint.value if self.ask_constraint else None,
            'ks_self': self.ks_self if self.ks_self else None,
            'ks_teammate': self.ks_teammate if self.ks_teammate else None,
            'ks_opponent': self.ks_opponent if self.ks_opponent else None,
            'correct_action': self.correct_action if self.correct_action else None,
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'Scenario':
        """Create scenario from dictionary."""
        return Scenario(
            round_num=data['round_num'],
            whose_turn=data['whose_turn'],
            who_answers=data['who_answers'],
            question_container=data['question_container'],
            events=[Event(**e) for e in data['events']],
            present_initially=data['present_initially'],
            id=data.get('id') if data.get('id') else None,
            epistemic_type=EpistemicType(data['epistemic_type']) if data.get('epistemic_type') else None,
            ask_constraint=AskConstraintType(data['ask_constraint']) if data.get('ask_constraint') else None,
            ks_self=data.get('ks_self') if data.get('ks_self') else None,
            ks_teammate=data.get('ks_teammate') if data.get('ks_teammate') else None,
            ks_opponent=data.get('ks_opponent') if data.get('ks_opponent') else None,
            correct_action=data.get('correct_action') if data.get('correct_action') else None,
        )

def load_scenarios(filename: str) -> Tuple[List[Scenario], List[str], List[CharacterType]]:
    """Load scenarios from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        # Handle old format for backward compatibility
        return [Scenario.from_dict(s) for s in data], [], []
    
    # Handle new format
    scenarios = [Scenario.from_dict(s) for s in data['scenarios']]
    chars = data['chars']
    chartypes = [CharacterType(value) for value in data['chartypes']]
    return scenarios, chars, chartypes


def save_scenarios(scenarios: List[Scenario], filename: str, chars: List[str], chartypes: List[CharacterType]):
    """Save scenarios to JSON file."""
    output_data = {
        'chars': chars,
        'chartypes': [ct.value for ct in chartypes],
        'scenarios': [s.to_dict() for s in scenarios]
    }
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)

def read_specs_from_csv(infile='ToM - scenarios.csv') -> List[SpecTuple]:
    import csv
    specs: List[SpecTuple] = []
    with open(infile, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            specs.append({
                'Id': row['Id'],
                'Extra': int(row.get('Extra', 0)),
                'Answerer': row['Answerer'],
                'KS_Self': EpistemicState(row['Self']),
                'KS_Teammate': EpistemicState(row['Teammate']),
                'KS_Opponent': EpistemicState(row['Opponent']),
                'Action': row['Action'],
                'Actor': CharacterType.LIVE_PLAYER 
            })
    return specs
