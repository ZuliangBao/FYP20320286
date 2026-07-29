from dataclasses import dataclass
from enum import Enum

class RelationType(Enum):
    FAMILY = "family"
    SCHOOLMATE = "schoolmate"
    WORKMATE = "workmate"
    FRIEND = "friend"


@dataclass
class Relationship:
    """An undirected social tie between two people.

    person_a_id and person_b_id are normalized in __post_init__ so
    that person_a_id < person_b_id always holds after construction,
    regardless of the order passed in. This gives each undirected
    relationship a single canonical representation, avoiding
    duplicate (a, b) / (b, a) entries for the same tie.

    Attributes:
        person_a_id: The smaller of the two person_ids (after
            normalization).
        person_b_id: The larger of the two person_ids (after
            normalization).
        relation_type: The kind of tie (FAMILY/SCHOOLMATE/WORKMATE/
            FRIEND).
        weight: Relative connection weight assigned at generation
            time (see generation.py). Not currently read by
            TransmissionSystem, which applies a flat
            infection_probability regardless of relationship weight;
            reserved for a future weighted-transmission extension.
    """
    # Identity
    person_a_id: int
    person_b_id: int
    relation_type: RelationType
    
    # Static attributes
    weight: float
    def __post_init__(self):
        if self.person_a_id == self.person_b_id:
            raise ValueError("A person cannot have a relationship with themselves.")

        # ensure person_a_id < person_b_id
        if self.person_a_id > self.person_b_id:
            self.person_a_id, self.person_b_id = (
                self.person_b_id,
                self.person_a_id,
            )
    