from dataclasses import dataclass
from enum import Enum

class RelationType(Enum):
    FAMILY = "family"
    SCHOOLMATE = "schoolmate"
    WORKMATE = "workmate"
    FRIEND = "friend"


@dataclass
class Relationship:
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
        


    