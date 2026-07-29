from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class PlaceType(Enum):
    HOME = "home"
    WORKPLACE = "workplace"
    SCHOOL = "school"
    PUBLIC = "public"



@dataclass
class Place:
    """A physical location that can hold people.

    Attributes:
        place_id: Unique identifier.
        place_type: The kind of place (HOME/WORKPLACE/SCHOOL/PUBLIC).
        capacity: Maximum simultaneous occupants, or None for no limit.
        occupants: person_ids of people currently at this place,
            kept in sync by World.move_person / World.remove_from_place.
    """
    # Identity
    place_id: int
    
    # Static attributes
    place_type: PlaceType
    capacity:Optional[int] = None

    # Dynamic state
    occupants: set[int] = field(default_factory=set)