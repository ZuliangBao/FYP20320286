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
    # Identity
    place_id: int
    
    # Static attributes
    place_type: PlaceType
    capacity:Optional[int] = None

    # Dynamic state
    occupants: set[int] = field(default_factory=set)


    