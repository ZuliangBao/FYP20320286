from dataclasses import dataclass
from enum import Enum
from typing import Optional
from ..events.event import Event

class Role(Enum):
    STUDENT = "student"
    WORKER = "worker"


class HealthState(Enum):
    SUSCEPTIBLE = "susceptible"
    INFECTED = "infected"
    RECOVERED = "recovered"
    DEAD = "dead"


@dataclass(kw_only=True)
class Person:
    # Identity
    person_id: int

    # Static attributes
    role: Role
    home_id: int
    workplace_id: Optional[int] = None
    school_id: Optional[int] = None
    public_id: Optional[int] = None

    # Dynamic state
    health_state: HealthState = HealthState.SUSCEPTIBLE
    current_place_id: int     

    # Next scheduled event
    pending_event: Optional[Event] = None

