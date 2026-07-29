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
    """A single simulated individual and their current state.

    Attributes:
        person_id: Unique identifier.
        role: Fixed role (STUDENT or WORKER) assigned at generation.
        home_id: Permanent household affiliation.
        workplace_id: Permanent workplace affiliation, if employed.
        school_id: Permanent school affiliation, if a student.
        public_id: Reserved for a future extension (e.g. a fixed
            public-place affiliation). Not currently read or written
            by any System; ScheduleSystem selects a public place
            uniformly at random each visit instead.
        health_state: Current SIRD health state.
        current_place_id: Where the person currently is. Becomes None
            only when health_state is DEAD (HealthEventSystem removes
            dead people from their place).
        pending_event: The next scheduled Event affecting this
            person's health state, if any. HealthEventSystem checks
            `person.pending_event is event` to detect stale events
            that have been superseded by a later one.
        recovered_at: Simulation time the person last recovered. Only
            set while health_state is RECOVERED; reset to None on
            infection, death, or immunity waning.
    """
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
    current_place_id: int | None = None    

    # Next scheduled event
    pending_event: Optional[Event] = None

    # Immunity recovered
    recovered_at: float | None = None
