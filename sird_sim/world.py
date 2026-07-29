from __future__ import annotations
from dataclasses import dataclass, field
from .domain.person import Person
from .domain.place import Place
from .domain.relationship import Relationship
from .config import SimulationConfig
from .events.event_queue import EventQueue
import numpy as np


@dataclass
class World:
    """
    person_id -> list of Relationships involving that person.

    Each Relationship object is indexed under both participants, so the
    same object appears in two different lists. Counting relationships
    by summing len() across all values will double-count; use a set of
    relationship identities (or iterate a single side's canonical list)
    to get the true total.
    """
    persons: dict[int, Person] = field(default_factory=dict)
    places: dict[int, Place] = field(default_factory=dict)
    relationships: dict[int, list[Relationship]] = field(default_factory=dict)
    pending_contacts: set[tuple[int, int]] = field(default_factory=set)
    event_queue: EventQueue = field(default_factory=EventQueue)
    
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    config: SimulationConfig | None = None
    current_time: float | None = None

    def get_person(self, person_id: int) -> Person:
        return self.persons[person_id]

    def get_place(self, place_id: int) -> Place:
        return self.places[place_id]

    def get_relationships(
        self,
        person_id: int,
    ) -> list[Relationship]:
        return self.relationships.get(person_id, [])
    
    def remove_from_place(self, person: Person) -> None:
        if person.current_place_id is None:
            return

        place = self.get_place(person.current_place_id)
        place.occupants.remove(person.person_id)
        person.current_place_id = None

    def add_to_place(
        self,
        person: Person,
        place_id: int,
    ) -> None:
        place = self.get_place(place_id)
        place.occupants.add(person.person_id)
        person.current_place_id = place_id

    def move_person(
        self,
        person: Person,
        place_id: int,
    ) -> None:
        self.remove_from_place(person)
        self.add_to_place(person, place_id)

    def require_config(self) -> SimulationConfig:
        """
        Return the simulation config or fail when the World has not
        been fully initialized.
        """
        config = self.config

        if config is None:
            raise RuntimeError(
                "world.config must be set"
            )

        return config

    def require_current_time(self) -> float:
        """
        Return the current simulation time or fail when it is unset.
        """
        current_time = self.current_time

        if current_time is None:
            raise RuntimeError(
                "world.current_time must be set"
            )

        return current_time