from dataclasses import dataclass, field
from .domain.person import Person
from .domain.place import Place
from .domain.relationship import Relationship
from .config import SimulationConfig
import numpy as np

@dataclass
class World:
    persons: dict[int, Person] = field(default_factory=dict)
    places: dict[int, Place] = field(default_factory=dict)
    relationships: dict[int, list[Relationship]] = field(default_factory=dict)
    pending_contacts: set[tuple[int, int]] = field(default_factory=set)
    
    rng: np.random.Generator = field(
        default_factory=np.random.default_rng
    )
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