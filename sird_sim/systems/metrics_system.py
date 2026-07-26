from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..domain.person import HealthState
from ..domain.place import PlaceType
from ..world import World


@dataclass(frozen=True)
class MetricsSnapshot:
    """
    Health-state counts recorded at one simulation time.
    """

    time: float
    susceptible: int
    infected: int
    recovered: int
    dead: int


@dataclass(frozen=True)
class OccupancySnapshot:
    """
    Population counts by current place type at one simulation time.
    """

    time: float
    home: int
    workplace: int
    school: int
    public: int


class MetricsSystem:
    """
    Collect population-level SIRD metrics over time.

    The recorded history belongs to this system rather than World,
    because it is simulation output rather than world state.
    """

    def __init__(self) -> None:
        self.history: list[MetricsSnapshot] = []
        self.occupancy_history: list[OccupancySnapshot] = []
        
    def step(self, world: World) -> None:
        """
        Count people in each health state and append one snapshot.
        """
        if world.current_time is None:
            raise RuntimeError(
                "world.current_time must be set before "
                "MetricsSystem.step()"
            )

        self._record_health_metrics(world)
        self._record_occupancy_metrics(world)

    def _record_health_metrics(
        self,
        world: World,
    ) -> None:
        state_counts = Counter(
            person.health_state
            for person in world.persons.values()
        )

        self.history.append(
            MetricsSnapshot(
                time=world.current_time,
                susceptible=state_counts[
                    HealthState.SUSCEPTIBLE
                ],
                infected=state_counts[
                    HealthState.INFECTED
                ],
                recovered=state_counts[
                    HealthState.RECOVERED
                ],
                dead=state_counts[
                    HealthState.DEAD
                ],
            )
        )

    def _record_occupancy_metrics(
        self,
        world: World,
    ) -> None:
        occupancy_counts: Counter[PlaceType] = Counter()

        for place in world.places.values():
            occupancy_counts[place.place_type] += len(
                place.occupants
            )

        self.occupancy_history.append(
            OccupancySnapshot(
                time=world.current_time,
                home=occupancy_counts[PlaceType.HOME],
                workplace=occupancy_counts[
                    PlaceType.WORKPLACE
                ],
                school=occupancy_counts[
                    PlaceType.SCHOOL
                ],
                public=occupancy_counts[
                    PlaceType.PUBLIC
                ],
            )
        )