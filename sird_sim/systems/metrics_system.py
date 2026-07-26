from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..domain.person import HealthState
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


class MetricsSystem:
    """
    Collect population-level SIRD metrics over time.

    The recorded history belongs to this system rather than World,
    because it is simulation output rather than world state.
    """

    def __init__(self) -> None:
        self.history: list[MetricsSnapshot] = []

    def step(self, world: World) -> None:
        """
        Count people in each health state and append one snapshot.
        """
        if world.current_time is None:
            raise RuntimeError(
                "world.current_time must be set before "
                "MetricsSystem.step()"
            )

        state_counts = Counter(
            person.health_state
            for person in world.persons.values()
        )

        snapshot = MetricsSnapshot(
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

        self.history.append(snapshot)