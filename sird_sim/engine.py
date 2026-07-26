from __future__ import annotations

import math

from .systems.contact_system import ContactSystem
from .systems.health_event_system import HealthEventSystem
from .systems.metrics_system import MetricsSystem
from .systems.schedule_system import ScheduleSystem
from .systems.transmission_system import TransmissionSystem
from .world import World


class Engine:
    """
    Coordinate all simulation systems in their required business order.
    """

    def __init__(
        self,
        *,
        world: World,
        health_event_system: HealthEventSystem,
        schedule_system: ScheduleSystem,
        contact_system: ContactSystem,
        transmission_system: TransmissionSystem,
        metrics_system: MetricsSystem,
    ) -> None:
        self.world = world

        self.health_event_system = health_event_system
        self.schedule_system = schedule_system
        self.contact_system = contact_system
        self.transmission_system = transmission_system
        self.metrics_system = metrics_system

    def step(self) -> None:
        """
        Execute one simulation tick.

        Order:

            1. Process due health events.
            2. Update people's locations.
            3. Generate current contacts.
            4. Process transmission.
            5. Record metrics.
            6. Advance simulation time.
        """
        if self.world.config is None:
            raise RuntimeError(
                "world.config must be set before Engine.step()"
            )

        if self.world.current_time is None:
            raise RuntimeError(
                "world.current_time must be set before Engine.step()"
            )

        self.health_event_system.step(self.world)
        self.schedule_system.step(self.world)
        self.contact_system.step(self.world)
        self.transmission_system.step(self.world)
        self.metrics_system.step(self.world)

        self.world.current_time += (
            self.world.config.tick_duration
        )

    def run(self, total_days: float) -> None:
        """
        Run the simulation forward from its current time.

        The current simulation time is not reset, allowing multiple
        consecutive calls such as:

            engine.run(10)
            engine.run(20)

        which advances the same world by a total of 30 days.
        """
        self._validate_total_days(total_days)

        if self.world.config is None:
            raise RuntimeError(
                "world.config must be set before Engine.run()"
            )

        tick_duration = self.world.config.tick_duration

        self._validate_tick_duration(tick_duration)

        ticks_per_day = 24.0 / tick_duration
        rounded_ticks_per_day = round(ticks_per_day)

        if not math.isclose(
            ticks_per_day,
            rounded_ticks_per_day,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "tick_duration must divide 24 hours exactly; "
                f"received tick_duration={tick_duration}, "
                f"which gives {ticks_per_day} ticks per day"
            )

        total_ticks_float = (
            total_days * rounded_ticks_per_day
        )
        total_ticks = round(total_ticks_float)

        if not math.isclose(
            total_ticks_float,
            total_ticks,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "total_days must correspond to a whole number "
                "of simulation ticks; "
                f"received total_days={total_days}, "
                f"tick_duration={tick_duration}, "
                f"which gives {total_ticks_float} ticks"
            )

        for _ in range(total_ticks):
            self.step()

    @staticmethod
    def _validate_total_days(
        total_days: float,
    ) -> None:
        if isinstance(total_days, bool):
            raise TypeError(
                "total_days must be a number, not bool"
            )

        if not isinstance(total_days, (int, float)):
            raise TypeError(
                "total_days must be an int or float"
            )

        if not math.isfinite(total_days):
            raise ValueError(
                "total_days must be finite"
            )

        if total_days < 0.0:
            raise ValueError(
                "total_days cannot be negative"
            )

    @staticmethod
    def _validate_tick_duration(
        tick_duration: float,
    ) -> None:
        if isinstance(tick_duration, bool):
            raise TypeError(
                "tick_duration must be a number, not bool"
            )

        if not isinstance(tick_duration, (int, float)):
            raise TypeError(
                "tick_duration must be an int or float"
            )

        if not math.isfinite(tick_duration):
            raise ValueError(
                "tick_duration must be finite"
            )

        if tick_duration <= 0.0:
            raise ValueError(
                "tick_duration must be greater than 0"
            )