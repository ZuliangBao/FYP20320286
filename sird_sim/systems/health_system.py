from __future__ import annotations

import math
from collections.abc import Callable

from ..domain.person import HealthState, Person
from ..events.event import (
    BecomeInfectiousEvent,
    DieEvent,
    Event,
    EventType,
    ImmunityWanesEvent,
    RecoverEvent,
)
from ..world import World


EventHandler = Callable[[World, Event], None]


class HealthSystem:
    """
    Process health-state transitions and disease transmission.

    Processing order:

        1. Process all due health events.
        2. Process new infections from current contacts.
    """

    def __init__(self) -> None:
        # 创建一次即可，不需要每个 step 重新建立
        self._event_handlers: dict[EventType, EventHandler] = {
            EventType.BECOME_INFECTIOUS:
                self._handle_become_infectious,
            EventType.RECOVER:
                self._handle_recover,
            EventType.DIE:
                self._handle_die,
            EventType.IMMUNITY_WANES:
                self._handle_immunity_wanes,
        }

    def step(self, world: World) -> None:
        """
        Process one health-system step.
        """
        if world.config is None:
            raise RuntimeError(
                "world.config must be set before HealthSystem.step()"
            )

        if world.current_time is None:
            raise RuntimeError(
                "world.current_time must be set before "
                "HealthSystem.step()"
            )

        self._process_due_events(world)
        self._process_new_infections(world)

    # ============================================================
    # Phase 1: process due events
    # ============================================================

    def _process_due_events(self, world: World) -> None:
        """
        Process every event whose scheduled time is no later than the
        current simulation time.
        """
        while True:
            next_event_time = world.event_queue.peek_time()

            # None means the queue is empty
            if next_event_time is None:
                break

            if next_event_time > world.current_time:
                break

            event = world.event_queue.pop_next()

            if event is None:
                break

            handler = self._event_handlers.get(event.kind)

            if handler is None:
                raise ValueError(
                    f"No HealthSystem handler registered for "
                    f"{event.kind!r}"
                )

            handler(world, event)

    def _handle_become_infectious(
        self,
        world: World,
        event: Event,
    ) -> None:
        """
        Make a person infectious and schedule either recovery or death
        using competing continuous-time hazards.
        """
        person = world.get_person(event.person_id)

        # 防止异常旧事件把死亡状态重新覆盖
        if person.health_state == HealthState.DEAD:
            person.pending_event = None
            return

        person.health_state = HealthState.INFECTED

        lambda_recover = self._probability_to_hazard(
            probability=world.config.recovery_rate,
            tick_duration=world.config.tick_duration,
        )

        lambda_die = self._probability_to_hazard(
            probability=world.config.deadly_rate,
            tick_duration=world.config.tick_duration,
        )

        lambda_total = lambda_recover + lambda_die

        # Both risks are zero, and this person will remain INFECTED indefinitely.
        if lambda_total == 0.0:
            person.pending_event = None
            return

        waiting_time = float(
            world.rng.exponential(
                scale=1.0 / lambda_total,
            )
        )

        event_time = world.current_time + waiting_time
        death_probability = lambda_die / lambda_total

        if world.rng.random() < death_probability:
            next_event = world.event_queue.schedule(
                DieEvent,
                time=event_time,
                person_id=person.person_id,
            )
        else:
            next_event = world.event_queue.schedule(
                RecoverEvent,
                time=event_time,
                person_id=person.person_id,
            )

        # Cover the event that has just been processed as "BecomeInfectiousEvent"
        person.pending_event = next_event

    def _handle_recover(
        self,
        world: World,
        event: Event,
    ) -> None:
        """
        Move a person into the recovered state.

        Immunity is permanent until immunity-duration configuration is
        added.
        """
        person = world.get_person(event.person_id)

        if person.health_state == HealthState.DEAD:
            person.pending_event = None
            return

        person.health_state = HealthState.RECOVERED
        person.pending_event = None

        # When supporting immune regression:
        #
        # immunity_event = world.event_queue.schedule(
        #     ImmunityWanesEvent,
        #     time=(
        #         world.current_time
        #         + world.config.immunity_duration
        #     ),
        #     person_id=person.person_id,
        # )
        #
        # person.pending_event = immunity_event

    def _handle_die(
        self,
        world: World,
        event: Event,
    ) -> None:
        """
        Move a person into the terminal DEAD state and remove them from
        their current physical place.
        """
        person = world.get_person(event.person_id)

        person.health_state = HealthState.DEAD
        person.pending_event = None

        # 防止 ContactSystem 继续把死者算作场所成员
        world.remove_from_place(person)

    def _handle_immunity_wanes(
        self,
        world: World,
        event: Event,
    ) -> None:
        """
        Return a recovered person to the susceptible state.
        """
        person = world.get_person(event.person_id)

        if person.health_state == HealthState.DEAD:
            person.pending_event = None
            return

        person.health_state = HealthState.SUSCEPTIBLE
        person.pending_event = None

    # ============================================================
    # Phase 2: process new infections
    # ============================================================

    def _process_new_infections(self, world: World) -> None:
        """
        Process transmission attempts for every current contact pair.
        """
        for person_a_id, person_b_id in world.pending_contacts:
            person_a = world.get_person(person_a_id)
            person_b = world.get_person(person_b_id)

            transmission_pair = self._get_transmission_pair(
                person_a,
                person_b,
            )

            if transmission_pair is None:
                continue

            susceptible, infected = transmission_pair

            # 已经被其他接触成功感染并排入事件队列
            if susceptible.pending_event is not None:
                continue

            transmission_probability = (
                self._transmission_probability(
                    world=world,
                    susceptible=susceptible,
                    infected=infected,
                )
            )

            if not 0.0 <= transmission_probability <= 1.0:
                raise ValueError(
                    "Transmission probability must be in [0.0, 1.0]"
                )

            if world.rng.random() >= transmission_probability:
                continue

            infection_event = world.event_queue.schedule(
                BecomeInfectiousEvent,
                time=world.current_time,
                person_id=susceptible.person_id,
                source_person_id=infected.person_id,
            )

            susceptible.pending_event = infection_event

    @staticmethod
    def _get_transmission_pair(
        person_a: Person,
        person_b: Person,
    ) -> tuple[Person, Person] | None:
        """
        Return (susceptible, infected) when the pair contains exactly
        one susceptible person and one infected person.
        """
        if (
            person_a.health_state == HealthState.SUSCEPTIBLE
            and person_b.health_state == HealthState.INFECTED
        ):
            return person_a, person_b

        if (
            person_b.health_state == HealthState.SUSCEPTIBLE
            and person_a.health_state == HealthState.INFECTED
        ):
            return person_b, person_a

        return None

    @staticmethod
    def _transmission_probability(
        world: World,
        susceptible: Person,
        infected: Person,
    ) -> float:
        """
        Calculate transmission probability for one contact.

        The person parameters are retained for future extensions such
        as relationship weights, place types, infectiousness, and
        susceptibility.
        """
        return world.config.infection_probability

    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def _probability_to_hazard(
        probability: float,
        tick_duration: float,
    ) -> float:
        """
        Convert a per-tick probability into a continuous-time hazard:

            lambda = -ln(1 - probability) / tick_duration
        """
        if not math.isfinite(probability):
            raise ValueError(
                "Probability must be finite"
            )

        if not 0.0 <= probability < 1.0:
            raise ValueError(
                "Probability must be in [0.0, 1.0)"
            )

        if not math.isfinite(tick_duration):
            raise ValueError(
                "tick_duration must be finite"
            )

        if tick_duration <= 0.0:
            raise ValueError(
                "tick_duration must be greater than 0"
            )

        return (
            -math.log1p(-probability)
            / tick_duration
        )