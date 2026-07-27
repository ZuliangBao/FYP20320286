from __future__ import annotations
import math
from collections.abc import Callable
from ..domain.person import HealthState
from ..events.event import (DieEvent,Event,EventType,RecoverEvent)
from ..domain.person import HealthState, Person
from ..world import World

EventHandler = Callable[[World, Event], None]

class HealthEventSystem:
    """
    Process due health events.

    This system only handles events that have already been scheduled.
    It does not inspect contacts or create new infections.
    """

    def __init__(self) -> None:
        # Build once instead of rebuilding the dispatch table
        # during every simulation step.
        self._event_handlers: dict[
            EventType,
            EventHandler,
        ] = {
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
        Process every event whose time is less than or equal to the
        current simulation time.
        """
        if world.config is None:
            raise RuntimeError(
                "world.config must be set before "
                "HealthEventSystem.step()"
            )

        if world.current_time is None:
            raise RuntimeError(
                "world.current_time must be set before "
                "HealthEventSystem.step()"
            )

        while True:
            next_event_time = world.event_queue.peek_time()

            if next_event_time is None:
                break

            if next_event_time > world.current_time:
                break

            event = world.event_queue.pop_next()

            # After peek_time() has returned an event time,
            # pop_next() should return that active event.
            if event is None:
                raise RuntimeError(
                    "EventQueue.pop_next() returned None after "
                    "peek_time() returned an event time"
                )

            handler = self._event_handlers.get(event.kind)

            if handler is None:
                raise ValueError(
                    "No HealthEventSystem handler registered for "
                    f"{event.kind!r}"
                )

            handler(world, event)

    def _handle_become_infectious(
        self,
        world: World,
        event: Event,
    ) -> None:
        """
        Change the person to INFECTED and schedule their disease outcome.
        """
        person = world.get_person(event.person_id)

        # A terminal state must not be overwritten by a stale event.
        if person.health_state == HealthState.DEAD:
            if person.pending_event is event:
                person.pending_event = None
            return

        person.health_state = HealthState.INFECTED

        self._schedule_outcome(
            world,
            person,
        )

    def _schedule_outcome(
        self,
        world: World,
        person: Person,
    ) -> None:
        """
        Schedule either recovery or death for an infected person using
        competing continuous-time hazards.

        If both hazards are zero, the person remains infected indefinitely
        and no outcome event is scheduled.
        """
        config = world.config
        current_time = world.current_time

        if config is None or current_time is None:
            raise RuntimeError(
                "World configuration and current time are required"
            )

        lambda_recover = self._probability_to_hazard(
            probability=config.recovery_rate,
            tick_duration=config.tick_duration,
        )

        lambda_die = self._probability_to_hazard(
            probability=config.deadly_rate,
            tick_duration=config.tick_duration,
        )

        lambda_total = lambda_recover + lambda_die

        # Both probabilities are zero, so the person remains infected
        # indefinitely and no later event is scheduled.
        if lambda_total == 0.0:
            person.pending_event = None
            return

        waiting_time = float(
            world.rng.exponential(
                scale=1.0 / lambda_total
            )
        )

        outcome_time = current_time + waiting_time
        death_probability = lambda_die / lambda_total

        if world.rng.random() < death_probability:
            outcome_event = world.event_queue.schedule(
                DieEvent,
                time=outcome_time,
                person_id=person.person_id,
            )
        else:
            outcome_event = world.event_queue.schedule(
                RecoverEvent,
                time=outcome_time,
                person_id=person.person_id,
            )

        person.pending_event = outcome_event

    def reschedule_all_infected(
        self,
        world: World,
    ) -> None:
        """
        Reschedule recovery/death outcomes for every currently infected
        person using the current runtime configuration.
    
        Existing outcome events are cancelled first. New waiting times are
        sampled from world.current_time because exponential waiting times
        are memoryless.
        """
        for person in world.persons.values():
            if person.health_state != HealthState.INFECTED:
                continue
            
            old_event = person.pending_event
    
            if old_event is not None:
                world.event_queue.cancel(old_event)
    
                # Avoid leaving a reference to an event that is already
                # logically cancelled if scheduling the replacement fails.
                person.pending_event = None
    
            self._schedule_outcome(
                world,
                person,
            )

    def _handle_recover(
        self,
        world: World,
        event: Event,
    ) -> None:
        """
        Change the person to RECOVERED.

        Immunity is currently permanent, so no later event is
        scheduled.
        """
        person = world.get_person(event.person_id)

        if person.health_state == HealthState.DEAD:
            if person.pending_event is event:
                person.pending_event = None
            return

        person.health_state = HealthState.RECOVERED
        person.pending_event = None

        # Future immunity-waning support:
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
        Change the person to DEAD and remove them from their current
        physical place.
        """
        person = world.get_person(event.person_id)

        person.health_state = HealthState.DEAD
        person.pending_event = None

        world.remove_from_place(person)

    def _handle_immunity_wanes(
        self,
        world: World,
        event: Event,
    ) -> None:
        """
        Return a recovered person to the SUSCEPTIBLE state.
        """
        person = world.get_person(event.person_id)

        if person.health_state == HealthState.DEAD:
            if person.pending_event is event:
                person.pending_event = None
            return

        person.health_state = HealthState.SUSCEPTIBLE
        person.pending_event = None

    @staticmethod
    def _probability_to_hazard(
        probability: float,
        tick_duration: float,
    ) -> float:
        """
        Convert a per-tick probability to a continuous-time hazard:

            lambda = -ln(1 - probability) / tick_duration
        """
        if isinstance(probability, bool):
            raise TypeError(
                "probability must be a number, not bool"
            )

        if not math.isfinite(probability):
            raise ValueError(
                "probability must be finite"
            )

        if not 0.0 <= probability < 1.0:
            raise ValueError(
                "probability must be in [0.0, 1.0)"
            )

        if isinstance(tick_duration, bool):
            raise TypeError(
                "tick_duration must be a number, not bool"
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