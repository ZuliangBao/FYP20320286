from __future__ import annotations

import math

from ..domain.person import (
    HealthState,
    Person,
)
from ..events.event import BecomeInfectiousEvent
from ..world import World


class TransmissionSystem:
    """
    Process disease transmission through the current contact pairs.

    This system does not immediately change a susceptible person's
    health state. A successful transmission schedules a
    BecomeInfectiousEvent instead.
    """

    def step(self, world: World) -> None:
        """
        Process all contact pairs generated for the current step.

        Pairs are processed in sorted order (not set iteration order) so
        that seeded runs remain deterministic. A susceptible person who
        was already infected earlier in this same step is skipped, since
        their BecomeInfectiousEvent has already been scheduled.
        """
        if world.config is None:
            raise RuntimeError(
                "world.config must be set before "
                "TransmissionSystem.step()"
            )

        if world.current_time is None:
            raise RuntimeError(
                "world.current_time must be set before "
                "TransmissionSystem.step()"
            )

        # Sorting makes seeded runs deterministic even though
        # pending_contacts is a set.
        for person_a_id, person_b_id in sorted(
            world.pending_contacts
        ):
            person_a = world.get_person(person_a_id)
            person_b = world.get_person(person_b_id)

            transmission_pair = self._get_transmission_pair(person_a,person_b)

            if transmission_pair is None:
                continue

            susceptible, infected = transmission_pair

            # A successful earlier contact in this same simulation step
            # has already scheduled this person's infection event.
            if susceptible.pending_event is not None:
                continue

            probability = self._transmission_probability(
                world=world,
                susceptible=susceptible,
                infected=infected,
            )

            self._validate_probability(probability)

            if world.rng.random() >= probability:
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
        Return:
            (susceptible_person, infected_person)

        only when the pair contains exactly one susceptible person and
        one infected person.
        """
        if (
            person_a.health_state
            == HealthState.SUSCEPTIBLE
            and person_b.health_state
            == HealthState.INFECTED
        ):
            return person_a, person_b

        if (
            person_b.health_state
            == HealthState.SUSCEPTIBLE
            and person_a.health_state
            == HealthState.INFECTED
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
        Return the transmission probability for one contact.

        susceptible and infected are retained for future extensions,
        such as:

        - different susceptibility values
        - different infectiousness values
        - place-specific transmission
        - relationship weights
        """
        return world.require_config().infection_probability

    @staticmethod
    def _validate_probability(
        probability: float,
    ) -> None:
        if isinstance(probability, bool):
            raise TypeError(
                "Transmission probability must be a number, "
                "not bool"
            )

        if not math.isfinite(probability):
            raise ValueError("Transmission probability must be finite")

        if not 0.0 <= probability <= 1.0:
            raise ValueError("Transmission probability must be in [0.0, 1.0]")