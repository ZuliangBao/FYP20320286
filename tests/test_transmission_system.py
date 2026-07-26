import math
from itertools import product
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pytest

from sird_sim.domain.person import (
    HealthState,
    Person,
    Role,
)
from sird_sim.domain.place import Place, PlaceType
from sird_sim.events.event import (
    BecomeInfectiousEvent,
)
from sird_sim.events.event_queue import EventQueue
from sird_sim.systems.transmission_system import TransmissionSystem
from sird_sim.world import World


class ControlledIterationSet(set):
    """
    A set with deliberately controlled iteration order.

    It is used only to verify that TransmissionSystem sorts contacts
    before consuming random numbers. Both instances can contain exactly
    the same pairs while yielding them in opposite orders.
    """

    def __init__(
        self,
        values: Iterable[tuple[int, int]],
        iteration_order: Iterable[tuple[int, int]],
    ) -> None:
        values_tuple = tuple(values)
        iteration_order_tuple = tuple(iteration_order)

        super().__init__(values_tuple)

        if set(iteration_order_tuple) != set(values_tuple):
            raise ValueError(
                "iteration_order must contain exactly the set values"
            )

        self._iteration_order = iteration_order_tuple

    def __iter__(self):
        return iter(self._iteration_order)


def _build_world(
    *,
    seed: int = 20260726,
    infection_probability: float = 0.3,
) -> World:
    """
    Build a minimal reusable World for TransmissionSystem tests.

    Person IDs 0 and 1 are normally used as susceptible people.
    Person IDs 2, 3, and 4 are available as independent infection
    sources. All people occupy one home because TransmissionSystem only
    needs persons, contacts, configuration, RNG, time, and EventQueue.
    """
    home_id = 0

    persons = {
        person_id: Person(
            person_id=person_id,
            role=Role.WORKER,
            home_id=home_id,
            current_place_id=home_id,
            health_state=HealthState.SUSCEPTIBLE,
        )
        for person_id in range(5)
    }

    home = Place(
        place_id=home_id,
        place_type=PlaceType.HOME,
        occupants=set(persons),
    )

    world = World(
        persons=persons,
        places={home_id: home},
        relationships={
            person_id: []
            for person_id in persons
        },
        rng=np.random.default_rng(seed),
        config=SimpleNamespace(
            infection_probability=infection_probability,
        ),
        current_time=0.0,
    )

    # Compatible with World versions where these fields were added
    # after the original dataclass definition.
    world.event_queue = EventQueue()
    world.pending_contacts = set()

    return world


@pytest.fixture
def minimal_world() -> World:
    return _build_world()


@pytest.fixture
def transmission_system() -> TransmissionSystem:
    return TransmissionSystem()


def _set_person_state(
    world: World,
    person_id: int,
    state: HealthState,
) -> Person:
    person = world.get_person(person_id)
    person.health_state = state
    person.pending_event = None
    return person


def _pending_event_signature(
    world: World,
    person_ids: Iterable[int],
) -> tuple:
    """
    Return a deterministic representation of pending infection events.
    """
    signature = []

    for person_id in sorted(person_ids):
        event = world.get_person(person_id).pending_event

        if event is None:
            signature.append((person_id, None))
            continue

        signature.append(
            (
                person_id,
                type(event).__name__,
                event.time,
                event.sequence,
                event.person_id,
                getattr(event, "source_person_id", None),
            )
        )

    return tuple(signature)


# ============================================================
# _get_transmission_pair
# ============================================================

ALL_STATE_PAIRS = list(
    product(
        list(HealthState),
        repeat=2,
    )
)


@pytest.mark.parametrize(
    ("state_a", "state_b"),
    ALL_STATE_PAIRS,
)
def test_get_transmission_pair_covers_all_health_state_combinations(
    state_a: HealthState,
    state_b: HealthState,
    minimal_world: World,
) -> None:
    person_a = _set_person_state(
        minimal_world,
        0,
        state_a,
    )
    person_b = _set_person_state(
        minimal_world,
        1,
        state_b,
    )

    result = TransmissionSystem._get_transmission_pair(
        person_a,
        person_b,
    )

    if (
        state_a == HealthState.SUSCEPTIBLE
        and state_b == HealthState.INFECTED
    ):
        assert result == (person_a, person_b)
    elif (
        state_a == HealthState.INFECTED
        and state_b == HealthState.SUSCEPTIBLE
    ):
        assert result == (person_b, person_a)
    else:
        assert result is None


# ============================================================
# _transmission_probability
# ============================================================

@pytest.mark.parametrize(
    "configured_probability",
    [
        0.17,
        0.83,
    ],
)
def test_transmission_probability_returns_config_value_only(
    configured_probability: float,
    minimal_world: World,
) -> None:
    minimal_world.config.infection_probability = (
        configured_probability
    )

    first_person = minimal_world.get_person(0)
    second_person = minimal_world.get_person(1)

    first_person.health_state = HealthState.DEAD
    second_person.health_state = HealthState.RECOVERED

    first_result = TransmissionSystem._transmission_probability(
        world=minimal_world,
        susceptible=first_person,
        infected=second_person,
    )

    # Swap the objects and change their states. The current placeholder
    # implementation is intentionally independent of Person data.
    first_person.health_state = HealthState.INFECTED
    second_person.health_state = HealthState.SUSCEPTIBLE

    second_result = TransmissionSystem._transmission_probability(
        world=minimal_world,
        susceptible=second_person,
        infected=first_person,
    )

    assert first_result == configured_probability
    assert second_result == configured_probability


# ============================================================
# _validate_probability
# ============================================================

@pytest.mark.parametrize(
    "probability",
    [
        0.0,
        0.5,
        1.0,
    ],
)
def test_validate_probability_accepts_closed_interval(
    probability: float,
) -> None:
    TransmissionSystem._validate_probability(probability)


@pytest.mark.parametrize(
    "probability",
    [
        True,
        False,
    ],
)
def test_validate_probability_rejects_bool(
    probability: bool,
) -> None:
    with pytest.raises(TypeError):
        TransmissionSystem._validate_probability(
            probability
        )


@pytest.mark.parametrize(
    "probability",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_validate_probability_rejects_non_finite_values(
    probability: float,
) -> None:
    with pytest.raises(ValueError):
        TransmissionSystem._validate_probability(
            probability
        )


@pytest.mark.parametrize(
    "probability",
    [
        -0.0001,
        -1.0,
        1.0001,
        2.0,
    ],
)
def test_validate_probability_rejects_values_outside_closed_interval(
    probability: float,
) -> None:
    with pytest.raises(ValueError):
        TransmissionSystem._validate_probability(
            probability
        )


# ============================================================
# step: deterministic boundaries and event fields
# ============================================================

def test_step_with_probability_one_always_schedules_infection(
    minimal_world: World,
    transmission_system: TransmissionSystem,
) -> None:
    susceptible = _set_person_state(
        minimal_world,
        0,
        HealthState.SUSCEPTIBLE,
    )
    infected = _set_person_state(
        minimal_world,
        2,
        HealthState.INFECTED,
    )

    minimal_world.config.infection_probability = 1.0
    minimal_world.current_time = 12.5
    minimal_world.pending_contacts = {
        (susceptible.person_id, infected.person_id)
    }

    transmission_system.step(minimal_world)

    event = susceptible.pending_event

    assert isinstance(
        event,
        BecomeInfectiousEvent,
    )
    assert event.time == minimal_world.current_time
    assert event.person_id == susceptible.person_id
    assert event.source_person_id == infected.person_id
    assert len(minimal_world.event_queue) == 1
    assert minimal_world.event_queue.peek_time() == 12.5


def test_step_with_probability_zero_never_schedules_infection(
    minimal_world: World,
    transmission_system: TransmissionSystem,
) -> None:
    susceptible = _set_person_state(
        minimal_world,
        0,
        HealthState.SUSCEPTIBLE,
    )
    infected = _set_person_state(
        minimal_world,
        2,
        HealthState.INFECTED,
    )

    minimal_world.config.infection_probability = 0.0
    minimal_world.pending_contacts = {
        (susceptible.person_id, infected.person_id)
    }

    transmission_system.step(minimal_world)

    assert susceptible.pending_event is None
    assert minimal_world.event_queue.is_empty()
    assert len(minimal_world.event_queue) == 0


def test_step_does_nothing_when_pending_contacts_is_empty(
    minimal_world: World,
    transmission_system: TransmissionSystem,
) -> None:
    original_states = {
        person_id: person.health_state
        for person_id, person in minimal_world.persons.items()
    }

    minimal_world.pending_contacts = set()

    transmission_system.step(minimal_world)

    assert minimal_world.event_queue.is_empty()
    assert all(
        person.pending_event is None
        for person in minimal_world.persons.values()
    )
    assert {
        person_id: person.health_state
        for person_id, person in minimal_world.persons.items()
    } == original_states


@pytest.mark.parametrize(
    ("state_a", "state_b"),
    [
        (
            HealthState.SUSCEPTIBLE,
            HealthState.SUSCEPTIBLE,
        ),
        (
            HealthState.INFECTED,
            HealthState.INFECTED,
        ),
        (
            HealthState.RECOVERED,
            HealthState.INFECTED,
        ),
        (
            HealthState.DEAD,
            HealthState.INFECTED,
        ),
        (
            HealthState.RECOVERED,
            HealthState.SUSCEPTIBLE,
        ),
        (
            HealthState.DEAD,
            HealthState.SUSCEPTIBLE,
        ),
    ],
)
def test_step_skips_non_transmitting_health_state_pairs(
    state_a: HealthState,
    state_b: HealthState,
    minimal_world: World,
    transmission_system: TransmissionSystem,
) -> None:
    person_a = _set_person_state(
        minimal_world,
        0,
        state_a,
    )
    person_b = _set_person_state(
        minimal_world,
        1,
        state_b,
    )

    minimal_world.config.infection_probability = 1.0
    minimal_world.pending_contacts = {
        (person_a.person_id, person_b.person_id)
    }

    transmission_system.step(minimal_world)

    assert person_a.pending_event is None
    assert person_b.pending_event is None
    assert minimal_world.event_queue.is_empty()


# ============================================================
# pending_event guard
# ============================================================

def test_step_skips_susceptible_with_existing_pending_event(
    minimal_world: World,
    transmission_system: TransmissionSystem,
) -> None:
    susceptible = _set_person_state(
        minimal_world,
        0,
        HealthState.SUSCEPTIBLE,
    )
    infected = _set_person_state(
        minimal_world,
        2,
        HealthState.INFECTED,
    )

    existing_event = minimal_world.event_queue.schedule(
        BecomeInfectiousEvent,
        time=99.0,
        person_id=susceptible.person_id,
        source_person_id=3,
    )
    susceptible.pending_event = existing_event

    minimal_world.config.infection_probability = 1.0
    minimal_world.pending_contacts = {
        (susceptible.person_id, infected.person_id)
    }

    original_queue_length = len(
        minimal_world.event_queue
    )

    transmission_system.step(minimal_world)

    assert len(minimal_world.event_queue) == (
        original_queue_length
    )
    assert susceptible.pending_event is existing_event
    assert existing_event.source_person_id == 3


# ============================================================
# statistical test: multiple-contact compound probability
# ============================================================

@pytest.mark.parametrize(
    "infected_source_count",
    [
        1,
        3,
    ],
)
def test_multiple_contacts_match_compound_infection_probability(
    infected_source_count: int,
) -> None:
    """
    For k independent infected contacts with per-contact probability p,
    the probability of at least one successful transmission is:

        1 - (1 - p) ** k

    The tolerance is five binomial standard errors rather than an
    arbitrary fixed percentage.
    """
    sample_size = 5_000
    per_contact_probability = 0.30

    world = _build_world(
        seed=20260726 + infected_source_count,
        infection_probability=per_contact_probability,
    )
    system = TransmissionSystem()

    susceptible = _set_person_state(
        world,
        0,
        HealthState.SUSCEPTIBLE,
    )

    infected_ids = list(
        range(
            2,
            2 + infected_source_count,
        )
    )

    for infected_id in infected_ids:
        _set_person_state(
            world,
            infected_id,
            HealthState.INFECTED,
        )

    contacts = {
        (susceptible.person_id, infected_id)
        for infected_id in infected_ids
    }

    successful_trials = 0

    for _ in range(sample_size):
        world.event_queue = EventQueue()
        world.pending_contacts = contacts
        susceptible.health_state = (
            HealthState.SUSCEPTIBLE
        )
        susceptible.pending_event = None

        system.step(world)

        if isinstance(
            susceptible.pending_event,
            BecomeInfectiousEvent,
        ):
            successful_trials += 1

    observed_probability = (
        successful_trials / sample_size
    )

    expected_probability = (
        1.0
        - (1.0 - per_contact_probability)
        ** infected_source_count
    )

    standard_error = math.sqrt(
        expected_probability
        * (1.0 - expected_probability)
        / sample_size
    )

    tolerance = 5.0 * standard_error

    assert observed_probability == pytest.approx(
        expected_probability,
        abs=tolerance,
    )


# ============================================================
# reproducibility
# ============================================================

def test_same_seed_and_contacts_produce_identical_results() -> None:
    contacts = {
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
    }

    first_world = _build_world(
        seed=1234,
        infection_probability=0.45,
    )
    second_world = _build_world(
        seed=1234,
        infection_probability=0.45,
    )

    for world in (first_world, second_world):
        _set_person_state(
            world,
            0,
            HealthState.SUSCEPTIBLE,
        )
        _set_person_state(
            world,
            1,
            HealthState.SUSCEPTIBLE,
        )
        _set_person_state(
            world,
            2,
            HealthState.INFECTED,
        )
        _set_person_state(
            world,
            3,
            HealthState.INFECTED,
        )
        world.pending_contacts = set(contacts)

    first_system = TransmissionSystem()
    second_system = TransmissionSystem()

    first_system.step(first_world)
    second_system.step(second_world)

    assert _pending_event_signature(
        first_world,
        [0, 1],
    ) == _pending_event_signature(
        second_world,
        [0, 1],
    )

    assert len(first_world.event_queue) == len(
        second_world.event_queue
    )


def test_sorted_contacts_make_different_iteration_orders_reproducible() -> None:
    """
    The two contact containers have exactly the same set contents but
    deliberately expose opposite iteration orders.

    With sorting inside TransmissionSystem.step(), both runs consume RNG
    in the same contact order and therefore produce identical events.
    Without sorting, this test is designed to produce different source
    IDs and fail.
    """
    contact_order = [
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
    ]
    reversed_order = list(
        reversed(contact_order)
    )

    first_world = _build_world(
        seed=42,
        infection_probability=0.80,
    )
    second_world = _build_world(
        seed=42,
        infection_probability=0.80,
    )

    for world in (first_world, second_world):
        _set_person_state(
            world,
            0,
            HealthState.SUSCEPTIBLE,
        )
        _set_person_state(
            world,
            1,
            HealthState.SUSCEPTIBLE,
        )
        _set_person_state(
            world,
            2,
            HealthState.INFECTED,
        )
        _set_person_state(
            world,
            3,
            HealthState.INFECTED,
        )

    first_world.pending_contacts = ControlledIterationSet(
        contact_order,
        contact_order,
    )
    second_world.pending_contacts = ControlledIterationSet(
        contact_order,
        reversed_order,
    )

    assert first_world.pending_contacts == (
        second_world.pending_contacts
    )
    assert list(first_world.pending_contacts) != list(
        second_world.pending_contacts
    )

    TransmissionSystem().step(first_world)
    TransmissionSystem().step(second_world)

    first_signature = _pending_event_signature(
        first_world,
        [0, 1],
    )
    second_signature = _pending_event_signature(
        second_world,
        [0, 1],
    )

    assert first_signature == second_signature

    # This also makes the expected sorted behavior explicit:
    # contacts with source 2 are considered before source 3.
    assert (
        first_world.get_person(0)
        .pending_event.source_person_id
        == 2
    )
    assert (
        first_world.get_person(1)
        .pending_event.source_person_id
        == 2
    )
