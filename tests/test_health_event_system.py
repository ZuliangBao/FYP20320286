import math
from types import SimpleNamespace

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
    DieEvent,
    ImmunityWanesEvent,
    RecoverEvent,
)
from sird_sim.events.event_queue import EventQueue
from sird_sim.systems.health_event_system import HealthEventSystem
from sird_sim.world import World
from sird_sim.config import ImmunityDurationMode

@pytest.fixture
def minimal_world() -> World:
    """
    Build the smallest reusable World needed by HealthEventSystem tests.

    Person 0 is placed in home 0. The other people are included so tests
    can schedule multiple independent events without state interference.
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
        for person_id in range(4)
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
        rng=np.random.default_rng(20260726),
        config=SimpleNamespace(
            tick_duration=1.0,
            recovery_rate=0.2,
            deadly_rate=0.05,
            immunity_duration_mode=ImmunityDurationMode.EXPONENTIAL,
            mean_immunity_duration=90.0,
        ),
        current_time=0.0,
    )

    # Keep these assignments compatible with World versions where
    # event_queue has not yet been declared as a dataclass field.
    world.event_queue = EventQueue()

    return world


@pytest.fixture
def health_event_system() -> HealthEventSystem:
    return HealthEventSystem()


# ============================================================
# _probability_to_hazard
# ============================================================

def test_probability_to_hazard_zero_probability() -> None:
    hazard = HealthEventSystem._probability_to_hazard(
        probability=0.0,
        tick_duration=1.0,
    )

    assert hazard == 0.0


def test_probability_to_hazard_matches_known_formula_value() -> None:
    hazard = HealthEventSystem._probability_to_hazard(
        probability=0.5,
        tick_duration=1.0,
    )

    assert hazard == pytest.approx(
        -math.log(0.5),
        rel=1e-12,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    "probability",
    [
        True,
        False,
    ],
)
def test_probability_to_hazard_rejects_boolean_probability(
    probability: bool,
) -> None:
    with pytest.raises(TypeError):
        HealthEventSystem._probability_to_hazard(
            probability=probability,
            tick_duration=1.0,
        )


@pytest.mark.parametrize(
    "probability",
    [
        "0.5",
        None,
        object(),
    ],
)
def test_probability_to_hazard_rejects_non_numeric_probability(
    probability: object,
) -> None:
    with pytest.raises(TypeError):
        HealthEventSystem._probability_to_hazard(
            probability=probability,  # type: ignore[arg-type]
            tick_duration=1.0,
        )


@pytest.mark.parametrize(
    "probability",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_probability_to_hazard_rejects_non_finite_probability(
    probability: float,
) -> None:
    with pytest.raises(ValueError):
        HealthEventSystem._probability_to_hazard(
            probability=probability,
            tick_duration=1.0,
        )


@pytest.mark.parametrize(
    "probability",
    [
        -0.01,
        -1.0,
        1.0,
        1.01,
    ],
)
def test_probability_to_hazard_rejects_probability_outside_half_open_interval(
    probability: float,
) -> None:
    with pytest.raises(ValueError):
        HealthEventSystem._probability_to_hazard(
            probability=probability,
            tick_duration=1.0,
        )


@pytest.mark.parametrize(
    "tick_duration",
    [
        True,
        False,
    ],
)
def test_probability_to_hazard_rejects_boolean_tick_duration(
    tick_duration: bool,
) -> None:
    with pytest.raises(TypeError):
        HealthEventSystem._probability_to_hazard(
            probability=0.5,
            tick_duration=tick_duration,
        )


@pytest.mark.parametrize(
    "tick_duration",
    [
        "1.0",
        None,
        object(),
    ],
)
def test_probability_to_hazard_rejects_non_numeric_tick_duration(
    tick_duration: object,
) -> None:
    with pytest.raises(TypeError):
        HealthEventSystem._probability_to_hazard(
            probability=0.5,
            tick_duration=tick_duration,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "tick_duration",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_probability_to_hazard_rejects_non_finite_tick_duration(
    tick_duration: float,
) -> None:
    with pytest.raises(ValueError):
        HealthEventSystem._probability_to_hazard(
            probability=0.5,
            tick_duration=tick_duration,
        )


@pytest.mark.parametrize(
    "tick_duration",
    [
        0.0,
        -0.1,
        -10.0,
    ],
)
def test_probability_to_hazard_rejects_non_positive_tick_duration(
    tick_duration: float,
) -> None:
    with pytest.raises(ValueError):
        HealthEventSystem._probability_to_hazard(
            probability=0.5,
            tick_duration=tick_duration,
        )


# ============================================================
# step scheduling behavior
# ============================================================

def test_step_does_nothing_when_event_queue_is_empty(
    minimal_world: World,
    health_event_system: HealthEventSystem,
) -> None:
    original_states = {
        person_id: person.health_state
        for person_id, person in minimal_world.persons.items()
    }

    health_event_system.step(minimal_world)

    assert minimal_world.event_queue.is_empty()
    assert {
        person_id: person.health_state
        for person_id, person in minimal_world.persons.items()
    } == original_states


def test_step_leaves_future_event_in_queue(
    minimal_world: World,
    health_event_system: HealthEventSystem,
) -> None:
    person = minimal_world.get_person(0)
    person.health_state = HealthState.INFECTED

    event = minimal_world.event_queue.schedule(
        RecoverEvent,
        time=10.0,
        person_id=person.person_id,
    )
    person.pending_event = event
    minimal_world.current_time = 5.0

    health_event_system.step(minimal_world)

    assert len(minimal_world.event_queue) == 1
    assert minimal_world.event_queue.peek_time() == 10.0
    assert person.health_state == HealthState.INFECTED
    assert person.pending_event is event


def test_step_processes_all_due_events_in_one_call(
    minimal_world: World,
    health_event_system: HealthEventSystem,
) -> None:
    recovering_person = minimal_world.get_person(0)
    immunity_waning_person = minimal_world.get_person(1)

    recovering_person.health_state = HealthState.INFECTED
    immunity_waning_person.health_state = HealthState.RECOVERED

    recover_event = minimal_world.event_queue.schedule(
        RecoverEvent,
        time=1.0,
        person_id=recovering_person.person_id,
    )
    immunity_event = minimal_world.event_queue.schedule(
        ImmunityWanesEvent,
        time=3.0,
        person_id=immunity_waning_person.person_id,
    )

    recovering_person.pending_event = recover_event
    immunity_waning_person.pending_event = immunity_event
    minimal_world.current_time = 5.0

    health_event_system.step(minimal_world)

    assert recovering_person.health_state == HealthState.RECOVERED
    assert isinstance(recovering_person.pending_event, ImmunityWanesEvent)

    assert immunity_waning_person.health_state == HealthState.SUSCEPTIBLE
    assert immunity_waning_person.pending_event is None

    # recovering_person now has a freshly scheduled ImmunityWanesEvent,
    # so the queue is no longer empty after this step.
    assert len(minimal_world.event_queue) == 1


# ============================================================
# _handle_become_infectious
# ============================================================

def test_handle_become_infectious_changes_state_and_schedules_outcome(
    minimal_world: World,
    health_event_system: HealthEventSystem,
) -> None:
    person = minimal_world.get_person(0)

    event = BecomeInfectiousEvent(
        time=0.0,
        sequence=0,
        person_id=person.person_id,
    )
    person.pending_event = event

    health_event_system._handle_become_infectious(
        minimal_world,
        event,
    )

    assert person.health_state == HealthState.INFECTED
    assert isinstance(
        person.pending_event,
        (RecoverEvent, DieEvent),
    )
    assert len(minimal_world.event_queue) == 1
    assert person.pending_event.time >= minimal_world.current_time


def test_stale_become_infectious_event_does_not_revive_dead_person_and_clears_itself(
    minimal_world: World,
    health_event_system: HealthEventSystem,
) -> None:
    person = minimal_world.get_person(0)
    person.health_state = HealthState.DEAD

    stale_event = BecomeInfectiousEvent(
        time=0.0,
        sequence=0,
        person_id=person.person_id,
    )
    person.pending_event = stale_event

    health_event_system._handle_become_infectious(
        minimal_world,
        stale_event,
    )

    assert person.health_state == HealthState.DEAD
    assert person.pending_event is None
    assert minimal_world.event_queue.is_empty()


def test_stale_become_infectious_event_does_not_clear_newer_pending_event(
    minimal_world: World,
    health_event_system: HealthEventSystem,
) -> None:
    person = minimal_world.get_person(0)
    person.health_state = HealthState.DEAD

    stale_event = BecomeInfectiousEvent(
        time=0.0,
        sequence=0,
        person_id=person.person_id,
    )
    newer_event = RecoverEvent(
        time=10.0,
        sequence=1,
        person_id=person.person_id,
    )
    person.pending_event = newer_event

    health_event_system._handle_become_infectious(
        minimal_world,
        stale_event,
    )

    assert person.health_state == HealthState.DEAD
    assert person.pending_event is newer_event
    assert minimal_world.event_queue.is_empty()


def test_become_infectious_with_zero_hazards_schedules_no_outcome(
    minimal_world: World,
    health_event_system: HealthEventSystem,
) -> None:
    minimal_world.config.recovery_rate = 0.0
    minimal_world.config.deadly_rate = 0.0

    person = minimal_world.get_person(0)
    event = BecomeInfectiousEvent(
        time=0.0,
        sequence=0,
        person_id=person.person_id,
    )
    person.pending_event = event

    health_event_system._handle_become_infectious(
        minimal_world,
        event,
    )

    assert person.health_state == HealthState.INFECTED
    assert person.pending_event is None
    assert minimal_world.event_queue.is_empty()
    assert len(minimal_world.event_queue) == 0


def test_competing_risks_death_branch_matches_theoretical_probability(
    minimal_world: World,
    health_event_system: HealthEventSystem,
) -> None:
    """
    Statistical regression test for the competing-risks branch choice.

    With fixed parameters, the probability of choosing DieEvent should be:

        lambda_die / (lambda_die + lambda_recover)
    """
    sample_size = 5_000

    recovery_rate = 0.20
    deadly_rate = 0.05
    tick_duration = 1.0

    minimal_world.config.recovery_rate = recovery_rate
    minimal_world.config.deadly_rate = deadly_rate
    minimal_world.config.tick_duration = tick_duration
    minimal_world.current_time = 0.0
    minimal_world.rng = np.random.default_rng(20260726)

    lambda_recover = -math.log1p(-recovery_rate) / tick_duration
    lambda_die = -math.log1p(-deadly_rate) / tick_duration
    expected_death_probability = (
        lambda_die / (lambda_recover + lambda_die)
    )

    person = minimal_world.get_person(0)
    death_count = 0

    for _ in range(sample_size):
        minimal_world.event_queue = EventQueue()
        person.health_state = HealthState.SUSCEPTIBLE
        person.pending_event = None

        become_event = minimal_world.event_queue.schedule(
            BecomeInfectiousEvent,
            time=minimal_world.current_time,
            person_id=person.person_id,
        )
        person.pending_event = become_event

        health_event_system.step(minimal_world)

        outcome_event = person.pending_event

        assert isinstance(
            outcome_event,
            (RecoverEvent, DieEvent),
        )

        if isinstance(outcome_event, DieEvent):
            death_count += 1

    observed_death_probability = death_count / sample_size

    # About 4–5 standard errors for these parameters and sample size.
    tolerance = 0.025

    assert observed_death_probability == pytest.approx(
        expected_death_probability,
        abs=tolerance,
    )


# ============================================================
# _handle_recover
# ============================================================

def test_handle_recover_sets_recovered_and_schedules_immunity_waning(
    minimal_world: World,
    health_event_system: HealthEventSystem,
) -> None:
    person = minimal_world.get_person(0)
    person.health_state = HealthState.INFECTED

    event = RecoverEvent(
        time=1.0,
        sequence=0,
        person_id=person.person_id,
    )
    person.pending_event = event

    health_event_system._handle_recover(
        minimal_world,
        event,
    )

    assert person.health_state == HealthState.RECOVERED
    assert isinstance(person.pending_event, ImmunityWanesEvent)
    assert person.recovered_at == minimal_world.current_time


def test_stale_recover_event_for_dead_person_clears_itself(
    minimal_world: World,
    health_event_system: HealthEventSystem,
) -> None:
    person = minimal_world.get_person(0)
    person.health_state = HealthState.DEAD

    stale_event = RecoverEvent(
        time=1.0,
        sequence=0,
        person_id=person.person_id,
    )
    person.pending_event = stale_event

    health_event_system._handle_recover(
        minimal_world,
        stale_event,
    )

    assert person.health_state == HealthState.DEAD
    assert person.pending_event is None


def test_stale_recover_event_does_not_clear_newer_pending_event(
    minimal_world: World,
    health_event_system: HealthEventSystem,
) -> None:
    person = minimal_world.get_person(0)
    person.health_state = HealthState.DEAD

    stale_event = RecoverEvent(
        time=1.0,
        sequence=0,
        person_id=person.person_id,
    )
    newer_event = DieEvent(
        time=2.0,
        sequence=1,
        person_id=person.person_id,
    )
    person.pending_event = newer_event

    health_event_system._handle_recover(
        minimal_world,
        stale_event,
    )

    assert person.health_state == HealthState.DEAD
    assert person.pending_event is newer_event


# ============================================================
# _handle_die
# ============================================================

def test_handle_die_sets_dead_clears_event_and_removes_person_from_place(
    minimal_world: World,
    health_event_system: HealthEventSystem,
) -> None:
    person = minimal_world.get_person(0)
    home = minimal_world.get_place(0)
    person.health_state = HealthState.INFECTED

    event = DieEvent(
        time=1.0,
        sequence=0,
        person_id=person.person_id,
    )
    person.pending_event = event

    health_event_system._handle_die(
        minimal_world,
        event,
    )

    assert person.health_state == HealthState.DEAD
    assert person.pending_event is None
    assert person.current_place_id is None
    assert person.person_id not in home.occupants


def test_handle_die_is_idempotent_when_processed_twice(
    minimal_world: World,
    health_event_system: HealthEventSystem,
) -> None:
    person = minimal_world.get_person(0)
    home = minimal_world.get_place(0)

    first_event = DieEvent(
        time=1.0,
        sequence=0,
        person_id=person.person_id,
    )
    second_event = DieEvent(
        time=2.0,
        sequence=1,
        person_id=person.person_id,
    )

    health_event_system._handle_die(
        minimal_world,
        first_event,
    )
    health_event_system._handle_die(
        minimal_world,
        second_event,
    )

    assert person.health_state == HealthState.DEAD
    assert person.pending_event is None
    assert person.current_place_id is None
    assert person.person_id not in home.occupants


# ============================================================
# _handle_immunity_wanes
# ============================================================

def test_handle_immunity_wanes_sets_susceptible_and_clears_pending_event(
    minimal_world: World,
    health_event_system: HealthEventSystem,
) -> None:
    person = minimal_world.get_person(0)
    person.health_state = HealthState.RECOVERED

    event = ImmunityWanesEvent(
        time=1.0,
        sequence=0,
        person_id=person.person_id,
    )
    person.pending_event = event

    health_event_system._handle_immunity_wanes(
        minimal_world,
        event,
    )

    assert person.health_state == HealthState.SUSCEPTIBLE
    assert person.pending_event is None


def test_stale_immunity_wanes_event_for_dead_person_clears_itself(
    minimal_world: World,
    health_event_system: HealthEventSystem,
) -> None:
    person = minimal_world.get_person(0)
    person.health_state = HealthState.DEAD

    stale_event = ImmunityWanesEvent(
        time=1.0,
        sequence=0,
        person_id=person.person_id,
    )
    person.pending_event = stale_event

    health_event_system._handle_immunity_wanes(
        minimal_world,
        stale_event,
    )

    assert person.health_state == HealthState.DEAD
    assert person.pending_event is None


def test_stale_immunity_wanes_event_does_not_clear_newer_pending_event(
    minimal_world: World,
    health_event_system: HealthEventSystem,
) -> None:
    person = minimal_world.get_person(0)
    person.health_state = HealthState.DEAD

    stale_event = ImmunityWanesEvent(
        time=1.0,
        sequence=0,
        person_id=person.person_id,
    )
    newer_event = RecoverEvent(
        time=2.0,
        sequence=1,
        person_id=person.person_id,
    )
    person.pending_event = newer_event

    health_event_system._handle_immunity_wanes(
        minimal_world,
        stale_event,
    )

    assert person.health_state == HealthState.DEAD
    assert person.pending_event is newer_event

def test_reschedule_all_infected_replaces_existing_events(
    minimal_world: World,
    health_event_system: HealthEventSystem,
) -> None:
    person = minimal_world.get_person(0)
    person.health_state = HealthState.INFECTED

    old_event = minimal_world.event_queue.schedule(
        RecoverEvent,
        time=100.0,
        person_id=person.person_id,
    )
    person.pending_event = old_event

    health_event_system.reschedule_all_infected(
        minimal_world
    )

    assert old_event.cancelled is True
    assert person.pending_event is not None
    assert person.pending_event is not old_event
    assert len(minimal_world.event_queue) == 1

def test_reschedule_all_infected_skips_non_infected_people(
    minimal_world: World,
    health_event_system: HealthEventSystem,
) -> None:
    person = minimal_world.get_person(0)
    person.health_state = HealthState.SUSCEPTIBLE
    person.pending_event = None

    health_event_system.reschedule_all_infected(
        minimal_world
    )

    assert person.pending_event is None
    assert minimal_world.event_queue.is_empty()

def test_reschedule_all_infected_handles_missing_old_event(
    minimal_world: World,
    health_event_system: HealthEventSystem,
) -> None:
    person = minimal_world.get_person(0)
    person.health_state = HealthState.INFECTED
    person.pending_event = None

    health_event_system.reschedule_all_infected(
        minimal_world
    )

    assert isinstance(
        person.pending_event,
        (RecoverEvent, DieEvent),
    )
    assert len(minimal_world.event_queue) == 1

def test_reschedule_all_infected_with_zero_hazards_cancels_old_event_only(
    minimal_world: World,
    health_event_system: HealthEventSystem,
) -> None:
    minimal_world.config.recovery_rate = 0.0
    minimal_world.config.deadly_rate = 0.0

    person = minimal_world.get_person(0)
    person.health_state = HealthState.INFECTED

    old_event = minimal_world.event_queue.schedule(
        RecoverEvent,
        time=100.0,
        person_id=person.person_id,
    )
    person.pending_event = old_event

    health_event_system.reschedule_all_infected(
        minimal_world
    )

    assert old_event.cancelled is True
    assert person.pending_event is None
    assert minimal_world.event_queue.is_empty()
