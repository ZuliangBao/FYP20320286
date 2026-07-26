from __future__ import annotations

from dataclasses import fields, MISSING
import inspect

import numpy as np

from sird_sim.config import SimulationConfig
from sird_sim.domain.person import HealthState
from sird_sim.domain.place import PlaceType
from sird_sim.engine import Engine
from sird_sim.systems.contact_system import ContactSystem
from sird_sim.systems.health_event_system import HealthEventSystem
from sird_sim.systems.metrics_system import MetricsSystem
from sird_sim.systems.schedule_system import ScheduleSystem
from sird_sim.systems.transmission_system import TransmissionSystem

# Adjust this import only if generate_world lives in another module.
from sird_sim.generation import generate_world


SEED = 20260726
TOTAL_DAYS = 3.0
INITIAL_INFECTED = 2


def build_smoke_config() -> SimulationConfig:
    """Build a small deterministic configuration for the end-to-end test."""
    values = {
        # Time and movement schedule
        "tick_duration": 1.0,
        "work_start_hour": 8.0,
        "work_end_hour": 17.0,
        "school_start_hour": 8.0,
        "school_end_hour": 15.0,
        "public_start_hour": 18.0,
        "public_end_hour": 22.0,
        "public_visit_probability_weekday": 0.0,
        "public_visit_probability_weekend": 0.0,

        # Population generation
        "population_size": 24,
        "student_ratio": 0.25,
        "employment_rate": 0.90,
        "school_utilization_rate": 0.80,
        # Fixed household size 4 guarantees at least two susceptible
        # housemates remain after seeding two infected people.
        "household_size_distribution": {4: 1.0},
        "workplace_size_distribution": {4: 1.0},
        "school_size_distribution": {6: 1.0},
        "public_place_count": 2,
        "public_place_capacity": 50,

        # Fixed relationship generation
        "workmate_target_degree": 2,
        "schoolmate_target_degree": 2,
        "workmate_weight": 1.0,
        "schoolmate_weight": 1.0,
        "min_friend_count": 1,
        "max_friend_count": 3,
        "friend_weight": 1.0,

        # Contact generation
        # HOME=3 means every member of a four-person household contacts
        # every other member during the first midnight tick.
        "contact_k": {
            PlaceType.HOME: 3,
            PlaceType.WORKPLACE: 2,
            PlaceType.SCHOOL: 2,
            PlaceType.PUBLIC: 2,
        },

        # Health dynamics
        # 1.0 makes first-tick household transmission deterministic.
        "infection_probability": 1.0,
        "recovery_rate": 0.20,
        "deadly_rate": 0.00,
    }

    # Pass only fields that actually exist in the current dataclass.
    config_fields = {field.name: field for field in fields(SimulationConfig)}
    kwargs = {
        name: value
        for name, value in values.items()
        if name in config_fields
    }

    missing_required = sorted(
        name
        for name, field in config_fields.items()
        if field.default is MISSING
        and field.default_factory is MISSING
        and name not in kwargs
    )

    if missing_required:
        raise RuntimeError(
            "Smoke-test configuration is missing required "
            f"SimulationConfig fields: {missing_required}. "
            "Add test values for those fields in build_smoke_config()."
        )

    return SimulationConfig(**kwargs)


def build_world(config: SimulationConfig):
    """Call generate_world while supporting the usual config/rng signature."""
    signature = inspect.signature(generate_world)
    parameters = signature.parameters

    kwargs = {}
    if "config" in parameters:
        kwargs["config"] = config
    elif "simulation_config" in parameters:
        kwargs["simulation_config"] = config

    rng = np.random.default_rng(SEED)
    if "rng" in parameters:
        kwargs["rng"] = rng
    elif "seed" in parameters:
        kwargs["seed"] = SEED

    if kwargs:
        world = generate_world(**kwargs)
    else:
        # Fallback for a simple positional generate_world(config, rng).
        world = generate_world(config, rng)

    # Keep the test deterministic even if generate_world accepts only config.
    world.rng = rng
    world.current_time = 0.0
    return world


def seed_infections(world, count: int) -> tuple[int, ...]:
    """Seed infections inside one household so transmission is guaranteed."""
    candidate_homes = [
        place
        for place in world.places.values()
        if place.place_type == PlaceType.HOME
        and len(place.occupants) >= count + 1
    ]

    if not candidate_homes:
        raise AssertionError(
            "No household contains enough people to seed infections "
            "and leave at least one susceptible contact."
        )

    home = max(candidate_homes, key=lambda place: len(place.occupants))
    infected_ids = tuple(sorted(home.occupants)[:count])

    for person_id in infected_ids:
        person = world.get_person(person_id)
        person.health_state = HealthState.INFECTED
        person.pending_event = None

    return infected_ids


def assert_population_is_conserved(history, population_size: int) -> None:
    for snapshot in history:
        recorded_population = (
            snapshot.susceptible
            + snapshot.infected
            + snapshot.recovered
            + snapshot.dead
        )
        assert recorded_population == population_size, (
            "Population conservation failed at "
            f"t={snapshot.time}: expected {population_size}, "
            f"recorded {recorded_population}"
        )


def main() -> None:
    config = build_smoke_config()
    world = build_world(config)
    population_size = len(world.persons)

    assert population_size == config.population_size
    assert hasattr(world, "event_queue"), "World is missing event_queue"
    assert hasattr(world, "pending_contacts"), (
        "World is missing pending_contacts"
    )

    seeded_ids = seed_infections(world, INITIAL_INFECTED)

    metrics_system = MetricsSystem()
    engine = Engine(
        world=world,
        health_event_system=HealthEventSystem(),
        schedule_system=ScheduleSystem(),
        contact_system=ContactSystem(),
        transmission_system=TransmissionSystem(),
        metrics_system=metrics_system,
    )

    engine.run(total_days=TOTAL_DAYS)

    history = metrics_system.history
    expected_snapshots = round(
        TOTAL_DAYS * 24.0 / config.tick_duration
    )

    assert len(history) == expected_snapshots, (
        f"Expected {expected_snapshots} metric snapshots, "
        f"received {len(history)}"
    )

    assert_population_is_conserved(history, population_size)

    infected_series = [snapshot.infected for snapshot in history]
    assert infected_series[0] == INITIAL_INFECTED, (
        "The first snapshot should still contain only the manually "
        "seeded infections. Newly transmitted infections are events "
        "and should become INFECTED on the next engine step."
    )

    assert len(set(infected_series)) > 1, (
        "INFECTED count never changed. The end-to-end health/contact/"
        "transmission chain may not be running."
    )

    assert max(infected_series) > INITIAL_INFECTED, (
        "No secondary infection appeared. With infection_probability=1 "
        "and a four-person household, transmission should occur."
    )

    final = history[-1]

    print("End-to-end smoke test passed.")
    print(f"  population: {population_size}")
    print(f"  seeded infected IDs: {seeded_ids}")
    print(f"  simulated days: {TOTAL_DAYS}")
    print(f"  snapshots: {len(history)}")
    print(
        "  infected range: "
        f"{min(infected_series)} -> {max(infected_series)}"
    )
    print(
        "  final SIRD: "
        f"S={final.susceptible}, "
        f"I={final.infected}, "
        f"R={final.recovered}, "
        f"D={final.dead}"
    )
    print(f"  final world time: {world.current_time}")


if __name__ == "__main__":
    main()
