from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any

from .config import SimulationConfig
from .engine import Engine
from .events.event import BecomeInfectiousEvent
from .generation import generate_world
from .systems.contact_system import ContactSystem
from .systems.health_event_system import HealthEventSystem
from .systems.metrics_system import MetricsSystem
from .systems.schedule_system import ScheduleSystem
from .systems.transmission_system import TransmissionSystem
from .world import World
import numpy as np

def generate(
    config: SimulationConfig,
    seed: int | None = None
) -> World:
    """
    Generate a new simulation world from the supplied configuration.

    Exceptions raised during generation are intentionally allowed to
    propagate to the caller.
    """

    return generate_world(config,np.random.default_rng(seed))

def build_engine(
    world: World,
    config: SimulationConfig,
) -> tuple[Engine, MetricsSystem]:
    """
    Build the simulation systems and connect them to an Engine.

    The MetricsSystem is returned separately because the UI needs to
    retain its stateful history after the simulation runs.
    """
    schedule_system = ScheduleSystem()
    contact_system = ContactSystem()
    health_event_system = HealthEventSystem()
    transmission_system = TransmissionSystem()
    metrics_system = MetricsSystem()

    # generate_world(config) normally already stores this config in
    # world. Assigning it here also keeps build_engine(world, config)
    # correct when a manually constructed World is supplied.
    world.config = config

    engine = Engine(
        world=world,
        health_event_system=health_event_system,
        schedule_system=schedule_system,
        contact_system=contact_system,
        transmission_system=transmission_system,
        metrics_system=metrics_system,
    )

    return engine, metrics_system

def run(
    engine: Engine,
    total_days: float,
) -> None:
    """
    Continue running the existing engine for the requested duration.
    """
    engine.run(total_days)

def update_runtime_config(
    engine: Engine,
    *,
    apply_immunity_changes_to_recovered: bool = False,
    **overrides: Any,
) -> None:
    """
    Update persistent runtime configuration.

    See Engine.update_runtime_config for the exact rescheduling
    semantics (which config changes trigger which retroactive effects).
    """
    engine.update_runtime_config(
        apply_immunity_changes_to_recovered=(
            apply_immunity_changes_to_recovered
        ),
        **overrides,
    )

def seed_infections(
    world: World,
    count: int,
) -> None:
    """
    Schedule initial infections in distinct households.

    Exactly one person is selected from each sampled household. Health
    state is not changed directly; BecomeInfectiousEvent is processed by
    HealthEventSystem during the first engine step.

    Households and their members are processed in sorted order so that
    seeded runs are reproducible regardless of dict iteration order.
    """
    
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError("count must be a non-negative integer")

    if count < 0:
        raise ValueError("count must be a non-negative integer")

    current_time = world.require_current_time()

    members_by_home: dict[int, list[int]] = defaultdict(list)

    for person in world.persons.values():
        members_by_home[person.home_id].append(
            person.person_id
        )

    household_count = len(members_by_home)

    if count > household_count:
        raise ValueError(
            f"Cannot seed {count} infections across only "
            f"{household_count} households"
        )

    if count == 0:
        return

    # Sorting prevents person-dictionary insertion order from affecting
    # seeded results for the same RNG seed.
    home_ids = sorted(members_by_home)

    selected_home_ids = world.rng.choice(
        home_ids,
        size=count,
        replace=False,
    ).tolist()

    for home_id in selected_home_ids:
        member_ids = sorted(members_by_home[int(home_id)])

        selected_person_id = int(world.rng.choice(member_ids))

        event = world.event_queue.schedule(
            BecomeInfectiousEvent,
            time=current_time,
            person_id=selected_person_id,
        )

        world.persons[selected_person_id].pending_event = event