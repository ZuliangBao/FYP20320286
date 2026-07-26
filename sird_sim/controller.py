from __future__ import annotations

from dataclasses import replace
from typing import Any

from .config import SimulationConfig
from .engine import Engine
from .generation import generate_world
from .systems.contact_system import ContactSystem
from .systems.health_event_system import HealthEventSystem
from .systems.metrics_system import MetricsSystem
from .systems.schedule_system import ScheduleSystem
from .systems.transmission_system import TransmissionSystem
from .world import World


def generate(
    config: SimulationConfig,
) -> World:
    """
    Generate a new simulation world from the supplied configuration.

    Exceptions raised during generation are intentionally allowed to
    propagate to the caller.
    """
    return generate_world(config)


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
    world: World,
    **overrides: Any,
) -> None:
    """
    Replace the world's immutable configuration with an updated copy.

    dataclasses.replace() reconstructs SimulationConfig, so its
    __post_init__ validation runs again.
    """
    new_config = replace(
        world.config,
        **overrides,
    )

    world.config = new_config