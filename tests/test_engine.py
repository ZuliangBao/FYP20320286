from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from sird_sim.config import ImmunityDurationMode
from sird_sim.engine import Engine
from sird_sim.world import World


@dataclass(frozen=True, kw_only=True)
class FakeConfig:
    """
    Minimal dataclass containing only the configuration fields used by Engine.

    Using a dataclass is important because Engine.update_runtime_config()
    calls dataclasses.replace().
    """

    tick_duration: float = 1.0
    infection_probability: float = 0.08
    recovery_rate: float = 0.05
    deadly_rate: float = 0.005
    immunity_duration_mode: ImmunityDurationMode = (
        ImmunityDurationMode.FIXED
    )
    mean_immunity_duration: float = 90.0 * 24.0

    def __post_init__(self) -> None:
        for field_name in (
            "infection_probability",
            "recovery_rate",
            "deadly_rate",
        ):
            value = getattr(self, field_name)

            if not 0.0 <= value < 1.0:
                raise ValueError(
                    f"{field_name} must be in [0, 1)"
                )

        if self.tick_duration <= 0.0:
            raise ValueError(
                "tick_duration must be greater than 0"
            )

        if self.mean_immunity_duration <= 0.0:
            raise ValueError(
                "mean_immunity_duration must be greater than 0"
            )


class RecordingSystem:
    """A fake System that records each step call."""

    def __init__(
        self,
        name: str,
        call_order: list[str],
    ) -> None:
        self.name = name
        self.call_order = call_order
        self.step_calls = 0

    def step(self, world: World) -> None:
        self.step_calls += 1
        self.call_order.append(self.name)


class RecordingHealthEventSystem(RecordingSystem):
    """A fake HealthEventSystem that also records rescheduling calls."""

    def __init__(
        self,
        call_order: list[str],
    ) -> None:
        super().__init__(
            name="health_event",
            call_order=call_order,
        )
        self.infected_reschedule_calls = 0
        self.recovered_reschedule_calls = 0

    def reschedule_all_infected(
        self,
        world: World,
    ) -> None:
        self.infected_reschedule_calls += 1

    def reschedule_all_recovered(
        self,
        world: World,
    ) -> None:
        self.recovered_reschedule_calls += 1


@dataclass
class EngineFixture:
    engine: Engine
    world: World
    health_event_system: RecordingHealthEventSystem
    schedule_system: RecordingSystem
    contact_system: RecordingSystem
    transmission_system: RecordingSystem
    metrics_system: RecordingSystem
    call_order: list[str]


def make_engine(
    *,
    tick_duration: float = 1.0,
    current_time: float = 0.0,
) -> EngineFixture:
    call_order: list[str] = []

    world = World()
    world.config = cast(
        Any,
        FakeConfig(
            tick_duration=tick_duration,
        ),
    )
    world.current_time = current_time

    health_event_system = RecordingHealthEventSystem(
        call_order
    )
    schedule_system = RecordingSystem(
        "schedule",
        call_order,
    )
    contact_system = RecordingSystem(
        "contact",
        call_order,
    )
    transmission_system = RecordingSystem(
        "transmission",
        call_order,
    )
    metrics_system = RecordingSystem(
        "metrics",
        call_order,
    )

    engine = Engine(
        world=world,
        health_event_system=cast(
            Any,
            health_event_system,
        ),
        schedule_system=cast(
            Any,
            schedule_system,
        ),
        contact_system=cast(
            Any,
            contact_system,
        ),
        transmission_system=cast(
            Any,
            transmission_system,
        ),
        metrics_system=cast(
            Any,
            metrics_system,
        ),
    )

    return EngineFixture(
        engine=engine,
        world=world,
        health_event_system=health_event_system,
        schedule_system=schedule_system,
        contact_system=contact_system,
        transmission_system=transmission_system,
        metrics_system=metrics_system,
        call_order=call_order,
    )


def test_step_calls_systems_in_required_order() -> None:
    fixture = make_engine()

    fixture.engine.step()

    assert fixture.call_order == [
        "health_event",
        "schedule",
        "contact",
        "transmission",
        "metrics",
    ]


@pytest.mark.parametrize(
    ("starting_time", "tick_duration"),
    [
        (0.0, 1.0),
        (5.5, 0.25),
        (24.0, 6.0),
    ],
)
def test_step_advances_time_by_exactly_one_tick(
    starting_time: float,
    tick_duration: float,
) -> None:
    fixture = make_engine(
        current_time=starting_time,
        tick_duration=tick_duration,
    )

    fixture.engine.step()

    assert fixture.world.require_current_time() == pytest.approx(
        starting_time + tick_duration
    )


@pytest.mark.parametrize(
    (
        "tick_duration",
        "total_days",
        "expected_step_calls",
    ),
    [
        (1.0, 2.0, 48),
        (0.5, 1.0, 48),
        (6.0, 2.5, 10),
        (24.0, 3.0, 3),
        (1.0, 0.0, 0),
    ],
)
def test_run_calls_step_expected_number_of_times(
    tick_duration: float,
    total_days: float,
    expected_step_calls: int,
) -> None:
    fixture = make_engine(
        tick_duration=tick_duration
    )

    fixture.engine.run(total_days)

    assert (
        fixture.health_event_system.step_calls
        == expected_step_calls
    )
    assert (
        fixture.schedule_system.step_calls
        == expected_step_calls
    )
    assert (
        fixture.contact_system.step_calls
        == expected_step_calls
    )
    assert (
        fixture.transmission_system.step_calls
        == expected_step_calls
    )
    assert (
        fixture.metrics_system.step_calls
        == expected_step_calls
    )


@pytest.mark.parametrize(
    "tick_duration",
    [
        5.0,
        7.0,
        10.0,
    ],
)
def test_run_rejects_tick_duration_that_does_not_divide_day(
    tick_duration: float,
) -> None:
    fixture = make_engine(
        tick_duration=tick_duration
    )

    with pytest.raises(
        ValueError,
        match="must divide 24 hours exactly",
    ):
        fixture.engine.run(1.0)


@pytest.mark.parametrize(
    ("tick_duration", "total_days"),
    [
        (6.0, 0.3),
        (3.0, 0.2),
        (1.0, 1.0 / 48.0),
    ],
)
def test_run_rejects_duration_that_is_not_whole_ticks(
    tick_duration: float,
    total_days: float,
) -> None:
    fixture = make_engine(
        tick_duration=tick_duration
    )

    with pytest.raises(
        ValueError,
        match="whole number of simulation ticks",
    ):
        fixture.engine.run(total_days)


@pytest.mark.parametrize(
    ("tick_duration", "total_days"),
    [
        (6.0, 0.5),
        (3.0, 0.25),
        (0.5, 1.5),
    ],
)
def test_run_accepts_valid_tick_combinations(
    tick_duration: float,
    total_days: float,
) -> None:
    fixture = make_engine(
        tick_duration=tick_duration
    )

    fixture.engine.run(total_days)


@pytest.mark.parametrize(
    ("field_name", "new_value"),
    [
        ("recovery_rate", 0.10),
        ("deadly_rate", 0.02),
    ],
)
def test_updating_outcome_rate_reschedules_infected(
    field_name: str,
    new_value: float,
) -> None:
    fixture = make_engine()

    fixture.engine.update_runtime_config(
        **{
            field_name: new_value,
        }
    )

    assert (
        fixture.health_event_system.infected_reschedule_calls
        == 1
    )
    assert (
        fixture.health_event_system.recovered_reschedule_calls
        == 0
    )
    assert getattr(
        fixture.world.require_config(),
        field_name,
    ) == new_value


def test_updating_unrelated_field_does_not_reschedule() -> None:
    fixture = make_engine()

    fixture.engine.update_runtime_config(
        infection_probability=0.15
    )

    assert (
        fixture.health_event_system.infected_reschedule_calls
        == 0
    )
    assert (
        fixture.health_event_system.recovered_reschedule_calls
        == 0
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "immunity_duration_mode":
                ImmunityDurationMode.EXPONENTIAL,
        },
        {
            "mean_immunity_duration": 30.0 * 24.0,
        },
    ],
)
def test_immunity_change_reschedules_recovered_when_enabled(
    overrides: dict[str, object],
) -> None:
    fixture = make_engine()

    fixture.engine.update_runtime_config(
        apply_immunity_changes_to_recovered=True,
        **overrides,
    )

    assert (
        fixture.health_event_system.recovered_reschedule_calls
        == 1
    )
    assert (
        fixture.health_event_system.infected_reschedule_calls
        == 0
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "immunity_duration_mode":
                ImmunityDurationMode.EXPONENTIAL,
        },
        {
            "mean_immunity_duration": 30.0 * 24.0,
        },
    ],
)
def test_immunity_change_does_not_reschedule_recovered_when_disabled(
    overrides: dict[str, object],
) -> None:
    fixture = make_engine()

    fixture.engine.update_runtime_config(
        apply_immunity_changes_to_recovered=False,
        **overrides,
    )

    assert (
        fixture.health_event_system.recovered_reschedule_calls
        == 0
    )
    assert (
        fixture.health_event_system.infected_reschedule_calls
        == 0
    )


def test_update_runtime_config_without_overrides_is_no_op() -> None:
    fixture = make_engine()
    old_config = fixture.world.config

    fixture.engine.update_runtime_config(
        apply_immunity_changes_to_recovered=True
    )

    assert fixture.world.config is old_config
    assert (
        fixture.health_event_system.infected_reschedule_calls
        == 0
    )
    assert (
        fixture.health_event_system.recovered_reschedule_calls
        == 0
    )


def test_setting_same_values_does_not_reschedule() -> None:
    fixture = make_engine()
    config = fixture.world.require_config()

    fixture.engine.update_runtime_config(
        recovery_rate=config.recovery_rate,
        deadly_rate=config.deadly_rate,
        immunity_duration_mode=(
            config.immunity_duration_mode
        ),
        mean_immunity_duration=(
            config.mean_immunity_duration
        ),
        apply_immunity_changes_to_recovered=True,
    )

    assert (
        fixture.health_event_system.infected_reschedule_calls
        == 0
    )
    assert (
        fixture.health_event_system.recovered_reschedule_calls
        == 0
    )


def test_invalid_config_update_propagates_and_is_atomic() -> None:
    fixture = make_engine()
    old_config = fixture.world.config

    with pytest.raises(
        ValueError,
        match="recovery_rate must be in",
    ):
        fixture.engine.update_runtime_config(
            recovery_rate=-0.1
        )

    # replace() failed before assignment, so the old config remains.
    assert fixture.world.config is old_config
    assert (
        fixture.health_event_system.infected_reschedule_calls
        == 0
    )
    assert (
        fixture.health_event_system.recovered_reschedule_calls
        == 0
    )


def test_update_runtime_config_rejects_non_bool_policy_flag() -> None:
    fixture = make_engine()

    with pytest.raises(
        TypeError,
        match=(
            "apply_immunity_changes_to_recovered must be bool"
        ),
    ):
        fixture.engine.update_runtime_config(
            apply_immunity_changes_to_recovered=cast(
                Any,
                "yes",
            ),
            infection_probability=0.1,
        )
