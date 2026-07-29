from __future__ import annotations

from dataclasses import replace

import matplotlib

# Use a non-interactive backend during pytest.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from sird_sim.config import (
    ImmunityDurationMode,
    SimulationConfig,
)
from sird_sim.domain.place import PlaceType
from sird_sim.mobility_experiment import (
    MobilityResult,
    MobilityScenario,
    draw_mobility_comparison,
    run_mobility_comparison,
)
from sird_sim.systems.metrics_system import MetricsSnapshot


@pytest.fixture
def base_config() -> SimulationConfig:
    """
    Small but strongly transmissible configuration for integration tests.

    The low-mobility scenario below has contact_k=0 everywhere, so its
    infected peak should stay near the initially seeded infections.
    The high-mobility scenario enables many contacts, making a different
    epidemic curve very likely and deterministic under the fixed seed.
    """
    return SimulationConfig(
        population_size=120,
        tick_duration=1.0,
        seed=20260729,

        student_ratio=0.25,
        employment_rate=0.80,
        school_utilization_rate=0.80,

        household_size_distribution={
            1: 0.20,
            2: 0.35,
            3: 0.30,
            4: 0.15,
        },
        workplace_size_distribution={
            5: 0.50,
            10: 0.50,
        },
        school_size_distribution={
            10: 0.50,
            20: 0.50,
        },

        public_place_count=4,
        public_place_capacity=120,

        workmate_target_degree=3,
        schoolmate_target_degree=4,
        workmate_weight=1.0,
        schoolmate_weight=1.0,
        min_friend_count=1,
        max_friend_count=3,
        friend_weight=1.0,

        work_start_hour=8.0,
        work_end_hour=17.0,
        school_start_hour=8.0,
        school_end_hour=15.0,
        public_start_hour=18.0,
        public_end_hour=22.0,

        public_visit_probability_weekday=0.20,
        public_visit_probability_weekend=0.40,

        contact_k={
            PlaceType.HOME: 3,
            PlaceType.WORKPLACE: 5,
            PlaceType.SCHOOL: 8,
            PlaceType.PUBLIC: 4,
        },

        # Deliberately strong transmission and slow removal make the
        # mobility difference visible in a short test run.
        infection_probability=0.60,
        recovery_rate=0.002,
        deadly_rate=0.0001,

        immunity_duration_mode=ImmunityDurationMode.FIXED,
        mean_immunity_duration=365.0 * 24.0,
    )


@pytest.fixture
def low_mobility_scenario() -> MobilityScenario:
    return MobilityScenario(
        name="Low mobility",
        contact_k={
            PlaceType.HOME: 0,
            PlaceType.WORKPLACE: 0,
            PlaceType.SCHOOL: 0,
            PlaceType.PUBLIC: 0,
        },
        public_visit_probability_weekday=0.20,
        public_visit_probability_weekend=0.40,
    )


@pytest.fixture
def high_mobility_scenario() -> MobilityScenario:
    return MobilityScenario(
        name="High mobility",
        contact_k={
            PlaceType.HOME: 8,
            PlaceType.WORKPLACE: 15,
            PlaceType.SCHOOL: 20,
            PlaceType.PUBLIC: 15,
        },
        public_visit_probability_weekday=0.20,
        public_visit_probability_weekend=0.40,
    )


def test_run_mobility_comparison_rejects_empty_scenarios(
    base_config: SimulationConfig,
) -> None:
    with pytest.raises(
        ValueError,
        match="At least one mobility scenario is required",
    ):
        run_mobility_comparison(
            base_config=base_config,
            scenarios=[],
            total_days=1.0,
            initial_infection_count=1,
        )


def test_run_mobility_comparison_requires_fixed_seed(
    base_config: SimulationConfig,
    low_mobility_scenario: MobilityScenario,
) -> None:
    config_without_seed = replace(
        base_config,
        seed=None,
    )

    with pytest.raises(
        ValueError,
        match="requires a fixed random seed",
    ):
        run_mobility_comparison(
            base_config=config_without_seed,
            scenarios=[low_mobility_scenario],
            total_days=1.0,
            initial_infection_count=1,
        )


def test_different_contact_k_values_change_epidemic_curve(
    base_config: SimulationConfig,
    low_mobility_scenario: MobilityScenario,
    high_mobility_scenario: MobilityScenario,
) -> None:
    results = run_mobility_comparison(
        base_config=base_config,
        scenarios=[
            low_mobility_scenario,
            high_mobility_scenario,
        ],
        total_days=7.0,
        initial_infection_count=3,
    )

    assert len(results) == 2
    assert results[0].history
    assert len(results[0].history) == len(results[1].history)

    low_peak = max(
        snapshot.infected
        for snapshot in results[0].history
    )
    high_peak = max(
        snapshot.infected
        for snapshot in results[1].history
    )

    assert high_peak > low_peak


def test_identical_scenarios_are_exactly_reproducible(
    base_config: SimulationConfig,
    high_mobility_scenario: MobilityScenario,
) -> None:
    duplicate_scenario = MobilityScenario(
        name="High mobility duplicate",
        contact_k=dict(high_mobility_scenario.contact_k),
        public_visit_probability_weekday=(
            high_mobility_scenario
            .public_visit_probability_weekday
        ),
        public_visit_probability_weekend=(
            high_mobility_scenario
            .public_visit_probability_weekend
        ),
    )

    results = run_mobility_comparison(
        base_config=base_config,
        scenarios=[
            high_mobility_scenario,
            duplicate_scenario,
        ],
        total_days=5.0,
        initial_infection_count=3,
    )

    assert len(results) == 2
    assert results[0].history
    assert results[0].history == results[1].history


def test_draw_mobility_comparison_rejects_empty_results() -> None:
    with pytest.raises(
        ValueError,
        match="results cannot be empty",
    ):
        draw_mobility_comparison([])


def test_draw_mobility_comparison_creates_four_axes_and_one_line_per_scenario(
) -> None:
    first_history = (
        MetricsSnapshot(
            time=0.0,
            susceptible=98,
            infected=2,
            recovered=0,
            dead=0,
        ),
        MetricsSnapshot(
            time=24.0,
            susceptible=95,
            infected=4,
            recovered=1,
            dead=0,
        ),
    )

    second_history = (
        MetricsSnapshot(
            time=0.0,
            susceptible=98,
            infected=2,
            recovered=0,
            dead=0,
        ),
        MetricsSnapshot(
            time=24.0,
            susceptible=90,
            infected=8,
            recovered=2,
            dead=0,
        ),
    )

    results = [
        MobilityResult(
            name="Low mobility",
            history=first_history,
        ),
        MobilityResult(
            name="High mobility",
            history=second_history,
        ),
    ]

    figure = draw_mobility_comparison(results)

    try:
        assert len(figure.axes) == 4

        for axis in figure.axes:
            assert len(axis.lines) == len(results)
    finally:
        plt.close(figure)
