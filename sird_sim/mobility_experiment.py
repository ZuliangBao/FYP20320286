from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from . import controller
from .config import SimulationConfig
from .domain.place import PlaceType
from .systems.metrics_system import MetricsSnapshot


@dataclass(frozen=True, slots=True)
class MobilityScenario:
    """One set of mobility parameters."""

    name: str
    contact_k: Mapping[PlaceType, int]
    public_visit_probability_weekday: float
    public_visit_probability_weekend: float


@dataclass(frozen=True, slots=True)
class MobilityResult:
    """Simulation result for one mobility scenario."""

    name: str
    history: tuple[MetricsSnapshot, ...]
    
def run_mobility_comparison(
    *,
    base_config: SimulationConfig,
    scenarios: Sequence[MobilityScenario],
    total_days: float,
    initial_infection_count: int,
) -> list[MobilityResult]:
    if base_config.seed is None:
        raise ValueError(
            "Mobility comparison requires a fixed random seed"
        )

    if not scenarios:
        raise ValueError(
            "At least one mobility scenario is required"
        )

    results: list[MobilityResult] = []

    for scenario in scenarios:
        scenario_config = replace(
            base_config,
            contact_k=dict(scenario.contact_k),
            public_visit_probability_weekday=(
                scenario.public_visit_probability_weekday
            ),
            public_visit_probability_weekend=(
                scenario.public_visit_probability_weekend
            ),
        )

        world = controller.generate(
            scenario_config,
            seed=base_config.seed,
        )

        controller.seed_infections(
            world,
            count=initial_infection_count,
        )

        engine, metrics_system = controller.build_engine(
            world,
            scenario_config,
        )

        controller.run(
            engine,
            total_days,
        )

        results.append(
            MobilityResult(
                name=scenario.name,
                history=tuple(metrics_system.history),
            )
        )

    return results


def draw_mobility_comparison(
    results: Sequence[MobilityResult],
) -> Figure:
    """
    Draw SIRD curves for all mobility scenarios in one figure.

    Each health state gets its own axis to avoid placing twelve or more
    lines on one coordinate system.
    """
    if not results:
        raise ValueError(
            "results cannot be empty"
        )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12, 8),
        sharex=True,
    )

    series = (
        (
            "susceptible",
            "Susceptible",
            axes[0, 0],
        ),
        (
            "infected",
            "Infected",
            axes[0, 1],
        ),
        (
            "recovered",
            "Recovered",
            axes[1, 0],
        ),
        (
            "dead",
            "Dead",
            axes[1, 1],
        ),
    )

    for attribute, title, ax in series:
        for result in results:
            times = [
                snapshot.time / 24.0
                for snapshot in result.history
            ]

            values = [
                getattr(snapshot, attribute)
                for snapshot in result.history
            ]

            ax.plot(
                times,
                values,
                label=result.name,
            )

        ax.set_title(title)
        ax.set_xlabel("Simulation day")
        ax.set_ylabel("Population")
        ax.grid(
            visible=True,
            alpha=0.3,
        )
        ax.legend()

    fig.suptitle(
        "SIRD comparison under different mobility levels"
    )

    fig.tight_layout()
    return fig