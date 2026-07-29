# SIRD Agent-Based Epidemic Simulator

[![CI](https://github.com/ZuliangBao/FYP20320286/actions/workflows/ci.yml/badge.svg)](https://github.com/ZuliangBao/FYP20320286/actions/workflows/ci.yml)

An agent-based epidemic simulation over a synthetically generated population with
realistic social structure (households, workplaces, schools, public places) and
multiplex contact networks (family / workmate / schoolmate / friend ties).
The base SIRD model is extended to SIRS via configurable immunity waning.

An interactive Streamlit dashboard lets you generate a population, run the
simulation, tweak runtime parameters mid-run, and compare scenarios (e.g.
different mobility levels) side by side.

## Features

- **Population generation** — households, workplaces, and schools sized from
  configurable distributions; a public-place pool; and a multiplex relationship
  network (family, workmate, schoolmate, friend) with tunable target degrees.
- **Daily scheduling** — each person moves between home, work/school, and
  public places according to role, time of day, and weekday/weekend rules.
- **Contact sampling** — bounded, per-place-type contact sampling (`contact_k`)
  approximates realistic mixing without full O(n²) pairwise contacts.
- **Disease dynamics** — transmission per contact, followed by continuous-time,
  competing-risk recovery/death hazards (Gillespie-style), and configurable
  immunity waning (fixed-duration or exponential) that returns recovered
  people to susceptible — i.e. SIRD extended to SIRS.
- **Live runtime reconfiguration** — recovery/death rates and immunity settings
  can be changed mid-simulation; already-infected (and optionally
  already-recovered) people are rescheduled accordingly, without restarting.
- **Mobility comparison experiments** — run several mobility scenarios from a
  shared random seed to isolate the effect of contact intensity / public-space
  visitation on epidemic outcomes.
- **Visualization** — SIRD curves, place-occupancy over time, the relationship
  network, and generation-time distribution histograms.

## Architecture

The simulation is organized as five independent systems coordinated by a
single `Engine` in a fixed order each tick:

```
ScheduleSystem → ContactSystem → TransmissionSystem → HealthEventSystem → MetricsSystem
```

- `domain/` — `Person`, `Place`, `Relationship` data structures.
- `world.py` — holds all mutable simulation state (persons, places,
  relationships, event queue, RNG, config, current time).
- `events/` — `Event` subclasses and a priority-queue `EventQueue` with
  lazy-deletion cancellation.
- `systems/` — the five systems listed above, each operating only on `World`.
- `engine.py` — orchestrates one simulation tick and exposes
  `run()` / `update_runtime_config()`.
- `generation.py` / `partition.py` — synthesize the initial population, places,
  and relationship network from a `SimulationConfig`.
- `config.py` — the immutable, validated `SimulationConfig` dataclass.
- `controller.py` — thin glue layer between the UI and the engine.
- `plotting.py` — pure Matplotlib chart-drawing functions (no Streamlit calls).
- `view.py` / `main.py` — the Streamlit dashboard.
- `mobility_experiment.py` — runs and plots multi-scenario mobility comparisons.

## Dependencies

All dependencies are declared in `pyproject.toml` and installed together via
`pip install -e ".[dev]"`.

**Runtime**

| Package | Used for |
| --- | --- |
| `numpy` | Random number generation, array-based sampling in generation/partitioning/contact sampling |
| `matplotlib` | All chart rendering (`plotting.py`, `mobility_experiment.py`) |
| `networkx` | Relationship-network graph construction and layout (`plotting.py`) |
| `streamlit` | The interactive dashboard (`view.py`, `main.py`) |

**Development**

| Package | Used for |
| --- | --- |
| `pytest` | Test runner |
| `pytest-cov` | Test coverage measurement |
| `mypy` | Static type checking |
| `types-networkx` | Type stubs for `networkx`, so mypy can check code that uses it |
| `build` | Building distributable packages (`python -m build`) |

## Installation

Requires Python 3.12+.

```bash
pip install -e ".[dev]"
```

On Windows, you can instead double-click `setup.bat` in the project root. It
checks that Python 3.12+ is active, then installs the project and its
dependencies directly into whatever Python environment is currently active
(fast if that environment already has packages like numpy/matplotlib/
streamlit installed).

## Usage

```bash
streamlit run main.py
```

## Testing

```bash
pytest --cov=sird_sim --cov-report=term-missing
```

## Type checking

```bash
mypy sird_sim
```

## Continuous integration

Every push and pull request runs the test suite (with coverage) and mypy via
GitHub Actions (see `.github/workflows/ci.yml`).
