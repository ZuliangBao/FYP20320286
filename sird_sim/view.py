from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import streamlit as st

from . import controller, plotting
from .config import SimulationConfig
from .domain.place import PlaceType


TICK_DURATION_OPTIONS = [0.25, 0.5, 1.0, 3.0, 6.0, 12.0, 24.0]

DEFAULT_GENERATION: dict[str, Any] = {
    "population_size": 1_000,
    "tick_duration": 1.0,
    "student_ratio": 0.25,
    "employment_rate": 0.90,
    "school_utilization_rate": 0.80,
    "household_size_distribution": {1: 0.25, 2: 0.35, 3: 0.25, 4: 0.15},
    "workplace_size_distribution": {5: 0.20, 10: 0.35, 20: 0.30, 50: 0.15},
    "school_size_distribution": {20: 0.25, 30: 0.40, 40: 0.25, 60: 0.10},
    "public_place_count": 10,
    "public_place_capacity": 200,
    "workmate_target_degree": 4,
    "schoolmate_target_degree": 6,
    "workmate_weight": 1.0,
    "schoolmate_weight": 1.0,
    "min_friend_count": 3,
    "max_friend_count": 8,
    "friend_weight": 1.0,
}

DEFAULT_RUNTIME: dict[str, Any] = {
    "work_start_hour": 8.0,
    "work_end_hour": 17.0,
    "school_start_hour": 8.0,
    "school_end_hour": 15.0,
    "public_start_hour": 18.0,
    "public_end_hour": 22.0,
    "public_visit_probability_weekday": 0.20,
    "public_visit_probability_weekend": 0.45,
    "contact_k": {
        PlaceType.HOME: 3,
        PlaceType.WORKPLACE: 5,
        PlaceType.SCHOOL: 8,
        PlaceType.PUBLIC: 4,
    },
    "infection_probability": 0.08,
    "recovery_rate": 0.05,
    "deadly_rate": 0.005,
}


def render() -> None:
    """Render the complete Streamlit page."""
    st.set_page_config(
        page_title="SIRD Agent-Based Simulator",
        layout="wide",
    )
    _initialize_session_state()
    _show_notice()

    st.title("SIRD Agent-Based Epidemic Simulator")

    with st.sidebar:
        generation_values = _render_generation_controls()
        runtime_values = _render_runtime_controls()

        st.divider()
        total_days = float(
            st.number_input(
                "Days to simulate",
                min_value=0.0,
                value=1.0,
                step=1.0,
                key="ui_total_days",
            )
        )

        _render_regenerate_button(
            generation_values,
            runtime_values,
        )
        _render_continue_button(
            runtime_values,
            total_days,
        )

    _render_main_area()


def _initialize_session_state() -> None:
    defaults = {
        "has_generated": False,
        "world": None,
        "engine": None,
        "metrics_system": None,
        "current_config": None,
        "day_count": 0.0,
        "_notice": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _show_notice() -> None:
    notice = st.session_state.get("_notice")
    if notice is None:
        return

    st.session_state["_notice"] = None
    level, message = notice
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.info(message)


def _render_generation_controls() -> dict[str, Any]:
    disabled = bool(st.session_state["has_generated"])

    st.header("World generation")
    if disabled:
        st.caption("Generation parameters are locked after generation.")

    population_size = int(
        st.number_input(
            "Population size",
            min_value=2,
            max_value=100_000,
            value=DEFAULT_GENERATION["population_size"],
            step=100,
            disabled=disabled,
            key="ui_population_size",
        )
    )

    initial_infection_count = int(
        st.number_input(
            "Initial infected households",
            min_value=0,
            max_value=population_size,
            value=1,
            step=1,
            disabled=disabled,
            key="ui_initial_infection_count",
            help=(
                "Seeds at most one initially infected person "
                "per selected household."
            ),
        )
    )

    tick_duration = float(
        st.selectbox(
            "Tick duration (hours)",
            options=TICK_DURATION_OPTIONS,
            index=TICK_DURATION_OPTIONS.index(
                DEFAULT_GENERATION["tick_duration"]
            ),
            disabled=disabled,
            key="ui_tick_duration",
            help=(
                "Tick duration is fixed when the world is generated. "
                "Changing it later would alter the number of contact "
                "sampling rounds per day."
            ),
        )
    )

    student_ratio = float(
        st.slider(
            "Student ratio",
            0.0,
            1.0,
            DEFAULT_GENERATION["student_ratio"],
            0.01,
            disabled=disabled,
            key="ui_student_ratio",
        )
    )
    employment_rate = float(
        st.slider(
            "Employment rate",
            0.0,
            1.0,
            DEFAULT_GENERATION["employment_rate"],
            0.01,
            disabled=disabled,
            key="ui_employment_rate",
        )
    )
    school_utilization_rate = float(
        st.slider(
            "School utilization rate",
            0.01,
            1.0,
            DEFAULT_GENERATION["school_utilization_rate"],
            0.01,
            disabled=disabled,
            key="ui_school_utilization_rate",
        )
    )

    household_distribution_text = st.text_area(
        "Household-size distribution",
        value=_mapping_to_json(
            DEFAULT_GENERATION["household_size_distribution"]
        ),
        disabled=disabled,
        key="ui_household_size_distribution",
    )
    workplace_distribution_text = st.text_area(
        "Workplace-size distribution",
        value=_mapping_to_json(
            DEFAULT_GENERATION["workplace_size_distribution"]
        ),
        disabled=disabled,
        key="ui_workplace_size_distribution",
    )
    school_distribution_text = st.text_area(
        "School-size distribution",
        value=_mapping_to_json(
            DEFAULT_GENERATION["school_size_distribution"]
        ),
        disabled=disabled,
        key="ui_school_size_distribution",
    )

    public_place_count = int(
        st.number_input(
            "Number of public places",
            min_value=0,
            value=DEFAULT_GENERATION["public_place_count"],
            step=1,
            disabled=disabled,
            key="ui_public_place_count",
        )
    )
    public_place_capacity = int(
        st.number_input(
            "Public-place capacity",
            min_value=1,
            value=DEFAULT_GENERATION["public_place_capacity"],
            step=10,
            disabled=disabled,
            key="ui_public_place_capacity",
        )
    )

    with st.expander("Relationship generation"):
        workmate_target_degree = int(
            st.number_input(
                "Workmate target degree",
                min_value=0,
                value=DEFAULT_GENERATION["workmate_target_degree"],
                disabled=disabled,
                key="ui_workmate_target_degree",
            )
        )
        schoolmate_target_degree = int(
            st.number_input(
                "Schoolmate target degree",
                min_value=0,
                value=DEFAULT_GENERATION["schoolmate_target_degree"],
                disabled=disabled,
                key="ui_schoolmate_target_degree",
            )
        )
        workmate_weight = float(
            st.number_input(
                "Workmate weight",
                min_value=0.0,
                value=DEFAULT_GENERATION["workmate_weight"],
                step=0.1,
                disabled=disabled,
                key="ui_workmate_weight",
            )
        )
        schoolmate_weight = float(
            st.number_input(
                "Schoolmate weight",
                min_value=0.0,
                value=DEFAULT_GENERATION["schoolmate_weight"],
                step=0.1,
                disabled=disabled,
                key="ui_schoolmate_weight",
            )
        )
        min_friend_count = int(
            st.number_input(
                "Minimum friend count",
                min_value=0,
                value=DEFAULT_GENERATION["min_friend_count"],
                disabled=disabled,
                key="ui_min_friend_count",
            )
        )
        max_friend_count = int(
            st.number_input(
                "Maximum friend count",
                min_value=0,
                value=DEFAULT_GENERATION["max_friend_count"],
                disabled=disabled,
                key="ui_max_friend_count",
            )
        )
        friend_weight = float(
            st.number_input(
                "Friend weight",
                min_value=0.0,
                value=DEFAULT_GENERATION["friend_weight"],
                step=0.1,
                disabled=disabled,
                key="ui_friend_weight",
            )
        )

    return {
        "population_size": population_size,
        "tick_duration": tick_duration,
        "initial_infection_count": initial_infection_count,
        "student_ratio": student_ratio,
        "employment_rate": employment_rate,
        "school_utilization_rate": school_utilization_rate,
        "household_size_distribution_text": household_distribution_text,
        "workplace_size_distribution_text": workplace_distribution_text,
        "school_size_distribution_text": school_distribution_text,
        "public_place_count": public_place_count,
        "public_place_capacity": public_place_capacity,
        "workmate_target_degree": workmate_target_degree,
        "schoolmate_target_degree": schoolmate_target_degree,
        "workmate_weight": workmate_weight,
        "schoolmate_weight": schoolmate_weight,
        "min_friend_count": min_friend_count,
        "max_friend_count": max_friend_count,
        "friend_weight": friend_weight,
    }


def _render_runtime_controls() -> dict[str, Any]:
    st.header("Runtime parameters")

    with st.expander("Daily schedule"):
        work_start_hour = _hour_input("Work start hour", "work_start_hour")
        work_end_hour = _hour_input("Work end hour", "work_end_hour")
        school_start_hour = _hour_input("School start hour", "school_start_hour")
        school_end_hour = _hour_input("School end hour", "school_end_hour")
        public_start_hour = _hour_input("Public start hour", "public_start_hour")
        public_end_hour = _hour_input("Public end hour", "public_end_hour")

        public_visit_probability_weekday = float(
            st.slider(
                "Weekday public-visit probability",
                0.0,
                1.0,
                DEFAULT_RUNTIME["public_visit_probability_weekday"],
                0.01,
                key="ui_public_visit_probability_weekday",
            )
        )
        public_visit_probability_weekend = float(
            st.slider(
                "Weekend public-visit probability",
                0.0,
                1.0,
                DEFAULT_RUNTIME["public_visit_probability_weekend"],
                0.01,
                key="ui_public_visit_probability_weekend",
            )
        )

    with st.expander("Contact sampling"):
        home_k = _contact_input("Home contact K", PlaceType.HOME)
        workplace_k = _contact_input("Workplace contact K", PlaceType.WORKPLACE)
        school_k = _contact_input("School contact K", PlaceType.SCHOOL)
        public_k = _contact_input("Public contact K", PlaceType.PUBLIC)

    infection_probability = float(
        st.number_input(
            "Transmission probability per contact",
            min_value=0.0,
            max_value=0.999,
            value=DEFAULT_RUNTIME["infection_probability"],
            step=0.001,
            format="%.4f",
            key="ui_infection_probability",
        )
    )
    
    recovery_rate = float(
        st.number_input(
            "Recovery probability per tick",
            min_value=0.0,
            max_value=0.999,
            value=DEFAULT_RUNTIME["recovery_rate"],
            step=0.001,
            format="%.4f",
            key="ui_recovery_rate",
        )
    )

    deadly_rate = float(
        st.number_input(
            "Recovery probability per tick",
            min_value=0.0,
            max_value=0.999,
            value=DEFAULT_RUNTIME["deadly_rate"],
            step=0.001,
            format="%.4f",
            key="ui_deadly_rate",
        )
    )

    return {
        "work_start_hour": work_start_hour,
        "work_end_hour": work_end_hour,
        "school_start_hour": school_start_hour,
        "school_end_hour": school_end_hour,
        "public_start_hour": public_start_hour,
        "public_end_hour": public_end_hour,
        "public_visit_probability_weekday": public_visit_probability_weekday,
        "public_visit_probability_weekend": public_visit_probability_weekend,
        "contact_k": {
            PlaceType.HOME: home_k,
            PlaceType.WORKPLACE: workplace_k,
            PlaceType.SCHOOL: school_k,
            PlaceType.PUBLIC: public_k,
        },
        "infection_probability": infection_probability,
        "recovery_rate": recovery_rate,
        "deadly_rate": deadly_rate,
    }


def _hour_input(label: str, config_name: str) -> float:
    return float(
        st.number_input(
            label,
            min_value=0.0,
            max_value=23.99,
            value=DEFAULT_RUNTIME[config_name],
            step=0.5,
            key=f"ui_{config_name}",
        )
    )


def _contact_input(label: str, place_type: PlaceType) -> int:
    return int(
        st.number_input(
            label,
            min_value=0,
            value=DEFAULT_RUNTIME["contact_k"][place_type],
            step=1,
            key=f"ui_contact_k_{place_type.name.lower()}",
        )
    )


def _render_regenerate_button(
    generation_values: Mapping[str, Any],
    runtime_values: Mapping[str, Any],
) -> None:
    if not st.button(
        "Regenerate world",
        type="primary",
        use_container_width=True,
    ):
        return

    generation_succeeded = False

    try:
        config = _build_config(generation_values, runtime_values)
        world = controller.generate(config)
        controller.seed_infections(world,count=generation_values["initial_infection_count"],)

        engine, metrics_system = controller.build_engine(world, config)

        st.session_state.update(
            {
                "world": world,
                "engine": engine,
                "metrics_system": metrics_system,
                "has_generated": True,
                "day_count": 0.0,
                "current_config": config,
                "initial_infection_count":generation_values["initial_infection_count"],
                "_notice": ("success","A new world was generated successfully.",),
            }
        )
        generation_succeeded = True
    except Exception as exc:
        st.error(f"World generation failed: {exc}")

    # Keep Streamlit's rerun control-flow signal outside the broad
    # Exception handler. This avoids swallowing it on versions where
    # the internal rerun exception inherits from Exception.
    if generation_succeeded:
        st.rerun()


def _render_continue_button(
    runtime_values: Mapping[str, Any],
    total_days: float,
) -> None:
    if not st.button(
        "Continue simulation",
        disabled=not st.session_state["has_generated"],
        use_container_width=True,
    ):
        return

    try:
        world = st.session_state["world"]
        engine = st.session_state["engine"]
        if world is None or engine is None:
            raise RuntimeError("World or Engine is missing from session_state.")

        changed_overrides = {
            name: value
            for name, value in runtime_values.items()
            if getattr(world.config, name) != value
        }
        if changed_overrides:
            controller.update_runtime_config(
                world,
                **changed_overrides,
            )
            st.session_state["current_config"] = world.config

        controller.run(engine, total_days)
        st.session_state["day_count"] += total_days
        st.success(f"Simulation advanced by {total_days:g} days.")
    except Exception as exc:
        st.error(f"Simulation failed: {exc}")


def _build_config(
    generation_values: Mapping[str, Any],
    runtime_values: Mapping[str, Any],
) -> SimulationConfig:
    generation_kwargs = {
        "population_size": generation_values["population_size"],
        "tick_duration": generation_values["tick_duration"],
        "student_ratio": generation_values["student_ratio"],
        "employment_rate": generation_values["employment_rate"],
        "school_utilization_rate": generation_values["school_utilization_rate"],
        "household_size_distribution": _parse_distribution(
            generation_values["household_size_distribution_text"],
            "Household-size distribution",
        ),
        "workplace_size_distribution": _parse_distribution(
            generation_values["workplace_size_distribution_text"],
            "Workplace-size distribution",
        ),
        "school_size_distribution": _parse_distribution(
            generation_values["school_size_distribution_text"],
            "School-size distribution",
        ),
        "public_place_count": generation_values["public_place_count"],
        "public_place_capacity": generation_values["public_place_capacity"],
        "workmate_target_degree": generation_values["workmate_target_degree"],
        "schoolmate_target_degree": generation_values["schoolmate_target_degree"],
        "workmate_weight": generation_values["workmate_weight"],
        "schoolmate_weight": generation_values["schoolmate_weight"],
        "min_friend_count": generation_values["min_friend_count"],
        "max_friend_count": generation_values["max_friend_count"],
        "friend_weight": generation_values["friend_weight"],
    }
    return SimulationConfig(**generation_kwargs, **dict(runtime_values))


def _render_main_area() -> None:
    if not st.session_state["has_generated"]:
        st.info("Configure the world in the sidebar and generate it to begin.")
        return

    world = st.session_state["world"]
    metrics_system = st.session_state["metrics_system"]
    if world is None or metrics_system is None:
        st.error("The current session is incomplete. Regenerate the world.")
        return

    columns = st.columns(4)
    columns[0].metric("Population", len(world.persons))
    columns[1].metric("Simulated days", f"{st.session_state['day_count']:g}")
    columns[2].metric(
        "Current simulation hour",
        "—" if world.current_time is None else f"{world.current_time:g}",
    )
    columns[3].metric("Snapshots", len(metrics_system.history))

    selected_view = st.radio(
        "View",
        ["Simulation results", "Generation process"],
        horizontal=True,
        key="ui_main_view",
    )

    if selected_view == "Simulation results":
        _render_results(metrics_system)
    else:
        _render_generation_plots(world)


def _render_results(metrics_system: Any) -> None:
    if not metrics_system.history:
        st.info("No simulation ticks have been run yet.")
        return

    sird_tab, occupancy_tab = st.tabs(["SIRD states", "Place occupancy"])
    with sird_tab:
        _display_plot(lambda: plotting.draw_sird_chart(metrics_system.history))
    with occupancy_tab:
        _display_plot(
            lambda: plotting.draw_occupancy_chart(
                metrics_system.occupancy_history
            )
        )


def _render_generation_plots(world: Any) -> None:
    choice = st.radio(
        "Generation visualization",
        ["Generated distributions", "Relationship network"],
        horizontal=True,
        key="ui_generation_plot",
    )

    if choice == "Generated distributions":
        _display_plot(lambda: plotting.draw_generation_histograms(world))
        return

    figure: Figure | None = None
    try:
        figure = plotting.draw_network_graph(world)
        if figure is None:
            st.info(
                "The network is hidden because the population exceeds "
                f"the limit ({plotting.NETWORK_GRAPH_PERSON_LIMIT})."
            )
            return
        st.pyplot(figure, clear_figure=False)
    except Exception as exc:
        st.error(f"Network visualization failed: {exc}")
    finally:
        if figure is not None:
            plt.close(figure)


def _display_plot(draw: Callable[[], Figure]) -> None:
    figure: Figure | None = None
    try:
        figure = draw()
        st.pyplot(figure, clear_figure=False)
    except Exception as exc:
        st.error(f"Chart rendering failed: {exc}")
    finally:
        if figure is not None:
            plt.close(figure)


def _parse_distribution(raw: Any, label: str) -> dict[int, float]:
    if not isinstance(raw, str):
        raise TypeError(f"{label} must be JSON text.")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")

    result: dict[int, float] = {}
    for raw_size, raw_weight in parsed.items():
        try:
            size = int(raw_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} contains an invalid group size: {raw_size!r}"
            ) from exc

        if isinstance(raw_weight, bool):
            raise ValueError(f"{label} contains a boolean weight for size {size}.")

        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} contains an invalid weight for size {size}."
            ) from exc

        result[size] = weight

    return result


def _mapping_to_json(mapping: Mapping[int, float]) -> str:
    return json.dumps(
        {str(key): value for key, value in mapping.items()},
        indent=2,
        sort_keys=True,
    )


