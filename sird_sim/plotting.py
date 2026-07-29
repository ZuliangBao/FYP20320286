from __future__ import annotations
from collections import Counter
from collections.abc import Sequence
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import networkx as nx  
import numpy as np
from .domain.place import PlaceType
from .domain.relationship import Relationship
from .systems.metrics_system import (
    MetricsSnapshot,
    OccupancySnapshot,
)
from .world import World

NETWORK_GRAPH_PERSON_LIMIT = 300

def draw_sird_chart(
    history: Sequence[MetricsSnapshot],
) -> Figure:
    """
    Draw SIRD population counts over simulation time.

    The function only creates and returns a Matplotlib Figure.
    It does not call plt.show().
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    if not history:
        _draw_empty_message(
            ax,
            title="SIRD population over time",
        )
        return fig

    times = [
        snapshot.time
        for snapshot in history
    ]

    susceptible = [
        snapshot.susceptible
        for snapshot in history
    ]

    infected = [
        snapshot.infected
        for snapshot in history
    ]

    recovered = [
        snapshot.recovered
        for snapshot in history
    ]

    dead = [
        snapshot.dead
        for snapshot in history
    ]

    ax.plot(
        times,
        susceptible,
        label="Susceptible",
    )
    ax.plot(
        times,
        infected,
        label="Infected",
    )
    ax.plot(
        times,
        recovered,
        label="Recovered",
    )
    ax.plot(
        times,
        dead,
        label="Dead",
    )

    ax.set_title("SIRD population over time")
    ax.set_xlabel("Simulation time")
    ax.set_ylabel("Population")
    ax.legend()
    ax.grid(visible=True,alpha=0.3)

    fig.tight_layout()
    return fig

def draw_occupancy_chart(
    occupancy_history: Sequence[OccupancySnapshot],
) -> Figure:
    """
    Draw population occupancy by place type as a stacked area chart.
    """
    fig, ax = plt.subplots(
        figsize=(10, 5),
    )

    if not occupancy_history:
        _draw_empty_message(
            ax,
            title="Place occupancy over time",
        )
        return fig

    times = [
        snapshot.time
        for snapshot in occupancy_history
    ]

    home = [
        snapshot.home
        for snapshot in occupancy_history
    ]

    workplace = [
        snapshot.workplace
        for snapshot in occupancy_history
    ]

    school = [
        snapshot.school
        for snapshot in occupancy_history
    ]

    public = [
        snapshot.public
        for snapshot in occupancy_history
    ]

    ax.stackplot(
        times,
        home,
        workplace,
        school,
        public,
        labels=[
            "Home",
            "Workplace",
            "School",
            "Public",
        ],
    )

    ax.set_title("Place occupancy over time")
    ax.set_xlabel("Simulation time")
    ax.set_ylabel("Current occupants")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig

def draw_network_graph(
    world: World,
) -> Figure | None:
    """
    Draw the static relationship network.

    Return None when the population exceeds the configured
    visualization threshold. The caller decides whether to hide the
    option or display an explanatory message.

    Node labels are only drawn when person_count <= 50, since larger
    networks become unreadable with labels. Layout positions use a
    fixed seed (independent of world.rng) so the same graph renders
    with a stable layout across repeated calls.
    """
    person_count = len(world.persons)

    if person_count > NETWORK_GRAPH_PERSON_LIMIT:
        return None

    graph: nx.Graph[int] = nx.Graph()

    # Add people first so isolated people are still displayed.
    graph.add_nodes_from(world.persons.keys())

    # The same Relationship object may appear in both endpoint
    # adjacency lists. nx.Graph automatically deduplicates the edge.
    for relationships in world.relationships.values():
        for relationship in relationships:
            person_a_id, person_b_id = (
                _relationship_endpoints(relationship)
            )

            graph.add_edge(person_a_id,person_b_id)

    fig, ax = plt.subplots(figsize=(9, 7))

    if graph.number_of_nodes() == 0:
        _draw_empty_message(
            ax,
            title="Relationship network",
        )
        return fig

    positions = nx.spring_layout(graph,seed=42)

    # Large node labels quickly become unreadable.
    show_labels = person_count <= 50

    nx.draw(
        graph,
        pos=positions,
        ax=ax,
        with_labels=show_labels,
        node_size=120 if show_labels else 45,
        font_size=7,
        width=0.7,
    )

    ax.set_title("Relationship network")
    ax.set_axis_off()
    fig.tight_layout()
    return fig

def draw_generation_histograms(
    world: World,
) -> Figure:
    """
    Draw distributions produced during world generation.

    These distributions use permanent affiliations stored on Person,
    rather than current Place.occupants, because occupants change as the
    simulation runs.

    Charts:

        1. Household size
        2. Workplace size
        3. School size
        4. Friend count per person
    """
    household_sizes = _affiliation_group_sizes(
        affiliation_ids=[
            person.home_id
            for person in world.persons.values()
        ]
    )

    workplace_sizes = _affiliation_group_sizes(
        affiliation_ids=[
            person.workplace_id
            for person in world.persons.values()
        ]
    )

    school_sizes = _affiliation_group_sizes(
        affiliation_ids=[
            person.school_id
            for person in world.persons.values()
        ]
    )

    friend_counts = [
        _friend_count(
            world=world,
            person_id=person_id,
        )
        for person_id in sorted(world.persons)
    ]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(11, 8),
    )

    _draw_integer_histogram(
        ax=axes[0, 0],
        values=household_sizes,
        title="Household size distribution",
        x_label="Household size",
    )

    _draw_integer_histogram(
        ax=axes[0, 1],
        values=workplace_sizes,
        title="Workplace size distribution",
        x_label="Workers per workplace",
    )

    _draw_integer_histogram(
        ax=axes[1, 0],
        values=school_sizes,
        title="School size distribution",
        x_label="Students per school",
    )

    _draw_integer_histogram(
        ax=axes[1, 1],
        values=friend_counts,
        title="Friend count distribution",
        x_label="Friends per person",
    )

    fig.suptitle("Generated world distributions")
    fig.tight_layout()

    return fig

def _affiliation_group_sizes(
    affiliation_ids: Sequence[int | None],
) -> list[int]:
    """
    Convert permanent person-to-place affiliations into group sizes.

    None is ignored, for example for unemployed workers.
    """
    counts = Counter(
        affiliation_id
        for affiliation_id in affiliation_ids
        if affiliation_id is not None
    )

    return list(counts.values())

def _friend_count(
    world: World,
    person_id: int,
) -> int:
    """
    Count unique FRIEND relationships for one person.
    """
    friend_ids: set[int] = set()

    for relationship in world.get_relationships(
        person_id
    ):
        relationship_type = (
            _relationship_type_name(
                relationship
            )
        )

        if relationship_type != "FRIEND":
            continue

        person_a_id, person_b_id = (
            _relationship_endpoints(
                relationship
            )
        )

        other_person_id = (
            person_b_id
            if person_a_id == person_id
            else person_a_id
        )

        friend_ids.add(other_person_id)

    return len(friend_ids)

def _relationship_endpoints(
    relationship: Relationship,
) -> tuple[int, int]:
    """
    Return the two endpoint IDs from a Relationship.

    This implementation assumes the current Relationship fields are:

        person_a_id
        person_b_id
    """
    try:
        person_a_id = relationship.person_a_id
        person_b_id = relationship.person_b_id
    except AttributeError as exc:
        raise AttributeError(
            "Relationship must define person_a_id "
            "and person_b_id"
        ) from exc

    return int(person_a_id), int(person_b_id)

def _relationship_type_name(
    relationship: Relationship,
) -> str:
    """
    Return the enum name of a relationship type.

    This implementation assumes the Relationship field is named:

        relation_type
    """
    try:
        relationship_type = (relationship.relation_type)
    except AttributeError as exc:
        raise AttributeError(
            "Relationship must define relation_type"
        ) from exc

    name = getattr(relationship_type,"name",None)

    if name is None:
        return str(relationship_type).upper()

    return str(name).upper()

def _draw_integer_histogram(
    *,
    ax,
    values: Sequence[int],
    title: str,
    x_label: str,
) -> None:
    """
    Draw a histogram whose bins are centered on integer values.
    """
    if not values:
        _draw_empty_message(
            ax,
            title=title,
        )
        return

    minimum = min(values)
    maximum = max(values)

    bins = np.arange(
        minimum - 0.5,
        maximum + 1.5,
        1.0,
    )

    ax.hist(
        values,
        bins=bins,
    )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.grid(
        visible=True,
        axis="y",
        alpha=0.3,
    )

def _draw_empty_message(
    ax,
    *,
    title: str,
) -> None:
    """
    Configure an empty axis when no data is available.
    """
    ax.set_title(title)

    ax.text(
        0.5,
        0.5,
        "No data available",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )

    ax.set_axis_off()
    ax.figure.tight_layout()