from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import matplotlib

# Select the non-interactive backend before importing pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure
import pytest

from sird_sim.domain.person import Person
from sird_sim.domain.relationship import (
    RelationType,
    Relationship,
)
from sird_sim.plotting import (
    NETWORK_GRAPH_PERSON_LIMIT,
    _affiliation_group_sizes,
    _draw_empty_message,
    _draw_integer_histogram,
    _friend_count,
    _relationship_endpoints,
    _relationship_type_name,
    draw_generation_histograms,
    draw_network_graph,
    draw_occupancy_chart,
    draw_sird_chart,
)
from sird_sim.systems.metrics_system import (
    MetricsSnapshot,
    OccupancySnapshot,
)
from sird_sim.world import World


@dataclass(slots=True)
class PlotPerson:
    """
    Minimal person-shaped object required by plotting.py.

    Plotting only reads these static identity/affiliation fields, so the
    tests do not need to construct the full simulation Person state.
    """

    person_id: int
    home_id: int
    workplace_id: int | None = None
    school_id: int | None = None


def _make_relationship(
    person_a_id: int,
    person_b_id: int,
    relation_type: RelationType,
) -> Relationship:
    return Relationship(
        person_a_id=person_a_id,
        person_b_id=person_b_id,
        relation_type=relation_type,
        weight=1.0,
    )


def _add_relationship(
    world: World,
    relationship: Relationship,
) -> None:
    """
    Store the same relationship in both endpoint adjacency lists.

    This matches the representation consumed by World.get_relationships()
    and also verifies that NetworkX deduplicates the repeated edge.
    """
    world.relationships.setdefault(
        relationship.person_a_id,
        [],
    ).append(relationship)

    world.relationships.setdefault(
        relationship.person_b_id,
        [],
    ).append(relationship)


def _make_world(
    people: list[PlotPerson],
) -> World:
    world = World()

    world.persons = {
        person.person_id: cast(Person, person)
        for person in people
    }

    return world


def _non_friend_relation_type() -> RelationType:
    return next(
        relation_type
        for relation_type in RelationType
        if relation_type is not RelationType.FRIEND
    )


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def test_affiliation_group_sizes_ignores_none_and_counts_groups() -> None:
    result = _affiliation_group_sizes(
        [1, 1, 2, None, 2, 2]
    )

    assert sorted(result) == [2, 3]


def test_relationship_endpoints_returns_both_person_ids() -> None:
    relationship = _make_relationship(
        person_a_id=10,
        person_b_id=20,
        relation_type=RelationType.FRIEND,
    )

    assert _relationship_endpoints(
        relationship
    ) == (10, 20)


def test_relationship_type_name_uses_enum_member_name() -> None:
    relationship = _make_relationship(
        person_a_id=1,
        person_b_id=2,
        relation_type=RelationType.FRIEND,
    )

    assert _relationship_type_name(
        relationship
    ) == "FRIEND"


def test_draw_integer_histogram_handles_empty_values() -> None:
    figure, axis = plt.subplots()

    try:
        _draw_integer_histogram(
            ax=axis,
            values=[],
            title="Empty integer histogram",
            x_label="Value",
        )

        assert axis.get_title() == "Empty integer histogram"
        assert any(
            text.get_text() == "No data available"
            for text in axis.texts
        )
        assert len(axis.patches) == 0
    finally:
        plt.close(figure)


def test_draw_integer_histogram_draws_bars_for_integer_values() -> None:
    figure, axis = plt.subplots()

    try:
        _draw_integer_histogram(
            ax=axis,
            values=[1, 1, 2, 3, 3, 3],
            title="Integer histogram",
            x_label="Value",
        )

        assert axis.get_title() == "Integer histogram"
        assert len(axis.patches) > 0

        heights = [
            patch.get_height()
            for patch in axis.patches
        ]
        assert heights == pytest.approx(
            [2.0, 1.0, 3.0]
        )
    finally:
        plt.close(figure)


def test_draw_empty_message_sets_title_and_text() -> None:
    figure, axis = plt.subplots()

    try:
        _draw_empty_message(
            axis,
            title="Nothing here",
        )

        assert axis.get_title() == "Nothing here"
        assert [
            text.get_text()
            for text in axis.texts
        ] == ["No data available"]
        assert not axis.axison
    finally:
        plt.close(figure)


# ---------------------------------------------------------------------------
# Helpers that depend on a small World and real Relationship objects
# ---------------------------------------------------------------------------


def test_friend_count_counts_only_unique_friend_relationships() -> None:
    world = _make_world(
        [
            PlotPerson(1, home_id=1),
            PlotPerson(2, home_id=1),
            PlotPerson(3, home_id=2),
            PlotPerson(4, home_id=2),
        ]
    )

    friend_1_to_2 = _make_relationship(
        1,
        2,
        RelationType.FRIEND,
    )
    friend_3_to_1 = _make_relationship(
        3,
        1,
        RelationType.FRIEND,
    )
    non_friend_1_to_4 = _make_relationship(
        1,
        4,
        _non_friend_relation_type(),
    )

    _add_relationship(world, friend_1_to_2)
    _add_relationship(world, friend_3_to_1)
    _add_relationship(world, non_friend_1_to_4)

    # Add the same friend relationship a second time to verify that
    # _friend_count() deduplicates by the other person's ID.
    world.relationships[1].append(friend_1_to_2)

    assert _friend_count(
        world,
        person_id=1,
    ) == 2

    # Person 1 is person_b_id here, verifying endpoint orientation.
    assert _friend_count(
        world,
        person_id=3,
    ) == 1

    # The non-FRIEND relationship must not count.
    assert _friend_count(
        world,
        person_id=4,
    ) == 0


# ---------------------------------------------------------------------------
# Top-level drawing functions
# ---------------------------------------------------------------------------


def test_draw_sird_chart_handles_empty_history() -> None:
    figure = draw_sird_chart([])

    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 1

        axis = figure.axes[0]
        assert axis.get_title() == "SIRD population over time"
        assert any(
            text.get_text() == "No data available"
            for text in axis.texts
        )
    finally:
        plt.close(figure)


def test_draw_sird_chart_draws_four_correct_series() -> None:
    history = [
        MetricsSnapshot(
            time=0.0,
            susceptible=98,
            infected=2,
            recovered=0,
            dead=0,
        ),
        MetricsSnapshot(
            time=1.0,
            susceptible=94,
            infected=5,
            recovered=1,
            dead=0,
        ),
        MetricsSnapshot(
            time=2.0,
            susceptible=90,
            infected=7,
            recovered=2,
            dead=1,
        ),
    ]

    figure = draw_sird_chart(history)

    try:
        assert len(figure.axes) == 1

        axis = figure.axes[0]
        assert len(axis.lines) == 4

        assert axis.lines[0].get_label() == "Susceptible"
        assert axis.lines[0].get_xdata().tolist() == [
            0.0,
            1.0,
            2.0,
        ]
        assert axis.lines[0].get_ydata().tolist() == [
            98,
            94,
            90,
        ]

        assert axis.lines[1].get_ydata().tolist() == [
            2,
            5,
            7,
        ]
        assert axis.lines[2].get_ydata().tolist() == [
            0,
            1,
            2,
        ]
        assert axis.lines[3].get_ydata().tolist() == [
            0,
            0,
            1,
        ]
    finally:
        plt.close(figure)


def test_draw_occupancy_chart_handles_empty_history() -> None:
    figure = draw_occupancy_chart([])

    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 1

        axis = figure.axes[0]
        assert axis.get_title() == "Place occupancy over time"
        assert any(
            text.get_text() == "No data available"
            for text in axis.texts
        )
    finally:
        plt.close(figure)


def test_draw_occupancy_chart_creates_four_stackplot_collections() -> None:
    history = [
        OccupancySnapshot(
            time=0.0,
            home=80,
            workplace=10,
            school=8,
            public=2,
        ),
        OccupancySnapshot(
            time=1.0,
            home=30,
            workplace=40,
            school=25,
            public=5,
        ),
        OccupancySnapshot(
            time=2.0,
            home=70,
            workplace=10,
            school=5,
            public=15,
        ),
    ]

    figure = draw_occupancy_chart(history)

    try:
        assert len(figure.axes) == 1
        axis = figure.axes[0]

        assert len(axis.lines) == 0
        assert len(axis.collections) == 4
    finally:
        plt.close(figure)


def test_draw_network_graph_returns_none_above_person_limit() -> None:
    people = [
        PlotPerson(
            person_id=person_id,
            home_id=person_id,
        )
        for person_id in range(
            NETWORK_GRAPH_PERSON_LIMIT + 1
        )
    ]

    world = _make_world(people)

    assert draw_network_graph(world) is None


def test_draw_network_graph_draws_isolated_nodes_without_edges() -> None:
    world = _make_world(
        [
            PlotPerson(1, home_id=1),
            PlotPerson(2, home_id=1),
            PlotPerson(3, home_id=2),
        ]
    )

    figure = draw_network_graph(world)

    assert figure is not None

    try:
        assert len(figure.axes) == 1
        axis = figure.axes[0]

        node_collections = [
            collection
            for collection in axis.collections
            if isinstance(
                collection,
                PathCollection,
            )
        ]
        edge_collections = [
            collection
            for collection in axis.collections
            if isinstance(
                collection,
                LineCollection,
            )
        ]

        assert node_collections
        assert len(
            node_collections[0].get_offsets()
        ) == 3
        assert edge_collections == []
    finally:
        plt.close(figure)


def test_draw_network_graph_draws_relationship_edge() -> None:
    world = _make_world(
        [
            PlotPerson(1, home_id=1),
            PlotPerson(2, home_id=1),
            PlotPerson(3, home_id=2),
        ]
    )

    relationship = _make_relationship(
        1,
        2,
        RelationType.FRIEND,
    )
    _add_relationship(world, relationship)

    figure = draw_network_graph(world)

    assert figure is not None

    try:
        axis = figure.axes[0]

        edge_collections = [
            collection
            for collection in axis.collections
            if isinstance(
                collection,
                LineCollection,
            )
        ]

        # The same Relationship is stored in both adjacency lists, but
        # nx.Graph should still draw only one edge segment.
        assert len(edge_collections) == 1
        assert len(
            edge_collections[0].get_segments()
        ) == 1
    finally:
        plt.close(figure)


def test_draw_generation_histograms_matches_world_distributions() -> None:
    world = _make_world(
        [
            PlotPerson(
                1,
                home_id=1,
                workplace_id=10,
                school_id=None,
            ),
            PlotPerson(
                2,
                home_id=1,
                workplace_id=10,
                school_id=None,
            ),
            PlotPerson(
                3,
                home_id=2,
                workplace_id=11,
                school_id=20,
            ),
            PlotPerson(
                4,
                home_id=2,
                workplace_id=None,
                school_id=20,
            ),
            PlotPerson(
                5,
                home_id=2,
                workplace_id=None,
                school_id=20,
            ),
        ]
    )

    friend_1_to_2 = _make_relationship(
        1,
        2,
        RelationType.FRIEND,
    )
    friend_2_to_3 = _make_relationship(
        2,
        3,
        RelationType.FRIEND,
    )
    non_friend_1_to_3 = _make_relationship(
        1,
        3,
        _non_friend_relation_type(),
    )

    _add_relationship(world, friend_1_to_2)
    _add_relationship(world, friend_2_to_3)
    _add_relationship(world, non_friend_1_to_3)

    figure = draw_generation_histograms(world)

    try:
        assert len(figure.axes) == 4

        (
            household_axis,
            workplace_axis,
            school_axis,
            friend_axis,
        ) = figure.axes

        assert household_axis.get_title() == (
            "Household size distribution"
        )
        assert workplace_axis.get_title() == (
            "Workplace size distribution"
        )
        assert school_axis.get_title() == (
            "School size distribution"
        )
        assert friend_axis.get_title() == (
            "Friend count distribution"
        )

        # Household group sizes are [2, 3].
        assert [
            patch.get_height()
            for patch in household_axis.patches
        ] == pytest.approx([1.0, 1.0])

        # Workplace group sizes are [2, 1].
        assert [
            patch.get_height()
            for patch in workplace_axis.patches
        ] == pytest.approx([1.0, 1.0])

        # The single school has three students.
        assert [
            patch.get_height()
            for patch in school_axis.patches
        ] == pytest.approx([1.0])

        # Friend counts per person are [1, 2, 1, 0, 0].
        # Histogram frequencies at 0, 1 and 2 are [2, 2, 1].
        assert [
            patch.get_height()
            for patch in friend_axis.patches
        ] == pytest.approx([2.0, 2.0, 1.0])
    finally:
        plt.close(figure)
