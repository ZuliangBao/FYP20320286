import numpy as np
import pytest
# Adjust this import if your module is stored elsewhere.
from sird_sim.partition import (
    partition_by_size_distribution,
    largest_partitionable_count,
    _build_reachable_table,
    _find_eligible_positions,
)

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260726)


# ============================================================
# partition_by_size_distribution: normal behavior
# ============================================================

def test_partition_covers_every_member_exactly_once(
    rng: np.random.Generator,
) -> None:
    members = list(range(24))
    allowed_sizes = {2, 3, 4, 6}

    groups = partition_by_size_distribution(
        indices=members,
        size_distribution={
            2: 0.20,
            3: 0.30,
            4: 0.30,
            6: 0.20,
        },
        rng=rng,
    )

    assert all(len(group) in allowed_sizes for group in groups)
    assert sum(len(group) for group in groups) == len(members)

    flattened = [
        member
        for group in groups
        for member in group
    ]

    assert set(flattened) == set(members)
    assert len(flattened) == len(set(flattened))


def test_partition_groups_are_pairwise_disjoint(
    rng: np.random.Generator,
) -> None:
    groups = partition_by_size_distribution(
        indices=range(18),
        size_distribution={2: 1.0, 3: 1.0},
        rng=rng,
    )

    for index, first_group in enumerate(groups):
        for second_group in groups[index + 1:]:
            assert first_group.isdisjoint(second_group)


def test_partition_is_reproducible_with_same_seed() -> None:
    distribution = {2: 0.4, 3: 0.4, 4: 0.2}

    first = partition_by_size_distribution(
        indices=range(20),
        size_distribution=distribution,
        rng=np.random.default_rng(12345),
    )

    second = partition_by_size_distribution(
        indices=range(20),
        size_distribution=distribution,
        rng=np.random.default_rng(12345),
    )

    assert first == second


def test_partition_accepts_generator_input(
    rng: np.random.Generator,
) -> None:
    indices = (index for index in range(8))

    groups = partition_by_size_distribution(
        indices=indices,
        size_distribution={2: 1.0},
        rng=rng,
    )

    assert len(groups) == 4
    assert all(len(group) == 2 for group in groups)


def test_partition_empty_indices_returns_empty_list(
    rng: np.random.Generator,
) -> None:
    result = partition_by_size_distribution(
        indices=[],
        size_distribution={1: 1.0},
        rng=rng,
    )

    assert result == []


def test_zero_weight_group_sizes_are_ignored(
    rng: np.random.Generator,
) -> None:
    groups = partition_by_size_distribution(
        indices=range(8),
        size_distribution={
            1: 0.0,
            2: 1.0,
            4: 0.0,
        },
        rng=rng,
    )

    assert all(len(group) == 2 for group in groups)


def test_partition_raises_when_exact_partition_is_impossible(
    rng: np.random.Generator,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"5 members cannot be partitioned exactly",
    ):
        partition_by_size_distribution(
            indices=range(5),
            size_distribution={2: 1.0, 4: 1.0},
            rng=rng,
        )


# ============================================================
# indices validation
# ============================================================

@pytest.mark.parametrize(
    "indices",
    [
        [0, True],
        [0, 1.5],
        [0, "1"],
        [0, None],
    ],
)
def test_partition_rejects_non_integer_indices(
    indices: list[object],
    rng: np.random.Generator,
) -> None:
    with pytest.raises(
        ValueError,
        match="All member indices must be integers",
    ):
        partition_by_size_distribution(
            indices=indices,
            size_distribution={1: 1.0},
            rng=rng,
        )


def test_partition_rejects_duplicate_indices(
    rng: np.random.Generator,
) -> None:
    with pytest.raises(
        ValueError,
        match="Member indices must be unique",
    ):
        partition_by_size_distribution(
            indices=[1, 2, 2, 3],
            size_distribution={1: 1.0},
            rng=rng,
        )


# ============================================================
# distribution validation
# ============================================================

def test_partition_rejects_empty_distribution(
    rng: np.random.Generator,
) -> None:
    with pytest.raises(
        ValueError,
        match="size_distribution cannot be empty",
    ):
        partition_by_size_distribution(
            indices=[0],
            size_distribution={},
            rng=rng,
        )


@pytest.mark.parametrize(
    "distribution",
    [
        {0: 1.0},
        {-1: 1.0},
        {1.5: 1.0},
        {True: 1.0},
        {"2": 1.0},
    ],
)
def test_partition_rejects_invalid_group_sizes(
    distribution: dict[object, float],
    rng: np.random.Generator,
) -> None:
    with pytest.raises(
        ValueError,
        match="Every group size must be a positive integer",
    ):
        partition_by_size_distribution(
            indices=[0, 1],
            size_distribution=distribution,
            rng=rng,
        )


@pytest.mark.parametrize(
    "distribution",
    [
        {1: "not-a-number"},
        {1: None},
    ],
)
def test_partition_rejects_non_numeric_weights(
    distribution: dict[int, object],
    rng: np.random.Generator,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"Weight for group size 1 must be numeric",
    ):
        partition_by_size_distribution(
            indices=[0],
            size_distribution=distribution,
            rng=rng,
        )


@pytest.mark.parametrize(
    "weight",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_partition_rejects_non_finite_weights(
    weight: float,
    rng: np.random.Generator,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"Weight for group size 1 must be finite",
    ):
        partition_by_size_distribution(
            indices=[0],
            size_distribution={1: weight},
            rng=rng,
        )


def test_partition_rejects_negative_weight(
    rng: np.random.Generator,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"Weight for group size 1 cannot be negative",
    ):
        partition_by_size_distribution(
            indices=[0],
            size_distribution={1: -0.1},
            rng=rng,
        )


def test_partition_rejects_all_zero_weights(
    rng: np.random.Generator,
) -> None:
    with pytest.raises(
        ValueError,
        match="At least one group size must have positive weight",
    ):
        partition_by_size_distribution(
            indices=[0, 1],
            size_distribution={1: 0.0, 2: 0.0},
            rng=rng,
        )


def test_partition_rejects_boolean_weight(
    rng: np.random.Generator,
) -> None:
    """
    Recommended strict boundary behavior.

    bool is a subclass of int in Python, so the implementation must
    reject it explicitly before float(weight).
    """
    with pytest.raises(
        ValueError,
        match=r"Weight for group size 1 must be numeric",
    ):
        partition_by_size_distribution(
            indices=[0],
            size_distribution={1: True},
            rng=rng,
        )


# ============================================================
# _build_reachable_table
# ============================================================

def test_build_reachable_table_known_case() -> None:
    reachable = _build_reachable_table(
        total_members=10,
        group_sizes=np.asarray(
            [3, 5],
            dtype=np.int64,
        ),
    )

    expected_reachable = {
        0,
        3,
        5,
        6,
        8,
        9,
        10,
    }

    assert reachable.dtype == np.bool_
    assert len(reachable) == 11

    for count in range(11):
        assert bool(reachable[count]) == (
            count in expected_reachable
        )


def test_build_reachable_table_zero_members() -> None:
    reachable = _build_reachable_table(
        total_members=0,
        group_sizes=np.asarray(
            [2, 3],
            dtype=np.int64,
        ),
    )

    assert reachable.tolist() == [True]


# ============================================================
# largest_partitionable_count
# ============================================================

@pytest.mark.parametrize(
    ("total_count", "allowed_sizes", "expected"),
    [
        (10, [2, 3], 10),
        (7, [4, 6], 6),
        (3, [4, 6], 0),
        (0, [2, 3], 0),
        (11, [5], 10),
    ],
)
def test_largest_partitionable_count(
    total_count: int,
    allowed_sizes: list[int],
    expected: int,
) -> None:
    assert largest_partitionable_count(
        total_count=total_count,
        allowed_sizes=allowed_sizes,
    ) == expected


def test_largest_partitionable_count_rejects_negative_total() -> None:
    with pytest.raises(
        ValueError,
        match="total_count cannot be negative",
    ):
        largest_partitionable_count(
            total_count=-1,
            allowed_sizes=[1, 2],
        )


@pytest.mark.parametrize(
    "total_count",
    [
        True,
        1.5,
        "10",
    ],
)
def test_largest_partitionable_count_rejects_non_integer_total(
    total_count: object,
) -> None:
    """
    Recommended strict boundary behavior.
    """
    with pytest.raises(
        ValueError,
        match="total_count must be a non-negative integer",
    ):
        largest_partitionable_count(
            total_count=total_count,  # type: ignore[arg-type]
            allowed_sizes=[1, 2],
        )


def test_largest_partitionable_count_rejects_empty_sizes() -> None:
    with pytest.raises(
        ValueError,
        match="allowed_sizes cannot be empty",
    ):
        largest_partitionable_count(
            total_count=10,
            allowed_sizes=[],
        )


@pytest.mark.parametrize(
    "allowed_sizes",
    [
        [0, 2],
        [-1, 2],
        [True, 2],
        [1.5, 2],
        ["2", 3],
        [1, "2"],
    ],
)
def test_largest_partitionable_count_rejects_invalid_sizes(
    allowed_sizes: list[object],
) -> None:
    """
    Validation should happen before sorted(set(...)) so mixed types
    raise the documented ValueError instead of an incidental TypeError.
    """
    with pytest.raises(
        ValueError,
        match="allowed_sizes must contain positive integers",
    ):
        largest_partitionable_count(
            total_count=10,
            allowed_sizes=allowed_sizes,  # type: ignore[arg-type]
        )


def test_largest_partitionable_count_ignores_duplicate_sizes() -> None:
    assert largest_partitionable_count(
        total_count=11,
        allowed_sizes=[3, 3, 5, 5],
    ) == 11


# ============================================================
# _find_eligible_positions
# ============================================================

def test_find_eligible_positions_only_keeps_safe_choices() -> None:
    group_sizes = np.asarray(
        [3, 4, 5],
        dtype=np.int64,
    )

    reachable = _build_reachable_table(
        total_members=8,
        group_sizes=group_sizes,
    )

    eligible_positions = _find_eligible_positions(
        remaining_members=8,
        group_sizes=group_sizes,
        reachable=reachable,
    )

    eligible_sizes = group_sizes[
        eligible_positions
    ].tolist()

    # 3 leaves 5, and 5 leaves 3. Both remainders are reachable.
    # 4 leaves 4, which is also reachable here because size 4 is allowed.
    assert eligible_sizes == [3, 4, 5]


def test_find_eligible_positions_rejects_inconsistent_state() -> None:
    group_sizes = np.asarray(
        [3, 5],
        dtype=np.int64,
    )

    # Deliberately inconsistent table: no remainder is reachable.
    reachable = np.zeros(
        9,
        dtype=np.bool_,
    )

    with pytest.raises(
        RuntimeError,
        match="No eligible group size was found",
    ):
        _find_eligible_positions(
            remaining_members=8,
            group_sizes=group_sizes,
            reachable=reachable,
        )
