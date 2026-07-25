import numpy as np
import pytest
from sird_sim.partition import partition_by_size_distribution


def test_partition_assigns_every_member_exactly_once():
    rng = np.random.default_rng(42)

    groups = partition_by_size_distribution(
        indices=range(1000),
        size_distribution={
            1: 0.28,
            2: 0.34,
            3: 0.18,
            4: 0.14,
            5: 0.06,
        },
        rng=rng,
    )

    flattened = [
        member
        for group in groups
        for member in group
    ]

    assert len(flattened) == 1000
    assert len(set(flattened)) == 1000
    assert set(flattened) == set(range(1000))

def test_partition_only_uses_allowed_group_sizes():
    rng = np.random.default_rng(42)

    allowed_sizes = {1, 2, 3, 4}

    groups = partition_by_size_distribution(
        indices=range(100),
        size_distribution={
            1: 0.2,
            2: 0.4,
            3: 0.3,
            4: 0.1,
        },
        rng=rng,
    )

    assert all(
        len(group) in allowed_sizes
        for group in groups
    )

def test_partition_rejects_impossible_total():
    rng = np.random.default_rng(42)

    with pytest.raises(
        ValueError,
        match="cannot be partitioned exactly",
    ):
        partition_by_size_distribution(
            indices=range(5),
            size_distribution={
                2: 0.5,
                4: 0.5,
            },
            rng=rng,
        )

def test_partition_is_reproducible():
    groups_a = partition_by_size_distribution(
        indices=range(100),
        size_distribution={1: 0.3, 2: 0.4, 3: 0.3},
        rng=np.random.default_rng(42),
    )

    groups_b = partition_by_size_distribution(
        indices=range(100),
        size_distribution={1: 0.3, 2: 0.4, 3: 0.3},
        rng=np.random.default_rng(42),
    )

    assert groups_a == groups_b