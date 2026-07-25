from collections.abc import Iterable, Mapping, Collection
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

@dataclass(frozen=True, slots=True)
class _Partition:
    """
    Standardized distribution data used internally.
    """

    sizes: npt.NDArray[np.int64]
    weights: npt.NDArray[np.float64]

def partition_by_size_distribution(
    indices: Iterable[int],
    size_distribution: Mapping[int, float],
    rng: np.random.Generator,
) -> list[set[int]]:
    """
    According to the group size distribution, completely divide a set of unique integer indices into mutually non-overlapping groups.

    This function is solely responsible for general segmentation and is unaware of domain concepts such as Person, Place, HOME, SCHOOL, or WORKPLACE.

    Args:
        indices:
            The index of the member that needs to be split.

        size_distribution:
            Mapping of group size to sampling weights. For example:

            {
                1: 0.25,
                2: 0.40,
                3: 0.25,
                4: 0.10,
            }

    Returns:
        Member grouping. Each member appears only once.

    Raises:
        ValueError:
            The input is invalid, or the total number of members cannot be precisely composed by the given group size.
    """
    member_indices = _validate_and_materialize_indices(indices)

    if not member_indices:
        return []

    distribution = _prepare_distribution(size_distribution)

    reachable = _build_reachable_table(
        total_members=len(member_indices),
        group_sizes=distribution.sizes,
    )

    if not reachable[len(member_indices)]:
        raise ValueError(
            f"{len(member_indices)} members cannot be partitioned "
            f"exactly using group sizes "
            f"{distribution.sizes.tolist()}"
        )

    shuffled_indices = np.asarray(
        member_indices,
        dtype=np.int64,
    ).copy()

    rng.shuffle(shuffled_indices)

    return _partition_indices(
        shuffled_indices=shuffled_indices,
        distribution=distribution,
        reachable=reachable,
        rng=rng,
    )

def _validate_and_materialize_indices(
    indices: Iterable[int],
) -> list[int]:
    """
    Convert any iterable to a list while verifying the member indices.
    """
    member_indices = list(indices)

    for index in member_indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(
                "All member indices must be integers"
            )

    if len(member_indices) != len(set(member_indices)):
        raise ValueError(
            "Member indices must be unique"
        )

    return member_indices

def _prepare_distribution(
    size_distribution: Mapping[int, float],
) -> _Partition:
    """
    Verify the size distribution and convert it into a sorted NumPy array.
    """
    if not size_distribution:
        raise ValueError(
            "size_distribution cannot be empty"
        )

    valid_items: list[tuple[int, float]] = []

    for group_size, weight in size_distribution.items():
        if (
            isinstance(group_size, bool)
            or not isinstance(group_size, int)
            or group_size <= 0
        ):
            raise ValueError(
                "Every group size must be a positive integer"
            )

        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Weight for group size {group_size} "
                "must be numeric"
            ) from exc

        if not np.isfinite(numeric_weight):
            raise ValueError(
                f"Weight for group size {group_size} "
                "must be finite"
            )

        if numeric_weight < 0:
            raise ValueError(
                f"Weight for group size {group_size} "
                "cannot be negative"
            )

        if numeric_weight > 0:
            valid_items.append(
                (group_size, numeric_weight)
            )

    if not valid_items:
        raise ValueError(
            "At least one group size must have positive weight"
        )

    valid_items.sort(key=lambda item: item[0])

    sizes = np.asarray(
        [item[0] for item in valid_items],
        dtype=np.int64,
    )

    weights = np.asarray(
        [item[1] for item in valid_items],
        dtype=np.float64,
    )

    return _Partition(
        sizes=sizes,
        weights=weights,
    )

def _build_reachable_table(
    total_members: int,
    group_sizes: npt.NDArray[np.int64],
) -> npt.NDArray[np.bool_]:
    """
    reachable [n] indicates whether n members can be precisely formed by the allowed group size.
    """
    reachable = np.zeros(
        total_members + 1,
        dtype=np.bool_,
    )

    reachable[0] = True

    for member_count in range(1, total_members + 1):
        usable_sizes = group_sizes[
            group_sizes <= member_count
        ]

        if usable_sizes.size == 0:
            continue

        previous_counts = member_count - usable_sizes

        reachable[member_count] = bool(
            np.any(reachable[previous_counts])
        )

    return reachable

def _partition_indices(
    shuffled_indices: npt.NDArray[np.int64],
    distribution: _Partition,
    reachable: npt.NDArray[np.bool_],
    rng: np.random.Generator,
) -> list[set[int]]:
    """
    Complete the actual segmentation using the prepared distribution and accessibility tables.
    """
    groups: list[set[int]] = []

    current_position = 0
    remaining_members = len(shuffled_indices)

    while remaining_members > 0:
        eligible_positions = _find_eligible_positions(
            remaining_members=remaining_members,
            group_sizes=distribution.sizes,
            reachable=reachable,
        )

        selected_size = _sample_group_size(
            eligible_positions=eligible_positions,
            distribution=distribution,
            rng=rng,
        )

        group_end = current_position + selected_size

        selected_members = shuffled_indices[
            current_position:group_end
        ]

        groups.append(
            {
                int(member_index)
                for member_index in selected_members
            }
        )

        current_position = group_end
        remaining_members -= selected_size

    return groups

def largest_partitionable_count(
    total_count: int,
    allowed_sizes: Collection[int],
) -> int:
    """
    Find the largest value no greater than total_count that can be
    represented as a sum of allowed group sizes.
    """
    if total_count < 0:
        raise ValueError(
            "total_count cannot be negative"
        )

    sizes = sorted(set(allowed_sizes))

    if not sizes:
        raise ValueError(
            "allowed_sizes cannot be empty"
        )

    if any(
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        for size in sizes
    ):
        raise ValueError(
            "allowed_sizes must contain positive integers"
        )

    reachable = _build_reachable_table(
        total_members=total_count,
        group_sizes=np.asarray(sizes, dtype=np.int64),
    )

    for count in range(total_count, -1, -1):
        if reachable[count]:
            return count

    raise RuntimeError("No partitionable count found")

def _find_eligible_positions(
    remaining_members: int,
    group_sizes: npt.NDArray[np.int64],
    reachable: npt.NDArray[np.bool_],
) -> npt.NDArray[np.int64]:
    """
    Identify the current position of the selectable group size. 
    The size of a group must simultaneously satisfy the following conditions:
    1. It should not exceed the remaining number of people;
    2. After selection, the remaining number of people should still be able to be divided legally.
    """
    fits_remaining = group_sizes <= remaining_members

    remaining_after_selection = (
        remaining_members - group_sizes
    )

    leaves_reachable_remainder = np.zeros(
        group_sizes.shape,
        dtype=np.bool_,
    )

    valid_remainder_positions = (
        remaining_after_selection >= 0
    )

    leaves_reachable_remainder[
        valid_remainder_positions
    ] = reachable[
        remaining_after_selection[
            valid_remainder_positions
        ]
    ]

    eligible_mask = (
        fits_remaining
        & leaves_reachable_remainder
    )

    eligible_positions = np.flatnonzero(
        eligible_mask
    ).astype(np.int64)

    if eligible_positions.size == 0:
        raise RuntimeError(
            "No eligible group size was found despite "
            "the remaining member count being reachable"
        )

    return eligible_positions

def _sample_group_size(
    eligible_positions: npt.NDArray[np.int64],
    distribution: _Partition,
    rng: np.random.Generator,
) -> int:
    """
    Select a size from the current legal group size based on the weight.
    """
    eligible_sizes = distribution.sizes[
        eligible_positions
    ]

    eligible_weights = distribution.weights[
        eligible_positions
    ]

    probabilities = (
        eligible_weights / eligible_weights.sum()
    )

    return int(
        rng.choice(
            eligible_sizes,
            p=probabilities,
        )
    )