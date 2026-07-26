from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any
from .domain.place import PlaceType


@dataclass(frozen=True, kw_only=True)
class SimulationConfig:
    # ============================================================
    # Schedule configuration
    # ============================================================

    # Work schedule
    work_start_hour: float
    work_end_hour: float

    # School schedule
    school_start_hour: float
    school_end_hour: float

    # Public activity schedule
    public_start_hour: float
    public_end_hour: float

    # Probability of visiting public places
    public_visit_probability_weekday: float
    public_visit_probability_weekend: float



    # ============================================================
    # Population, place, and relationship generation
    # ============================================================

    # Population
    population_size: int
    student_ratio: float

    # Group-size distributions
    employment_rate: float
    school_utilization_rate: float
    household_size_distribution: Mapping[int, float]
    workplace_size_distribution: Mapping[int, float]
    school_size_distribution: Mapping[int, float]

    # Workplace and school relationship networks
    workmate_target_degree: float
    schoolmate_target_degree: float

    workmate_weight: float
    schoolmate_weight: float

    # Public places
    public_place_count: int
    public_place_capacity: int

    # Friendship network
    min_friend_count: int
    max_friend_count: int
    friend_weight: float

    # ============================================================
    # Contact system config
    # ============================================================
    contact_k: Mapping[PlaceType, int]

    # ============================================================
    # Disease transmission and progression
    # ============================================================
    tick_duration: float
    infection_probability: float
    recovery_rate: float
    deadly_rate: float

    def __post_init__(self) -> None:
        # ========================================================
        # Schedule validation
        # ========================================================

        self._validate_hour(
            self.work_start_hour,
            "work_start_hour",
        )
        self._validate_hour(
            self.work_end_hour,
            "work_end_hour",
        )
        self._validate_hour(
            self.school_start_hour,
            "school_start_hour",
        )
        self._validate_hour(
            self.school_end_hour,
            "school_end_hour",
        )
        self._validate_hour(
            self.public_start_hour,
            "public_start_hour",
        )
        self._validate_hour(
            self.public_end_hour,
            "public_end_hour",
        )

        if self.work_start_hour >= self.work_end_hour:
            raise ValueError(
                "work_start_hour must be earlier than work_end_hour"
            )

        if self.school_start_hour >= self.school_end_hour:
            raise ValueError(
                "school_start_hour must be earlier than school_end_hour"
            )

        if self.public_start_hour >= self.public_end_hour:
            raise ValueError(
                "public_start_hour must be earlier than "
                "public_end_hour"
            )

        self._validate_probability(
            self.public_visit_probability_weekday,
            "public_visit_probability_weekday",
        )
        self._validate_probability(
            self.public_visit_probability_weekend,
            "public_visit_probability_weekend",
        )

        # ========================================================
        # Population and place validation
        # ========================================================

        self._validate_positive_int(
            self.population_size,
            "population_size",
        )

        self._validate_probability(
            self.student_ratio,
            "student_ratio",
        )

        self._validate_probability(
            self.employment_rate,
            "employment_rate",
        )

        if self.school_utilization_rate == 0.0:
            raise ValueError(
                "school_utilization_rate must be greater than 0"
            )

        self._validate_probability(
            self.school_utilization_rate,
            "school_utilization_rate",
        )

        self._validate_size_distribution(
            self.household_size_distribution,
            "household_size_distribution",
        )
        self._validate_size_distribution(
            self.workplace_size_distribution,
            "workplace_size_distribution",
        )
        self._validate_size_distribution(
            self.school_size_distribution,
            "school_size_distribution",
        )

        self._validate_nonnegative_number(
            self.workmate_target_degree,
            "workmate_target_degree",
        )
        self._validate_nonnegative_number(
            self.schoolmate_target_degree,
            "schoolmate_target_degree",
        )

        self._validate_nonnegative_number(
            self.workmate_weight,
            "workmate_weight",
        )
        self._validate_nonnegative_number(
            self.schoolmate_weight,
            "schoolmate_weight",
        )

        self._validate_nonnegative_int(
            self.public_place_count,
            "public_place_count",
        )
        self._validate_positive_int(
            self.public_place_capacity,
            "public_place_capacity",
        )

        self._validate_nonnegative_int(
            self.min_friend_count,
            "min_friend_count",
        )
        self._validate_nonnegative_int(
            self.max_friend_count,
            "max_friend_count",
        )

        if self.min_friend_count > self.max_friend_count:
            raise ValueError(
                "min_friend_count cannot be greater than "
                "max_friend_count"
            )

        self._validate_nonnegative_number(
            self.friend_weight,
            "friend_weight",
        )

        # ========================================================
        # Contact validation
        # ========================================================
        self._validate_contact_k(
            self.contact_k,
            "contact_k",
        )


        self._validate_positive_number(
            self.tick_duration,
            "tick_duration",
        )

        self._validate_probability(
            self.infection_probability,
            "infection_probability",
        )

        self._validate_probability_below_one(
            self.recovery_rate,
            "recovery_rate",
        )

        self._validate_probability_below_one(
            self.deadly_rate,
            "deadly_rate",
        )

    # ============================================================
    # Validation helpers
    # ============================================================
    @staticmethod
    def _validate_contact_k(
        value: Mapping[PlaceType, int],
        name: str = "contact_k",
    ) -> None:
        if not isinstance(value, Mapping):
            raise TypeError(
                f"{name} must be a mapping from PlaceType to int"
            )

        expected_place_types = {
            PlaceType.HOME,
            PlaceType.WORKPLACE,
            PlaceType.SCHOOL,
            PlaceType.PUBLIC,
        }

        for place_type, contact_count in value.items():
            if not isinstance(place_type, PlaceType):
                raise TypeError(
                    f"All keys in {name} must be PlaceType values; "
                    f"received {place_type!r}"
                )

            if (
                isinstance(contact_count, bool)
                or not isinstance(contact_count, int)
            ):
                raise TypeError(
                    f"All values in {name} must be integers; "
                    f"received {contact_count!r} for {place_type.name}"
                )

            if contact_count < 0:
                raise ValueError(
                    f"All values in {name} must be non-negative; "
                    f"received {contact_count} for {place_type.name}"
                )

        actual_place_types = set(value)

        missing_place_types = (
            expected_place_types - actual_place_types
        )

        if missing_place_types:
            missing_names = sorted(
                place_type.name
                for place_type in missing_place_types
            )

            raise ValueError(
                f"{name} is missing required PlaceType entries: "
                f"{missing_names}"
            )

    @staticmethod
    def _validate_hour(
        value: float,
        name: str,
    ) -> None:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name} must be a number"
            )

        if not math.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

        if not 0.0 <= value < 24.0:
            raise ValueError(
                f"{name} must be in [0, 24)"
            )

    @staticmethod
    def _validate_probability(
        value: float,
        name: str,
    ) -> None:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name} must be a number"
            )

        if not math.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be in [0.0, 1.0]"
            )

    @staticmethod
    def _validate_positive_int(
        value: int,
        name: str,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer"
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be greater than 0"
            )

    @staticmethod
    def _validate_nonnegative_int(
        value: int,
        name: str,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer"
            )

        if value < 0:
            raise ValueError(
                f"{name} cannot be negative"
            )

    @staticmethod
    def _validate_nonnegative_number(
        value: float,
        name: str,
    ) -> None:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name} must be a number"
            )

        if not math.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

        if value < 0:
            raise ValueError(
                f"{name} cannot be negative"
            )

    @staticmethod
    def _validate_size_distribution(
        distribution: Mapping[int, float],
        name: str,
    ) -> None:
        if not isinstance(distribution, Mapping):
            raise TypeError(
                f"{name} must be a mapping of group size to weight"
            )

        if not distribution:
            raise ValueError(
                f"{name} cannot be empty"
            )

        for group_size, weight in distribution.items():
            if (
                isinstance(group_size, bool)
                or not isinstance(group_size, int)
            ):
                raise TypeError(
                    f"All group sizes in {name} must be integers"
                )

            if group_size <= 0:
                raise ValueError(
                    f"All group sizes in {name} must be greater "
                    f"than 0; received {group_size}"
                )

            if isinstance(weight, bool) or not isinstance(
                weight,
                (int, float),
            ):
                raise TypeError(
                    f"All weights in {name} must be numbers"
                )

            if not math.isfinite(weight):
                raise ValueError(
                    f"All weights in {name} must be finite"
                )

            if weight <= 0:
                raise ValueError(
                    f"All weights in {name} must be greater than 0; "
                    f"received weight {weight} for size {group_size}"
                )

    @staticmethod
    def _validate_probability_below_one(
        value: float,
        name: str,
    ) -> None:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(f"{name} must be a number")

        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0.0, 1.0)")


    @staticmethod
    def _validate_positive_number(
        value: float,
        name: str,
    ) -> None:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(f"{name} must be a number")

        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

        if value <= 0.0:
            raise ValueError(f"{name} must be greater than 0")


    # ============================================================
    # Serialization
    # ============================================================

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(
        self,
        *,
        indent: int | None = 2,
    ) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> SimulationConfig:
        parsed_data = dict(data)

        # JSON object keys are always strings, so convert the size
        # distribution keys back to integers.
        distribution_fields = (
            "household_size_distribution",
            "workplace_size_distribution",
            "school_size_distribution",
        )

        for field_name in distribution_fields:
            if field_name not in parsed_data:
                continue

            distribution = parsed_data[field_name]

            if not isinstance(distribution, Mapping):
                raise TypeError(
                    f"{field_name} must be a mapping"
                )

            try:
                parsed_data[field_name] = {
                    int(group_size): weight
                    for group_size, weight in distribution.items()
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{field_name} contains a group size that "
                    f"cannot be converted to int"
                ) from exc

        return cls(**parsed_data)

    @classmethod
    def from_json(
        cls,
        json_text: str,
    ) -> SimulationConfig:
        data = json.loads(json_text)

        if not isinstance(data, dict):
            raise ValueError(
                "SimulationConfig JSON must contain an object"
            )

        return cls.from_dict(data)

    