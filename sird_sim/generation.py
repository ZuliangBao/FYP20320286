from collections.abc import Iterable, Collection, Mapping
from collections import defaultdict
from itertools import combinations
from .partition import partition_by_size_distribution, largest_partitionable_count
from .domain.place import Place, PlaceType
from .domain.person import Person, Role
from .domain.relationship import Relationship, RelationType
from .config import SimulationConfig
from .world import World
import numpy as np
import math

def _assign_roles(
    population_size: int,
    student_ratio: float,
    rng: np.random.Generator,
) -> dict[int, Role]:
    """
    Independently assign every person a STUDENT or WORKER role.
    Each person becomes a STUDENT with probability student_ratio;
    otherwise, the person becomes a WORKER.
    The realized student proportion is stochastic and does not have
    to exactly equal student_ratio.
    Args:
        population_size:
            Total number of persons. Person IDs are generated as
            integers from 0 to population_size - 1.
        student_ratio:
            Probability that an individual is assigned Role.STUDENT.
            Must be between 0.0 and 1.0.
        rng:
            NumPy random number generator.
    Returns:
        A mapping from person_id to Role.
    """
    if (
        isinstance(population_size, bool)
        or not isinstance(population_size, int)
    ):
        raise TypeError(
            "population_size must be an integer"
        )
    if population_size < 0:
        raise ValueError(
            "population_size cannot be negative"
        )
    if not np.isfinite(student_ratio):
        raise ValueError(
            "student_ratio must be finite"
        )
    if not 0.0 <= student_ratio <= 1.0:
        raise ValueError(
            "student_ratio must be between 0.0 and 1.0"
        )
    student_mask = (
        rng.random(population_size) < student_ratio
    )
    return {
        person_id: (
            Role.STUDENT
            if student_mask[person_id]
            else Role.WORKER
        )
        for person_id in range(population_size)
    }

def _generate_places_from_groups(
    groups: Iterable[set[int]],
    place_type: PlaceType,
    start_place_id: int,
    populate_occupants: bool = True,
) -> tuple[dict[int, Place], dict[int, int]]:
    """
    Package the members into a specified type of Place.
    Returns:
        places:
            place_id -> Place
        member_place_map:
            member index/person_id -> place_id
    """
    if start_place_id < 0:
        raise ValueError("start_place_id cannot be negative")
    places: dict[int, Place] = {}
    member_place_map: dict[int, int] = {}
    for offset, members in enumerate(groups):
        if not members:
            raise ValueError("A place group cannot be empty")
        place_id = start_place_id + offset
        place = Place(
            place_id=place_id,
            place_type=place_type,
            capacity=len(members),
            occupants=set(members) if populate_occupants else set(),
        )
        places[place_id] = place
        for member_index in members:
            if member_index in member_place_map:
                raise ValueError(
                    f"Member {member_index} appears in "
                    "more than one group"
                )
            member_place_map[member_index] = place_id
    return places, member_place_map

def _generate_homes(
        population_size: int, 
        household_size_distribution: Mapping[int, float], 
        rng: np.random.Generator,
        start_place_id:int
        ) -> tuple[dict[int, Place], dict[int, int],int]:
    if population_size < 0:
        raise ValueError("population_size cannot be negative")
    household_groups = partition_by_size_distribution(
        indices=range(population_size),
        size_distribution=household_size_distribution,
        rng=rng,
    )
    homes, person_home_map = _generate_places_from_groups(
        groups=household_groups,
        place_type=PlaceType.HOME,
        start_place_id=start_place_id,
    )    
    next_place_id = start_place_id + len(homes)
    return homes, person_home_map, next_place_id

def _generate_workplaces(
    roles: Mapping[int, Role],
    workplace_size_distribution: Mapping[int, float],
    employment_rate: float,
    rng: np.random.Generator,
    start_place_id: int,
) -> tuple[
    dict[int, Place],
    dict[int, int],
    int,
]:
    """
    Generate workplaces after applying an explicit employment rate.

    Workers excluded by employment_rate are unemployed.

    If the target employed population cannot be exactly partitioned
    using the configured workplace sizes, the smallest unavoidable
    remainder is also left unemployed.

    Returns:
        workplaces:
            workplace_id -> Place

        person_workplace_map:
            employed worker person_id -> workplace_id

        next_place_id:
            Next unused global place_id
    """
    if start_place_id < 0:
        raise ValueError(
            "start_place_id cannot be negative"
        )

    if (
        not np.isfinite(employment_rate)
        or not 0.0 <= employment_rate <= 1.0
    ):
        raise ValueError(
            "employment_rate must be in [0.0, 1.0]"
        )

    worker_ids = [
        person_id
        for person_id, role in roles.items()
        if role == Role.WORKER
    ]

    if not worker_ids or employment_rate == 0.0:
        return {}, {}, start_place_id

    target_employed_count = math.floor(
        len(worker_ids) * employment_rate + 0.5
    )

    target_employed_count = min(
        target_employed_count,
        len(worker_ids),
    )

    if target_employed_count == 0:
        return {}, {}, start_place_id

    worker_ids_array = np.asarray(
        worker_ids,
        dtype=int,
    )
    rng.shuffle(worker_ids_array)

    employment_candidates = worker_ids_array[
        :target_employed_count
    ].tolist()

    assignable_count = largest_partitionable_count(
        total_count=target_employed_count,
        allowed_sizes=workplace_size_distribution.keys(),
    )

    if assignable_count == 0:
        return {}, {}, start_place_id

    employed_worker_ids = employment_candidates[
        :assignable_count
    ]

    workplace_groups = partition_by_size_distribution(
        indices=employed_worker_ids,
        size_distribution=workplace_size_distribution,
        rng=rng,
    )

    workplaces, person_workplace_map = (
        _generate_places_from_groups(
            groups=workplace_groups,
            place_type=PlaceType.WORKPLACE,
            start_place_id=start_place_id,
            populate_occupants=False,
        )
    )

    next_place_id = start_place_id + len(workplaces)

    return (
        workplaces,
        person_workplace_map,
        next_place_id,
    )

def _sample_capacities_until_target(
    target_total_capacity: int,
    size_distribution: Mapping[int, float],
    rng: np.random.Generator,
) -> list[int]:
    """
    Sample place capacities until their sum reaches or exceeds the
    requested total capacity.
    """
    if target_total_capacity < 0:
        raise ValueError(
            "target_total_capacity cannot be negative"
        )

    if target_total_capacity == 0:
        return []

    sizes = np.asarray(
        list(size_distribution.keys()),
        dtype=int,
    )

    weights = np.asarray(
        list(size_distribution.values()),
        dtype=float,
    )

    probabilities = weights / weights.sum()

    capacities: list[int] = []
    total_capacity = 0

    while total_capacity < target_total_capacity:
        capacity = int(
            rng.choice(
                sizes,
                p=probabilities,
            )
        )

        capacities.append(capacity)
        total_capacity += capacity

    return capacities

def _generate_schools(
    roles: Mapping[int, Role],
    school_size_distribution: Mapping[int, float],
    school_utilization_rate: float,
    rng: np.random.Generator,
    start_place_id: int,
) -> tuple[
    dict[int, Place],
    dict[int, int],
    int,
]:
    """
    Generate enough school capacity to achieve approximately the
    configured utilization rate, then randomly assign all students
    without requiring schools to be full.

    Place.occupants remains empty because it represents current
    physical presence, not permanent school membership.
    """
    if start_place_id < 0:
        raise ValueError(
            "start_place_id cannot be negative"
        )

    if (
        not np.isfinite(school_utilization_rate)
        or not 0.0 < school_utilization_rate <= 1.0
    ):
        raise ValueError(
            "school_utilization_rate must be in (0.0, 1.0]"
        )

    student_ids = [
        person_id
        for person_id, role in roles.items()
        if role == Role.STUDENT
    ]

    if not student_ids:
        return {}, {}, start_place_id

    target_total_capacity = math.ceil(
        len(student_ids) / school_utilization_rate
    )

    school_capacities = _sample_capacities_until_target(
        target_total_capacity=target_total_capacity,
        size_distribution=school_size_distribution,
        rng=rng,
    )

    schools: dict[int, Place] = {}

    for offset, capacity in enumerate(school_capacities):
        school_id = start_place_id + offset

        schools[school_id] = Place(
            place_id=school_id,
            place_type=PlaceType.SCHOOL,
            capacity=capacity,
            occupants=set(),
        )

    student_ids_array = np.asarray(
        student_ids,
        dtype=int,
    )
    rng.shuffle(student_ids_array)

    # Temporary record: How much capacity is still available at each school
    remaining_capacity: dict[int, int] = {
        school_id: school.capacity
        for school_id, school in schools.items()
        if school.capacity is not None
    }

    person_school_map: dict[int, int] = {}

    for person_id_value in student_ids_array:
        person_id = int(person_id_value)

        available_school_ids = [
            school_id
            for school_id, capacity
            in remaining_capacity.items()
            if capacity > 0
        ]

        if not available_school_ids:
            raise RuntimeError(
                "Generated school capacity is insufficient "
                "for all students"
            )

        # Random selection is made based on the weighted number of remaining seats.
        # Schools with larger capacities are more likely to be assigned students,
        # but there will be no situation where the first school is filled up first.
        available_capacities = np.asarray(
            [
                remaining_capacity[school_id]
                for school_id in available_school_ids
            ],
            dtype=float,
        )

        probabilities = (
            available_capacities
            / available_capacities.sum()
        )

        selected_school_id = int(
            rng.choice(
                available_school_ids,
                p=probabilities,
            )
        )

        person_school_map[person_id] = selected_school_id
        remaining_capacity[selected_school_id] -= 1

    next_place_id = start_place_id + len(schools)

    return (
        schools,
        person_school_map,
        next_place_id,
    )

def _generate_public_places(public_place_count:int, public_place_capacity:int,start_place_id: int,) -> tuple[dict[int, Place], int]:
    """
        Returns:
        public_places:
            place_id -> PUBLIC Place
        next_place_id:
            next unused place_id
    """
    if public_place_count < 0:
        raise ValueError(
            "public_place_count cannot be negative"
        )
    if public_place_capacity <= 0:
        raise ValueError(
            "public_place_capacity must be positive"
        )
    if start_place_id < 0:
        raise ValueError(
            "start_place_id cannot be negative"
        )
    public_places: dict[int, Place] = {}
    for offset in range(public_place_count):
        place_id = start_place_id + offset
        public_places[place_id] = Place(
            place_id=place_id,
            place_type=PlaceType.PUBLIC,
            capacity=public_place_capacity,
            occupants=set()
        )
    next_place_id = start_place_id + public_place_count
    return public_places, next_place_id            

def _generate_persons(roles, home_map, workplace_map, school_map) -> dict[int, Person]:
    persons = {}
    for person_id, role in roles.items():
        home_id = home_map[person_id]
        workplace_id = workplace_map.get(person_id)
        school_id = school_map.get(person_id)
        person = Person(
            person_id=person_id,
            role=role,
            home_id=home_id,
            workplace_id=workplace_id,
            school_id=school_id,
            current_place_id=home_id,
        )
        persons[person_id] = person
    return persons

def _generate_family_relationships(homes: dict[int, Place],) -> list[Relationship]:
        relationships = []

        for home in homes.values():
            members = sorted(home.occupants)

            for person_a_id, person_b_id in combinations(members,2):
                relationship = Relationship(
                    person_a_id = person_a_id,
                    person_b_id = person_b_id,
                    relation_type = RelationType.FAMILY,
                    weight=1,
                )

                relationships.append(relationship)

        return relationships

def _generate_group_relationships(
    members_by_place: Mapping[int, Collection[int]],
    relation_type: RelationType,
    target_degree: float,
    weight: float,
    rng: np.random.Generator,
) -> list[Relationship]:
    """
    Generate a sparse peer relationship network randomly within a member group. 
    Each candidate edge is generated independently, so as to ensure the expected average degree of the members
    close target_degree。
    """
    if not np.isfinite(target_degree) or target_degree < 0:
        raise ValueError(
            "target_degree must be finite and non-negative"
        )
    if not np.isfinite(weight) or weight < 0:
        raise ValueError(
            "weight must be finite and non-negative"
        )
    
    relationships: list[Relationship] = []
    # Zero or one person cannot establish a relationship between individuals.
    for member_ids in members_by_place.values():
        members = sorted(member_ids)
        member_count = len(members)
        if member_count < 2:
            continue
        connection_probability = min(
            1.0,
            target_degree / (member_count - 1),
        )
        for person_a_id, person_b_id in combinations(members, 2):
            if rng.random() >= connection_probability:
                continue
            pair = (person_a_id, person_b_id)
            relationships.append(
                Relationship(
                    person_a_id=person_a_id,
                    person_b_id=person_b_id,
                    relation_type=relation_type,
                    weight=weight,
                )
            )
    return relationships

def _generate_friend_relationships(
    persons: Mapping[int, Person],
    family_relationships: Iterable[Relationship],
    rng: np.random.Generator,
    min_friends: int = 3,
    max_friends: int = 10,
    weight: float = 0.5,
) -> list[Relationship]:
    """
    Generate a sparse friendship network using greedy pairing.
    Each person receives a randomly sampled target friend count between
    min_friends and max_friends. Existing FAMILY pairs are excluded.
    The algorithm attempts to satisfy everyone's target degree, but exact
    satisfaction is not guaranteed when the remaining constraints make
    further pairing impossible.
    """
    if min_friends < 0:
        raise ValueError("min_friends cannot be negative")
    if max_friends < min_friends:
        raise ValueError(
            "max_friends cannot be smaller than min_friends"
        )
    if not np.isfinite(weight) or weight < 0:
        raise ValueError(
            "weight must be finite and non-negative"
        )
    person_ids = sorted(persons)
    if len(person_ids) < 2:
        return []
    person_id_set = set(person_ids)
    # FAMILY pair 
    family_pairs: set[tuple[int, int]] = set()
    # What are the family members of each person? This is used to calculate the maximum number of selectable friends.
    family_neighbors: dict[int, set[int]] = {
        person_id: set()
        for person_id in person_ids
    }
    for relationship in family_relationships:
        if relationship.relation_type != RelationType.FAMILY:
            continue
        person_a_id = relationship.person_a_id
        person_b_id = relationship.person_b_id
        if (
            person_a_id not in person_id_set
            or person_b_id not in person_id_set
        ):
            raise ValueError(
                "Family relationship references an unknown person"
            )
        pair = (
            min(person_a_id, person_b_id),
            max(person_a_id, person_b_id),
        )
        family_pairs.add(pair)
        family_neighbors[person_a_id].add(person_b_id)
        family_neighbors[person_b_id].add(person_a_id)
    # Set the target number of friends for each person
    target_degrees: dict[int, int] = {}
    for person_id in person_ids:
        sampled_target = int(
            rng.integers(
                low=min_friends,
                high=max_friends + 1,
            )
        )
        # After excluding oneself and one's family members, the maximum number of friends this person can have
        maximum_possible = (
            len(person_ids)
            - 1
            - len(family_neighbors[person_id])
        )
        target_degrees[person_id] = min(
            sampled_target,
            maximum_possible,
        )
    current_degrees = {
        person_id: 0
        for person_id in person_ids
    }
    friend_pairs: set[tuple[int, int]] = set()
    relationships: list[Relationship] = []
    while True:
        active_people = [
            person_id
            for person_id in person_ids
            if current_degrees[person_id]
            < target_degrees[person_id]
        ]
        if len(active_people) < 2:
            break
        # Random numbers are used for random sorting under the same residual demand.
        tie_breakers = {
            person_id: rng.random()
            for person_id in active_people
        }
        # Give priority to addressing the individuals who have the greatest shortage of friends.
        active_people.sort(
            key=lambda person_id: (
                target_degrees[person_id]
                - current_degrees[person_id],
                tie_breakers[person_id],
            ),
            reverse=True,
        )
        made_progress = False
        for person_a_id in active_people:
            if (
                current_degrees[person_a_id]
                >= target_degrees[person_a_id]
            ):
                continue
            candidates: list[int] = []
            for person_b_id in active_people:
                if person_b_id == person_a_id:
                    continue
                if (
                    current_degrees[person_b_id]
                    >= target_degrees[person_b_id]
                ):
                    continue
                pair = (
                    min(person_a_id, person_b_id),
                    max(person_a_id, person_b_id),
                )
                if pair in family_pairs:
                    continue
                if pair in friend_pairs:
                    continue
                candidates.append(person_b_id)
            if not candidates:
                continue
            # Give priority to addressing the individuals who have the greatest shortage of friends.
            largest_remaining_need = max(
                target_degrees[candidate_id]
                - current_degrees[candidate_id]
                for candidate_id in candidates
            )
            best_candidates = [
                candidate_id
                for candidate_id in candidates
                if (
                    target_degrees[candidate_id]
                    - current_degrees[candidate_id]
                    == largest_remaining_need
                )
            ]
            person_b_id = int(
                rng.choice(best_candidates)
            )
            pair = (
                min(person_a_id, person_b_id),
                max(person_a_id, person_b_id),
            )
            relationships.append(
                Relationship(
                    person_a_id=pair[0],
                    person_b_id=pair[1],
                    relation_type=RelationType.FRIEND,
                    weight=weight,
                )
            )
            friend_pairs.add(pair)
            current_degrees[person_a_id] += 1
            current_degrees[person_b_id] += 1
            made_progress = True
        # Stop when there are no legal candidates, to prevent an infinite loop.
        if not made_progress:
            break
    return relationships

def _group_members_by_place(
    person_place_map: Mapping[int, int],
    valid_place_ids: Iterable[int],
) -> dict[int, set[int]]:
    """
    Convert:

        person_id -> place_id

    into:

        place_id -> set[person_id]

    All valid places are included, even when no person is assigned.
    """
    members_by_place: dict[int, set[int]] = {
        place_id: set()
        for place_id in valid_place_ids
    }

    for person_id, place_id in person_place_map.items():
        if place_id not in members_by_place:
            raise ValueError(
                f"Person {person_id} references unknown place "
                f"{place_id}"
            )

        members_by_place[place_id].add(person_id)

    return members_by_place

def generate_world(config: SimulationConfig, rng: np.random.Generator) -> World:
    population_size = config.population_size
    roles = _assign_roles(population_size, config.student_ratio, rng)

    next_place_id = 0
    homes, person_home_map, next_place_id = _generate_homes(population_size=population_size,
                                                            household_size_distribution=config.household_size_distribution,
                                                            rng=rng,
                                                            start_place_id=next_place_id,)
    workplaces, person_workplace_map, next_place_id = _generate_workplaces(roles=roles,
                                                                           workplace_size_distribution=config.workplace_size_distribution,
                                                                           employment_rate=config.employment_rate,
                                                                           rng=rng,
                                                                           start_place_id=next_place_id,)
    schools, person_school_map, next_place_id = _generate_schools(roles=roles,
                                                                  school_size_distribution = config.school_size_distribution,
                                                                  school_utilization_rate=config.school_utilization_rate,
                                                                  rng=rng,
                                                                  start_place_id=next_place_id,)
    public_places, next_place_id = _generate_public_places(
        public_place_count=config.public_place_count,
        public_place_capacity=config.public_place_capacity,
        start_place_id=next_place_id,
    )


    all_places = {**homes, **workplaces, **schools, **public_places}

    persons = _generate_persons(
        roles=roles,
        home_map=person_home_map,
        workplace_map=person_workplace_map,
        school_map=person_school_map,
    )
    family_relationships = _generate_family_relationships(homes)
    workplace_members = _group_members_by_place(
        person_place_map=person_workplace_map,
        valid_place_ids=workplaces.keys(),
    )

    school_members = _group_members_by_place(
        person_place_map=person_school_map,
        valid_place_ids=schools.keys(),
    )

    workmate_relationships = _generate_group_relationships(
        members_by_place=workplace_members,
        relation_type=RelationType.WORKMATE,
        target_degree=config.workmate_target_degree,
        weight=config.workmate_weight,
        rng=rng,
    )
    schoolmate_relationships = _generate_group_relationships(
        members_by_place=school_members,
        relation_type=RelationType.SCHOOLMATE,
        target_degree=config.schoolmate_target_degree,
        weight=config.schoolmate_weight,
        rng=rng,
    )
    friend_relationships = _generate_friend_relationships(
        persons=persons,
        family_relationships=family_relationships,
        rng=rng,
        min_friends=config.min_friend_count,
        max_friends=config.max_friend_count,
        weight=config.friend_weight,
    )

    all_relationships = (
        family_relationships + workmate_relationships
        + schoolmate_relationships + friend_relationships
    )

    relationships_by_person = defaultdict(list)
    for relationship in all_relationships:
        relationships_by_person[relationship.person_a_id].append(relationship)
        relationships_by_person[relationship.person_b_id].append(relationship)

    return World(
        persons=persons,
        places=all_places,
        relationships=dict(relationships_by_person),
        rng=rng,
        config=config,
        current_time=0.0,
    )