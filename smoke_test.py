import numpy as np
from sird_sim.config import SimulationConfig
from sird_sim.generation import generate_world
from sird_sim.domain.place import PlaceType

config = SimulationConfig(
    # Schedule
    work_start_hour=9.0,
    work_end_hour=17.0,

    school_start_hour=8.0,
    school_end_hour=15.0,

    public_start_hour=17.0,
    public_end_hour=22.0,

    public_visit_probability_weekday=0.20,
    public_visit_probability_weekend=0.50,

    # Population
    population_size=50,
    student_ratio=0.3,

    # school and company occupants rates
    employment_rate=0.85,
    school_utilization_rate=0.75,

    # Place-size distributions
    household_size_distribution={
        2: 0.2,
        3: 0.3,
        4: 0.3,
        5: 0.2,
    },

    workplace_size_distribution={
        10: 0.5,
        20: 0.5,
    },

    school_size_distribution={
        10: 0.5,
        20: 0.5,
    },

    # Group relationships
    workmate_target_degree=5,
    schoolmate_target_degree=5,
    workmate_weight=0.5,
    schoolmate_weight=0.5,

    # Public places
    public_place_count=3,
    public_place_capacity=20,

    # Friend relationships
    min_friend_count=3,
    max_friend_count=10,
    friend_weight=0.5,

    contact_k = {
        PlaceType.HOME: 4,
        PlaceType.WORKPLACE: 6,
        PlaceType.SCHOOL: 8,
        PlaceType.PUBLIC: 3,
    },

    tick_duration=1.0,
    infection_probability=0.08,
    recovery_rate=0.05,
    deadly_rate=0.005,
)

rng = np.random.default_rng(seed=42)
world = generate_world(config, rng)

print("人数:", len(world.persons))
print("场所数:", len(world.places))
print("有关系记录的人数:", len(world.relationships))