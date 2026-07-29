import numpy as np
from ..world import World
class ContactSystem:
    """
    Generate undirected contact pairs among people currently occupying
    the same place.
    """

    def step(self, world: World) -> None:
        """
        Clear contacts from the previous step and generate contacts for
        the current step.
        """
        if world.config is None:
            raise RuntimeError("World.config must be set before ContactSystem.step()")
        
        world.pending_contacts.clear()

        for place in world.places.values():
            member_count = len(place.occupants)

            if member_count < 2:
                continue

            contact_k = world.config.contact_k[place.place_type]

            effective_k = min(
                contact_k,
                member_count - 1,
            )

            if effective_k == 0:
                continue

            # O(member_count), executed once per place. Floyd's algorithm.
            # No repeated full-array filtering inside the person loop.
            members = tuple(place.occupants)

            for person_index, person_a_id in enumerate(members):
                selected_indices = _sample_indices_excluding(
                    population_size=member_count,
                    excluded_index=person_index,
                    sample_size=effective_k,
                    rng=world.rng,
                )

                for selected_index in selected_indices:
                    person_b_id = members[selected_index]

                    pair = (
                        (person_a_id, person_b_id)
                        if person_a_id < person_b_id
                        else (person_b_id, person_a_id)
                    )

                    world.pending_contacts.add(pair)

def _sample_unique_indices(
    population_size: int,
    sample_size: int,
    rng: np.random.Generator,
) -> set[int]:
    """
    Uniformly sample sample_size distinct integers from:

        range(population_size)

    using Floyd's algorithm.

    Expected time complexity: O(sample_size)
    Space complexity: O(sample_size)
    """
    if sample_size < 0:
        raise ValueError("sample_size cannot be negative")

    if sample_size > population_size:
        raise ValueError("sample_size cannot exceed population_size")

    selected: set[int] = set()

    for upper_bound in range(
        population_size - sample_size,
        population_size,
    ):
        candidate = int(
            rng.integers(
                low=0,
                high=upper_bound + 1,
            )
        )

        if candidate in selected:
            selected.add(upper_bound)
        else:
            selected.add(candidate)

    return selected

def _sample_indices_excluding(
    population_size: int,
    excluded_index: int,
    sample_size: int,
    rng: np.random.Generator,
) -> list[int]:
    """
    Sample distinct indices from range(population_size), skipping
    excluded_index, without building or filtering an excluded list.

    Draws sample_size indices from a virtual space of
    population_size - 1 slots (i.e. everyone except excluded_index),
    then shifts any virtual index >= excluded_index up by one to map
    it back into the real index space.
    """
    if not 0 <= excluded_index < population_size:
        raise ValueError("excluded_index is outside the population")

    available_count = population_size - 1

    if sample_size > available_count:
        raise ValueError("sample_size exceeds the number of available indices")

    virtual_indices = _sample_unique_indices(
        population_size=available_count,
        sample_size=sample_size,
        rng=rng,
    )

    # Map indices from the virtual n-1 space back to the real n space.
    return [
        virtual_index
        if virtual_index < excluded_index
        else virtual_index + 1
        for virtual_index in virtual_indices
    ]