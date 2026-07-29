from __future__ import annotations

from ..domain.person import HealthState, Person, Role
from ..domain.place import PlaceType
from ..world import World


class ScheduleSystem:
    """
    Determine where each living person should be based on their role,
    the current simulation hour, and the simulated weekday, then perform
    the required movement between places.
    """

    def step(self, world: World) -> None:
        """
        Update every eligible person's physical location for the current
        simulation tick.
        """
        for person in world.persons.values():
            if self._should_skip_person(person):
                continue

            target_place_id = self._determine_target_place(
                person=person,
                world=world,
            )

            self._move_person(
                person=person,
                target_place_id=target_place_id,
                world=world,
            )

    def _should_skip_person(
        self,
        person: Person,
    ) -> bool:
        """
        Return True when the person must not participate in movement.

        Dead people are permanently excluded from scheduling because
        HealthEventSystem removes them from their current place when
        death is processed.
        """
        return person.health_state == HealthState.DEAD

    def _determine_target_place(
        self,
        person: Person,
        world: World,
    ) -> int:
        """
        Determine the person's target location using this priority order:

            1. A worker goes to their workplace during work hours.
            2. A student goes to their school during school hours.
            3. During public hours, the person may visit an available
               public place according to the weekday or weekend
               visit probability.
            4. Otherwise, the person returns home.

        Work and school scheduling are disabled on weekends.
        """
        if (
            person.role == Role.WORKER
            and person.workplace_id is not None
            and self._is_work_time(world)
        ):
            return person.workplace_id

        if (
            person.role == Role.STUDENT
            and person.school_id is not None
            and self._is_school_time(world)
        ):
            return person.school_id

        if self._is_public_time(world):
            config = world.require_config()

            probability = (
                config.public_visit_probability_weekend
                if self._is_weekend(world)
                else config.public_visit_probability_weekday
            )

            if world.rng.random() < probability:
                public_place_id = self._choose_public_place(
                    world
                )

                if public_place_id is not None:
                    return public_place_id

        return person.home_id

    def _is_work_time(
        self,
        world: World,
    ) -> bool:
        if self._is_weekend(world):
            return False

        config = world.require_config()
        hour = self._current_hour(world)

        return (
            config.work_start_hour
            <= hour
            <= config.work_end_hour
        )

    def _is_school_time(
        self,
        world: World,
    ) -> bool:
        if self._is_weekend(world):
            return False

        config = world.require_config()
        hour = self._current_hour(world)

        return (
            config.school_start_hour
            <= hour
            <= config.school_end_hour
        )

    def _is_public_time(
        self,
        world: World,
    ) -> bool:
        config = world.require_config()
        hour = self._current_hour(world)

        return (
            config.public_start_hour
            <= hour
            <= config.public_end_hour
        )

    def _current_hour(
        self,
        world: World,
    ) -> float:
        """
        Return the hour within the current simulated day.

        current_time represents the number of hours elapsed since the
        simulation began.

        Examples:
            current_time = 9.5  -> day 0, 09:30
            current_time = 33.0 -> day 1, 09:00
        """
        current_time = world.require_current_time()

        return current_time % 24.0

    def _is_weekend(
        self,
        world: World,
    ) -> bool:
        """
        Return whether the current simulated day is a weekend.

        The simulation uses Monday as day index 0:

            0 = Monday
            1 = Tuesday
            2 = Wednesday
            3 = Thursday
            4 = Friday
            5 = Saturday
            6 = Sunday

        Therefore weekday indices 5 and 6 are treated as weekends.
        The seven-day cycle then repeats.
        """
        current_time = world.require_current_time()

        day_index = int(
            current_time // 24.0
        )

        weekday_index = day_index % 7

        return weekday_index in {5,6}

    def _choose_public_place(
        self,
        world: World,
    ) -> int | None:
        """
        Uniformly choose one PUBLIC place that has remaining capacity.

        Places that have reached their capacity are excluded. Return None
        when no public place is currently available.
        """
        available_place_ids: list[int] = []

        for place in world.places.values():
            if place.place_type != PlaceType.PUBLIC:
                continue

            if (
                place.capacity is not None
                and len(place.occupants)
                >= place.capacity
            ):
                continue

            available_place_ids.append(place.place_id)

        if not available_place_ids:
            return None

        return int(world.rng.choice(available_place_ids))

    def _move_person(
        self,
        person: Person,
        target_place_id: int,
        world: World,
    ) -> None:
        """
        Move a person to the target place and update both places'
        occupancy records.

        No movement occurs when:
            - the person is already at the target place; or
            - the target place has reached its capacity.

        Capacity failure is intentionally silent: the person remains at
        their current place and the caller is not notified.

        Raises:
            KeyError: If target_place_id does not exist in world.places.
        """
        if target_place_id == person.current_place_id:
            return

        if target_place_id not in world.places:
            raise KeyError(
                f"Target place {target_place_id} does not exist"
            )

        target_place = world.get_place(target_place_id)

        if (
            target_place.capacity is not None
            and len(target_place.occupants)
            >= target_place.capacity
        ):
            return

        world.move_person(
            person=person,
            place_id=target_place_id,
        )