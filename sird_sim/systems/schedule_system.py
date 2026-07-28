from ..world import World
from ..domain.person import Person, HealthState, Role
from ..domain.place import PlaceType
from typing import Optional

class ScheduleSystem:

    def step(self, world: World) -> None:
        """
        Update the locations where all personnel should be at the current time.
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

    def _should_skip_person(self, person: Person) -> bool:
        """
        The deceased and those involved in handling other incidents remain in place.
        """
        if person.health_state == HealthState.DEAD:
            return True

        # if person.pending_event is not None:
        #     return True

        return False

    def _determine_target_place(
        self,
        person: Person,
        world: World,
    ) -> int:
        """
        The target location is determined based on the personnel roles and the current time. 
        """
        if(
            person.role == Role.WORKER
            and person.workplace_id is not None
            and self._is_work_time(world)
        ):
            return person.workplace_id

        if(
            person.role == Role.STUDENT
            and person.school_id is not None
            and self._is_school_time(world)
        ):
            return person.school_id

        
        if(self._is_public_time(world)):
            probability = (
                world.require_config().public_visit_probability_weekend
                if self._is_weekend(world)
                else world.require_config().public_visit_probability_weekday
            )
            
            if world.rng.random() < probability:
                public_place_id = self._choose_public_place(world)

                if public_place_id is not None:
                    return public_place_id

        return person.home_id

    def _is_work_time(
        self,
        world: World,
    ) -> bool:
        if self._is_weekend(world):
            return False

        hour = self._current_hour(world)

        return(
            world.require_config().work_start_hour
            <= hour <=
            world.require_config().work_end_hour
        )
        

    def _is_school_time(
        self,
        world: World,
    ) -> bool:
        if self._is_weekend(world):
            return False

        hour = self._current_hour(world)

        return(
            world.require_config().school_start_hour
            <= hour <=
            world.require_config().school_end_hour
        )        

    def _is_public_time(
            self,
            world: World,
        ) -> bool:
            hour = self._current_hour(world)

            return(
                world.require_config().public_start_hour
                <= hour <=
                world.require_config().public_end_hour
            )   

    def _current_hour(self, world: World) -> float:
        """
        Assume that current_time represents the number of hours that have elapsed since the simulation began. 
        For example:
        current_time = 9.5  -> Day 1, 09:30
        current_time = 33.0 -> Day 2, 09:00
        """
        if world.current_time is None:
            raise ValueError( 
                "world.current_time must be set before mobility step"
            )

        return world.current_time % 24

    def _is_weekend(self,world:World) -> bool:
        if world.current_time is None:
                    raise ValueError( 
                        "world.current_time must be set before mobility step"
                    )
        day_index = int(world.current_time // 24)

        weekday = day_index % 7

        return weekday in {5, 6}

    def _choose_public_place(
        self,
        world: World,
    ) -> Optional[int]:
        available_place_ids = []

        for place in world.places.values():
            if place.place_type != PlaceType.PUBLIC:
                continue
            
            if(
                place.capacity is not None
                and len(place.occupants) >= place.capacity
            ):
                continue

            available_place_ids.append(place.place_id)

        if not available_place_ids:
            return None

        return int(
            world.rng.choice(available_place_ids)
        )

    def _move_person(
        self,
        person: Person,
        target_place_id: int,
        world: World,
    ) -> None:
        '''
        how person move
        '''
        

        if(target_place_id == person.current_place_id):
            return
        if(target_place_id not in world.places):
            raise KeyError(
                f"Target place {target_place_id} does not exist"
            )

        target_place = world.get_place(target_place_id)

        if(
            target_place.capacity is not None
            and len(target_place.occupants) >= target_place.capacity
        ):
            return

        world.move_person(
            person = person,
            place_id = target_place_id,
        )

