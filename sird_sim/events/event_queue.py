from __future__ import annotations

import heapq
import itertools
import logging
from typing import Optional

from .event import Event

logger = logging.getLogger(__name__)

class EventQueue:
    """
    Priority queue for simulation events.

    The heap stores:

        (time, sequence, event)

    Cancelled events use lazy deletion. They remain physically inside
    the heap until they reach the front, but they are no longer counted
    as active events.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, Event]] = []
        self._sequence_counter = itertools.count()

        # Number of non-cancelled events still scheduled in this queue.
        self._active_count = 0

        # sequence -> Event
        #
        # Used to distinguish queued events from events that have
        # already been popped, and to prevent incorrect double counting.
        self._queued_events: dict[int, Event] = {}

    def schedule(
        self,
        event_cls: type[Event],
        *,
        time: float,
        person_id: int,
        **extra,
    ) -> Event:
        """Construct and schedule an event, returning it so the caller
        can track or later cancel it.

        Args:
            event_cls: The Event subclass to instantiate (e.g.
                RecoverEvent, DieEvent).
            time: The simulation time at which the event should fire.
            person_id: The person this event applies to.
            **extra: Additional fields specific to event_cls (e.g.
                source_person_id for BecomeInfectiousEvent, cause for
                DieEvent).

        Returns:
            The newly created and scheduled Event instance.
        """
        sequence = next(self._sequence_counter)

        event = event_cls(
            time=time,
            sequence=sequence,
            person_id=person_id,
            **extra,
        )

        heapq.heappush(
            self._heap,
            (event.time, event.sequence, event),
        )

        self._queued_events[event.sequence] = event
        self._active_count += 1

        logger.debug(
            "scheduled %s person_id=%s t=%.2f",
            event.kind,
            person_id,
            time,
        )

        return event

    def _discard_cancelled_front(self) -> None:
        """
        Physically remove consecutive cancelled events from the front.

        Their active-count decrement already occurred in cancel(), so
        removing them here must not change _active_count again.
        """
        while self._heap:
            _, _, event = self._heap[0]

            if not event.cancelled:
                break

            heapq.heappop(self._heap)
            self._queued_events.pop(event.sequence, None)

            logger.debug(
                "discard cancelled %s person_id=%s t=%.2f",
                event.kind,
                event.person_id,
                event.time,
            )

    def peek_time(self) -> Optional[float]:
        """
        Return the time of the next active event.

        This method may physically discard cancelled events from the
        front, but it does not remove the next active event.
        """
        self._discard_cancelled_front()

        if not self._heap:
            return None

        return self._heap[0][0]

    def pop_next(self) -> Optional[Event]:
        """
        Pop and return the next active event.

        Return None when no active event remains.
        """
        self._discard_cancelled_front()

        if not self._heap:
            return None

        _, _, event = heapq.heappop(self._heap)

        self._queued_events.pop(event.sequence)
        self._active_count -= 1

        logger.debug(
            "popped %s person_id=%s t=%.2f",
            event.kind,
            event.person_id,
            event.time,
        )

        return event

    def cancel(self, event: Event) -> None:
        """Cancel an event using lazy deletion.

        The event stops counting as active immediately, but remains in
        the heap until it reaches the front. Calling this again on an
        already-cancelled event is a no-op.

        Raises:
            ValueError: If event is not currently scheduled in this
                queue (e.g. already popped, or belongs to a different
                EventQueue).
        """
        queued_event = self._queued_events.get(event.sequence)

        if queued_event is not event:
            raise ValueError(
                "Cannot cancel an event that is not currently "
                "scheduled in this EventQueue"
            )

        # Make cancellation idempotent.
        if event.cancelled:
            return

        event.cancelled = True
        self._active_count -= 1

        logger.debug(
            "cancelled %s person_id=%s t=%.2f",
            event.kind,
            event.person_id,
            event.time,
        )

    def is_empty(self) -> bool:
        """
        Return True when no active events remain.

        If every remaining heap entry is cancelled, clean them up now.
        """
        if self._active_count == 0:
            self._discard_cancelled_front()

        return self._active_count == 0

    def __len__(self) -> int:
        """
        Return the number of active, non-cancelled events.
        """
        return self._active_count