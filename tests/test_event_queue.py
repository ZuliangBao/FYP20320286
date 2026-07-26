import pytest

from sird_sim.events.event import (
    BecomeInfectiousEvent,
    DieEvent,
    EventType,
    RecoverEvent,
)
from sird_sim.events.event_queue import EventQueue


# ============================================================
# schedule
# ============================================================

def test_schedule_returns_created_event() -> None:
    queue = EventQueue()

    event = queue.schedule(
        BecomeInfectiousEvent,
        time=3.5,
        person_id=7,
        source_person_id=2,
    )

    assert isinstance(event, BecomeInfectiousEvent)
    assert event.kind is EventType.BECOME_INFECTIOUS
    assert event.time == 3.5
    assert event.person_id == 7
    assert event.source_person_id == 2
    assert event.sequence == 0
    assert event.cancelled is False


def test_schedule_assigns_increasing_sequence_numbers() -> None:
    queue = EventQueue()

    first = queue.schedule(
        RecoverEvent,
        time=1.0,
        person_id=1,
    )
    second = queue.schedule(
        RecoverEvent,
        time=1.0,
        person_id=2,
    )
    third = queue.schedule(
        RecoverEvent,
        time=1.0,
        person_id=3,
    )

    assert [
        first.sequence,
        second.sequence,
        third.sequence,
    ] == [0, 1, 2]


def test_len_and_is_empty_after_scheduling() -> None:
    queue = EventQueue()

    assert queue.is_empty()
    assert len(queue) == 0

    queue.schedule(
        RecoverEvent,
        time=1.0,
        person_id=1,
    )

    assert not queue.is_empty()
    assert len(queue) == 1


# ============================================================
# ordering
# ============================================================

def test_events_are_popped_in_time_order() -> None:
    queue = EventQueue()

    late = queue.schedule(
        RecoverEvent,
        time=10.0,
        person_id=1,
    )
    early = queue.schedule(
        DieEvent,
        time=2.0,
        person_id=2,
    )
    middle = queue.schedule(
        RecoverEvent,
        time=5.0,
        person_id=3,
    )

    assert queue.pop_next() is early
    assert queue.pop_next() is middle
    assert queue.pop_next() is late


def test_same_time_events_are_popped_in_schedule_order() -> None:
    queue = EventQueue()

    first = queue.schedule(
        RecoverEvent,
        time=5.0,
        person_id=1,
    )
    second = queue.schedule(
        DieEvent,
        time=5.0,
        person_id=2,
    )
    third = queue.schedule(
        RecoverEvent,
        time=5.0,
        person_id=3,
    )

    assert queue.pop_next() is first
    assert queue.pop_next() is second
    assert queue.pop_next() is third


def test_peek_time_does_not_remove_active_event() -> None:
    queue = EventQueue()

    event = queue.schedule(
        RecoverEvent,
        time=4.0,
        person_id=1,
    )

    assert queue.peek_time() == 4.0
    assert queue.peek_time() == 4.0
    assert len(queue) == 1
    assert queue.pop_next() is event


def test_empty_queue_returns_none() -> None:
    queue = EventQueue()

    assert queue.peek_time() is None
    assert queue.pop_next() is None
    assert queue.is_empty()
    assert len(queue) == 0


# ============================================================
# cancellation consistency
# ============================================================

def test_cancelled_event_stops_counting_immediately() -> None:
    queue = EventQueue()

    event = queue.schedule(
        RecoverEvent,
        time=2.0,
        person_id=1,
    )

    queue.cancel(event)

    assert event.cancelled is True
    assert len(queue) == 0
    assert queue.is_empty()


def test_peek_time_skips_cancelled_events_at_front() -> None:
    queue = EventQueue()

    cancelled_first = queue.schedule(
        RecoverEvent,
        time=1.0,
        person_id=1,
    )
    active_second = queue.schedule(
        DieEvent,
        time=3.0,
        person_id=2,
    )

    queue.cancel(cancelled_first)

    assert queue.peek_time() == 3.0
    assert len(queue) == 1
    assert queue.pop_next() is active_second


def test_peek_time_skips_multiple_cancelled_front_events() -> None:
    queue = EventQueue()

    first = queue.schedule(
        RecoverEvent,
        time=1.0,
        person_id=1,
    )
    second = queue.schedule(
        RecoverEvent,
        time=2.0,
        person_id=2,
    )
    active = queue.schedule(
        DieEvent,
        time=3.0,
        person_id=3,
    )

    queue.cancel(first)
    queue.cancel(second)

    assert queue.peek_time() == 3.0
    assert len(queue) == 1
    assert queue.pop_next() is active


def test_pop_next_skips_cancelled_front_events() -> None:
    queue = EventQueue()

    cancelled = queue.schedule(
        RecoverEvent,
        time=1.0,
        person_id=1,
    )
    active = queue.schedule(
        DieEvent,
        time=2.0,
        person_id=2,
    )

    queue.cancel(cancelled)

    assert queue.pop_next() is active
    assert queue.pop_next() is None


def test_all_cancelled_events_make_queue_logically_empty() -> None:
    queue = EventQueue()

    first = queue.schedule(
        RecoverEvent,
        time=1.0,
        person_id=1,
    )
    second = queue.schedule(
        DieEvent,
        time=2.0,
        person_id=2,
    )

    queue.cancel(first)
    queue.cancel(second)

    assert len(queue) == 0
    assert queue.is_empty()
    assert queue.peek_time() is None
    assert queue.pop_next() is None


def test_cancelled_event_inside_heap_is_not_counted() -> None:
    queue = EventQueue()

    active_first = queue.schedule(
        RecoverEvent,
        time=1.0,
        person_id=1,
    )
    cancelled_middle = queue.schedule(
        DieEvent,
        time=2.0,
        person_id=2,
    )
    active_last = queue.schedule(
        RecoverEvent,
        time=3.0,
        person_id=3,
    )

    queue.cancel(cancelled_middle)

    # The cancelled event is not at the front, so lazy deletion may
    # leave it physically in the heap. Public queue semantics must
    # nevertheless count only active events.
    assert len(queue) == 2
    assert not queue.is_empty()
    assert queue.peek_time() == 1.0

    assert queue.pop_next() is active_first
    assert queue.peek_time() == 3.0
    assert queue.pop_next() is active_last
    assert queue.is_empty()


def test_cancelling_event_twice_is_idempotent() -> None:
    queue = EventQueue()

    event = queue.schedule(
        RecoverEvent,
        time=1.0,
        person_id=1,
    )

    queue.cancel(event)
    queue.cancel(event)

    assert event.cancelled is True
    assert len(queue) == 0
    assert queue.is_empty()


# ============================================================
# count changes
# ============================================================

def test_pop_active_event_decrements_length() -> None:
    queue = EventQueue()

    first = queue.schedule(
        RecoverEvent,
        time=1.0,
        person_id=1,
    )
    second = queue.schedule(
        DieEvent,
        time=2.0,
        person_id=2,
    )

    assert len(queue) == 2

    assert queue.pop_next() is first
    assert len(queue) == 1

    assert queue.pop_next() is second
    assert len(queue) == 0
    assert queue.is_empty()


def test_discarding_cancelled_front_does_not_double_decrement() -> None:
    queue = EventQueue()

    cancelled = queue.schedule(
        RecoverEvent,
        time=1.0,
        person_id=1,
    )
    active = queue.schedule(
        DieEvent,
        time=2.0,
        person_id=2,
    )

    queue.cancel(cancelled)

    # cancel() already removed the event from the logical count.
    assert len(queue) == 1

    # Cleaning the cancelled heap entry must not decrement again.
    assert queue.peek_time() == 2.0
    assert len(queue) == 1

    assert queue.pop_next() is active
    assert len(queue) == 0


# ============================================================
# invalid cancellation
# ============================================================

def test_cannot_cancel_event_from_another_queue() -> None:
    first_queue = EventQueue()
    second_queue = EventQueue()

    foreign_event = first_queue.schedule(
        RecoverEvent,
        time=1.0,
        person_id=1,
    )

    with pytest.raises(
        ValueError,
        match="not currently scheduled",
    ):
        second_queue.cancel(foreign_event)


def test_cannot_cancel_event_after_it_was_popped() -> None:
    queue = EventQueue()

    event = queue.schedule(
        RecoverEvent,
        time=1.0,
        person_id=1,
    )

    assert queue.pop_next() is event

    with pytest.raises(
        ValueError,
        match="not currently scheduled",
    ):
        queue.cancel(event)


# ============================================================
# mixed workflow
# ============================================================

def test_mixed_schedule_cancel_peek_and_pop_workflow() -> None:
    queue = EventQueue()

    first = queue.schedule(
        RecoverEvent,
        time=4.0,
        person_id=1,
    )
    second = queue.schedule(
        DieEvent,
        time=1.0,
        person_id=2,
    )
    third = queue.schedule(
        RecoverEvent,
        time=2.0,
        person_id=3,
    )
    fourth = queue.schedule(
        DieEvent,
        time=3.0,
        person_id=4,
    )

    queue.cancel(second)
    queue.cancel(fourth)

    assert len(queue) == 2
    assert queue.peek_time() == 2.0
    assert queue.pop_next() is third

    assert len(queue) == 1
    assert queue.peek_time() == 4.0
    assert queue.pop_next() is first

    assert len(queue) == 0
    assert queue.peek_time() is None
    assert queue.pop_next() is None
    assert queue.is_empty()
