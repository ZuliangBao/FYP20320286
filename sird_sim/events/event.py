from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar, Optional

class EventType(Enum):
    BECOME_INFECTIOUS = auto()
    RECOVER = auto()
    DIE = auto()
    IMMUNITY_WANES = auto()

@dataclass(kw_only=True)
class Event:
    """Common fields shared by every event: when it happens, who it
    happens to, and whether it should be skipped.

    sequence breaks ties when time is equal, keeping the schedule
    order deterministic and reproducible. Subclasses only add fields
    specific to that event type; they should not repeat these three.
    """
    time: float
    sequence: int
    person_id: int
    cancelled: bool = False

    kind: ClassVar[EventType]  # Subclasses must override this to declare their event type

@dataclass(kw_only=True)
class BecomeInfectiousEvent(Event):
    kind: ClassVar[EventType] = EventType.BECOME_INFECTIOUS
    source_person_id: Optional[int] = None  # Who transmitted the infection; reserved for future transmission-chain tracing

@dataclass(kw_only=True)
class RecoverEvent(Event):
    kind: ClassVar[EventType] = EventType.RECOVER

@dataclass(kw_only=True)
class DieEvent(Event):
    kind: ClassVar[EventType] = EventType.DIE
    cause: str = "disease"  # Leaves room for adding other causes later, e.g. background mortality