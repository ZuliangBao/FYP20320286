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
    """所有事件的公共部分：什么时候发生、对谁发生、要不要被跳过。
    子类只增加"这种事件特有"的字段，不重复这三个。
    """
    time: float
    sequence: int
    person_id: int
    cancelled: bool = False

    kind: ClassVar[EventType]  # 子类必须覆盖，声明自己是哪种事件


@dataclass(kw_only=True)
class BecomeInfectiousEvent(Event):
    kind: ClassVar[EventType] = EventType.BECOME_INFECTIOUS
    source_person_id: Optional[int] = None  # 谁传染的，留给以后做传播链追溯


@dataclass(kw_only=True)
class RecoverEvent(Event):
    kind: ClassVar[EventType] = EventType.RECOVER


@dataclass(kw_only=True)
class DieEvent(Event):
    kind: ClassVar[EventType] = EventType.DIE
    cause: str = "disease"  # 留出以后加入背景死亡率等其他死因的空间


@dataclass(kw_only=True)
class ImmunityWanesEvent(Event):
    kind: ClassVar[EventType] = EventType.IMMUNITY_WANES