from __future__ import annotations

import heapq
import itertools
import logging
from typing import Optional

from .event import Event

logger = logging.getLogger(__name__)


class EventQueue:
    """对 heapq 的薄封装：只负责按时间顺序吐出事件，不知道任何业务逻辑。

    堆里存 (time, sequence, event) 三元组：
    - time 是主排序键
    - sequence 是同一时刻的 tie-breaker，保证同一个 seed 下处理顺序完全确定
    - heapq 比较 tuple 时，time/sequence 就已经能分出大小，永远不会去比较
      event 对象本身，所以 Event 不需要实现 __lt__。
    """

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, Event]] = []
        self._sequence_counter = itertools.count()

    def schedule(self, event_cls: type[Event], *, time: float, person_id: int, **extra) -> Event:
        event = event_cls(
            time=time,
            sequence=next(self._sequence_counter),
            person_id=person_id,
            **extra,
        )
        heapq.heappush(self._heap, (event.time, event.sequence, event))
        logger.debug("scheduled %s person_id=%s t=%.2f", event.kind, person_id, time)
        return event

    def pop_next(self) -> Optional[Event]:
        """弹出下一个未被取消的事件；队列空则返回 None。"""
        while self._heap:
            _, _, event = heapq.heappop(self._heap)
            if event.cancelled:
                logger.debug("skip cancelled %s person_id=%s", event.kind, event.person_id)
                continue
            return event
        return None

    def peek_time(self) -> Optional[float]:
        """只看下一个事件的时间，不弹出。"""
        return self._heap[0][0] if self._heap else None

    def cancel(self, event: Event) -> None:
        event.cancelled = True  # 懒删除：不从堆里搬移，弹出时再跳过

    def is_empty(self) -> bool:
        return not self._heap

    def __len__(self) -> int:
        return len(self._heap)