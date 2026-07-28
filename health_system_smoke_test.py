from types import SimpleNamespace

import numpy as np

from sird_sim.domain.person import (
    HealthState,
    Person,
    Role,
)
from sird_sim.domain.place import Place, PlaceType
from sird_sim.events.event import (
    BecomeInfectiousEvent,
    DieEvent,
    RecoverEvent,
)

from sird_sim.events.event_queue import EventQueue
from sird_sim.systems.health_event_system import (
    HealthEventSystem,
)
from sird_sim.world import World


def main() -> None:
    # ============================================================
    # 1. 手动构造一个只有两个人、一个家庭的小世界
    # ============================================================

    home_id = 0
    infected_id = 0
    susceptible_id = 1

    infected = Person(
        person_id=infected_id,
        role=Role.WORKER,
        home_id=home_id,
        current_place_id=home_id,
        health_state=HealthState.INFECTED,
    )

    susceptible = Person(
        person_id=susceptible_id,
        role=Role.WORKER,
        home_id=home_id,
        current_place_id=home_id,
        health_state=HealthState.SUSCEPTIBLE,
    )

    home = Place(
        place_id=home_id,
        place_type=PlaceType.HOME,
        occupants={
            infected_id,
            susceptible_id,
        },
    )

    config = SimpleNamespace(
        tick_duration=1.0,
        infection_probability=1.0,
        recovery_rate=0.5,
        deadly_rate=0.1,
    )

    world = World(
        persons={
            infected_id: infected,
            susceptible_id: susceptible,
        },
        places={
            home_id: home,
        },
        relationships={
            infected_id: [],
            susceptible_id: [],
        },
        rng=np.random.default_rng(20260726),
        config=config,
        current_time=0.0,
    )

    # 如果这两个字段已经正式加入 World，可以直接在 World(...)
    # 构造函数中传入。这里单独赋值也能完成 smoke test。
    world.event_queue = EventQueue()
    world.pending_contacts = {
        (infected_id, susceptible_id)
    }

    health_system = HealthEventSystem()

    # ============================================================
    # 2. 第一次 step：接触导致感染事件被排入队列
    # ============================================================

    health_system.step(world)

    infection_event = susceptible.pending_event

    assert (
        susceptible.health_state
        == HealthState.SUSCEPTIBLE
    ), (
        "第一次 step 后不应立即改变状态；"
        "BecomeInfectiousEvent 应留到下一次 step 处理"
    )

    assert isinstance(
        infection_event,
        BecomeInfectiousEvent,
    ), (
        "易感者应当获得 BecomeInfectiousEvent"
    )

    assert infection_event.person_id == susceptible_id
    assert infection_event.source_person_id == infected_id
    assert infection_event.time == world.current_time

    assert len(world.event_queue) == 1

    print("Phase 1 passed:")
    print(
        f"  scheduled {infection_event.kind.name} "
        f"for person {susceptible_id} "
        f"at t={infection_event.time:.4f}"
    )

    # ContactSystem 正常情况下会在下一步重建 contacts。
    # 此处清空，避免旧接触干扰事件阶段测试。
    world.pending_contacts.clear()

    # ============================================================
    # 3. 第二次 step：处理 BecomeInfectiousEvent
    # ============================================================

    world.current_time = infection_event.time

    health_system.step(world)

    assert (
        susceptible.health_state
        == HealthState.INFECTED
    ), (
        "BecomeInfectiousEvent 到期后，"
        "该人的状态应变成 INFECTED"
    )

    outcome_event = susceptible.pending_event

    assert isinstance(
        outcome_event,
        (RecoverEvent, DieEvent),
    ), (
        "变成 INFECTED 后，应排出 RecoverEvent "
        "或 DieEvent"
    )

    assert outcome_event.person_id == susceptible_id
    assert outcome_event.time >= world.current_time

    # BecomeInfectiousEvent 已经弹出，
    # RecoverEvent/DieEvent 又被加入，所以仍应有一个有效事件。
    assert len(world.event_queue) == 1

    print("Phase 2 passed:")
    print(
        f"  person {susceptible_id} became INFECTED"
    )
    print(
        f"  scheduled {outcome_event.kind.name} "
        f"at t={outcome_event.time:.4f}"
    )

    # ============================================================
    # 4. 第三次 step：进一步验证康复或死亡 handler
    # ============================================================

    world.current_time = outcome_event.time

    health_system.step(world)

    if isinstance(outcome_event, RecoverEvent):
        assert (
            susceptible.health_state
            == HealthState.RECOVERED
        )
        assert susceptible.pending_event is None
        assert susceptible.current_place_id == home_id
        assert susceptible_id in home.occupants

        print("Phase 3 passed:")
        print(
            f"  person {susceptible_id} recovered"
        )

    else:
        assert (
            susceptible.health_state
            == HealthState.DEAD
        )
        assert susceptible.pending_event is None
        assert susceptible.current_place_id is None
        assert susceptible_id not in home.occupants

        print("Phase 3 passed:")
        print(
            f"  person {susceptible_id} died "
            "and was removed from the place"
        )

    assert len(world.event_queue) == 0

    print()
    print("HealthSystem smoke test passed.")


if __name__ == "__main__":
    main()