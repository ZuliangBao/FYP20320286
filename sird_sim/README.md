sird_sim
├── config.py                  # 独立,不 import 项目里任何其他模块
├── domain
│   ├── __init__.py
│   ├── person.py              # Person dataclass
│   ├── place.py                # Place dataclass
│   └── relationship.py         # Relationship dataclass
├── world.py                    # import domain.，持有实体容器和索引
├── events
│   ├── __init__.py
│   ├── event.py                 # Event 基类+子类,只 import 标准库
│   └── event_queue.py            # import event.py
├── systems
│   ├── __init__.py
│   ├── schedule_system.py        # import world, domain
│   ├── movement_system.py        # import world, domain
│   ├── contact_system.py         # import world, domain, events
│   ├── health_system.py          # import world, events
│   └── metrics_system.py         # import world
├── engine.py                     # import world, events, systems.  —— 唯一的编排者
├── view.py                       # 只 import engine 暴露的公开方法
├── controller.py                 # import view, engine
└── main.py                       # 把上面全部装配起来
tests
├── test_domain.py
├── test_event_queue.py
├── test_systems.py
└── test_engine.py