sird_sim
├── config.py                     # imports domain.place (uses PlaceType as the contact_k mapping key)
├── partition.py                  # Standalone — only imports numpy; no internal project imports
├── domain
│   ├── __init__.py
│   ├── person.py                 # Person dataclass; imports events.event (type of the pending_event field)
│   ├── place.py                  # Place dataclass, standalone
│   └── relationship.py           # Relationship dataclass, standalone
├── world.py                      # imports domain.*, config, events.event_queue; holds entity containers and indices
├── events
│   ├── __init__.py
│   ├── event.py                  # Event base class + subclasses; only imports the standard library
│   └── event_queue.py            # imports event.py
├── systems
│   ├── __init__.py
│   ├── schedule_system.py        # imports world, domain (decides and performs person movement)
│   ├── contact_system.py         # imports world (samples contact pairs according to contact_k)
│   ├── transmission_system.py    # imports world, domain.person, events.event (processes transmission, schedules BecomeInfectiousEvent)
│   ├── health_event_system.py    # imports world, domain.person, events.event, config (processes recovery/death/immunity-waning events)
│   ├── metrics_system.py         # imports world, domain.person, domain.place (collects SIRD and place-occupancy metrics)
│   └── movement_system.py        # Empty file, unused (movement logic was merged into schedule_system.py)
├── engine.py                     # imports world, systems.*  — the sole orchestrator
├── generation.py                 # imports partition, domain.*, config, world (generates the initial population, places, and relationship network)
├── controller.py                 # imports config, engine, generation, systems.*, world — glue layer between the UI and the engine
├── plotting.py                   # imports domain.place, domain.relationship, systems.metrics_system, world — pure Matplotlib, does not import Streamlit
├── mobility_experiment.py        # imports controller, config, domain.place, systems.metrics_system — multi-scenario mobility comparison experiment
├── view.py                       # imports controller, plotting, config, domain.place, mobility_experiment — Streamlit dashboard
└── main.py                       # from sird_sim.view import render; render()

tests
├── test_domain.py
├── test_event_queue.py
├── test_partition.py
├── test_transmission_system.py
├── test_health_event_system.py
├── test_engine.py
├── test_mobility_experiment.py
└── test_plotting.py