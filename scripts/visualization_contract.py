"""Dependency-free state contract shared by the launcher and dashboard."""

IDLE_STATE = {
    "session": {"status": "waiting_for_replay"},
    "event": {
        "name": "FABLE idle",
        "description": "The stack is ready. Start an experiment from another terminal.",
        "roles": {},
        "pattern": {"path": "root", "kind": "operator", "op": "idle", "label": "Waiting", "children": []},
    },
    "counts": {"total": 0, "active": 0, "completed": 0, "expired": 0},
    "instances": [],
}
