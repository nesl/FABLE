"""Project FABLE runtime objects into the dashboard's small JSON contract."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from fable.language.pattern_parser import Expr
from fable.runtime.ce_instance import CEInstance, PatternPath


def runtime_visualization_state(runtime: object) -> dict[str, Any]:
    """Return current CE progress. Call while holding the runtime lock."""
    manager = runtime.manager  # type: ignore[attr-defined]
    event = runtime.event  # type: ignore[attr-defined]
    active = manager.active_instances()
    completed = manager.completed_instances()
    expired = manager.expired_instances()
    instances = [
        *(_instance(row, "active", event.pattern) for row in active),
        *(_instance(row, "completed", event.pattern) for row in completed),
        *(_instance(row, "expired", event.pattern) for row in expired),
    ]
    lead_pool = instances
    lead_id = None
    if lead_pool:
        lead = max(
            lead_pool,
            key=lambda row: (
                row["progress"]["fraction"], row["progress"]["satisfied"],
                row["status"] == "completed",
                -datetime.fromisoformat(
                    row["satisfied_at"].get("root", row["matched_at"])
                ).timestamp(),
                -datetime.fromisoformat(row["matched_at"]).timestamp(),
            ),
        )
        lead_id = lead["instance_id"]
    for row in instances:
        row["lead"] = row["instance_id"] == lead_id
    instances.sort(
        key=lambda row: (
            row["lead"], row["status"] == "active",
            row["progress"]["fraction"], row["matched_at"], row["instance_id"],
        ),
        reverse=True,
    )
    stages = _stages(event.pattern, (), event.roles)
    return {
        "lead_instance_id": lead_id,
        "event": {
            "name": event.name,
            "description": event.description,
            "roles": dict(event.roles),
            "pattern": _pattern(event.pattern, ()),
            "stages": stages,
        },
        "counts": {
            "total": len(instances),
            "active": len(active),
            "completed": len(completed),
            "expired": len(expired),
        },
        "instances": instances,
    }


def _instance(instance: CEInstance, status: str, pattern: Expr) -> dict[str, Any]:
    active_paths = set(instance.activated_at) - set(instance.satisfied_at) - set(instance.failed_paths)
    leaf_paths = _leaf_paths(pattern, ())
    satisfied_leaves = sum(path in instance.satisfied_at for path in leaf_paths)
    return {
        "instance_id": instance.instance_id,
        "status": status,
        "matched_at": instance.matched_at.isoformat(),
        "matched_source": instance.matched_source,
        "bindings": dict(sorted(instance.bindings.items())),
        "progress": {
            "satisfied": satisfied_leaves,
            "total": len(leaf_paths),
            "fraction": satisfied_leaves / len(leaf_paths) if leaf_paths else 0.0,
        },
        "satisfied_paths": [_path(path) for path in sorted(instance.satisfied_at)],
        "satisfied_at": {
            _path(path): timestamp.isoformat()
            for path, timestamp in sorted(instance.satisfied_at.items())
        },
        "active_paths": [_path(path) for path in sorted(active_paths)],
        "failed_paths": {
            _path(path): reason for path, reason in sorted(instance.failed_paths.items())
        },
        "expires_at": {
            _path(path): value.isoformat() for path, value in sorted(instance.expires_at.items())
        },
        "sustain_started_at": {
            _path(path): value.isoformat()
            for path, value in sorted(instance.sustain_started_at.items())
        },
    }


def _leaf_paths(expr: Expr, path: PatternPath) -> tuple[PatternPath, ...]:
    if expr.is_predicate:
        return (path,)
    return tuple(
        leaf
        for index, child in enumerate(expr.children)
        for leaf in _leaf_paths(child, path + (index,))
    )


def _stages(expr: Expr, path: PatternPath, roles: dict[str, str]) -> list[dict[str, Any]]:
    # Earlier convoy members may leave before the final member arrives. Flatten
    # the nested ordering/concurrency structure into four presentation stages.
    if (
        expr.op == "seq" and len(expr.children) == 2
        and expr.children[0].op == "enters"
        and expr.children[1].op == "all" and len(expr.children[1].children) == 2
        and expr.children[1].children[0].op == "exits"
        and expr.children[1].children[1].op == "seq"
        and len(expr.children[1].children[1].children) == 2
        and expr.children[1].children[1].children[0].op == "enters"
        and expr.children[1].children[1].children[1].op == "all"
        and len(expr.children[1].children[1].children[1].children) == 2
        and expr.children[1].children[1].children[1].children[0].op == "exits"
        and expr.children[1].children[1].children[1].children[1].op == "seq"
        and len(expr.children[1].children[1].children[1].children[1].children) == 2
        and expr.children[1].children[1].children[1].children[1].children[0].op == "enters"
        and expr.children[1].children[1].children[1].children[1].children[1].op == "exits"
    ):
        arrival_paths = [path + (0,), path + (1, 1, 0), path + (1, 1, 1, 1, 0)]
        exit_paths = [path + (1, 0), path + (1, 1, 1, 0), path + (1, 1, 1, 1, 1)]
        stages = [
            {
                "path": _path(arrival_path),
                "label": f"Vehicle {index + 1} arrives",
                "leaf_paths": [_path(arrival_path)],
            }
            for index, arrival_path in enumerate(arrival_paths)
        ]
        stages.append({
            "path": _path(path),
            "label": "All three vehicles gone",
            "leaf_paths": [_path(exit_path) for exit_path in exit_paths],
            "activation_paths": [_path(arrival_path) for arrival_path in arrival_paths],
        })
        return stages
    children = tuple(enumerate(expr.children)) if expr.op == "seq" else ((None, expr),)
    stages = []
    for index, child in children:
        child_path = path if index is None else path + (index,)
        stages.append({
            "path": _path(child_path),
            "label": _stage_label(child, roles),
            "leaf_paths": [_path(value) for value in _leaf_paths(child, child_path)],
        })
    return stages


def _stage_label(expr: Expr, roles: dict[str, str]) -> str:
    leaves = _predicate_leaves(expr)
    if expr.op == "all" and leaves and all(leaf.op == "present" for leaf in leaves):
        subjects = [_predicate_subject(leaf, roles) for leaf in leaves]
        if len(subjects) == 2 and {roles.get(role) for leaf in leaves for role in leaf.args.values() if isinstance(role, str) and role in roles} == {"vehicle", "person"}:
            return "Vehicle and person present"
    if expr.op == "all" and leaves and all(leaf.op == "near" for leaf in leaves):
        participants = {
            str(value)
            for leaf in leaves
            for value in leaf.args.values()
            if isinstance(value, str) and value in roles
        }
        if len(participants) == 3 and all(
            _role_label(role, roles).startswith("Vehicle ") for role in participants
        ):
            return "Three vehicles together"
    if leaves and len(leaves) > 1 and all(leaf.op == "exits" for leaf in leaves):
        subjects = [_predicate_subject(leaf, roles) for leaf in leaves]
        participant_roles = [
            str(value)
            for leaf in leaves
            for value in leaf.args.values()
            if isinstance(value, str) and value in roles
        ]
        if all(roles[role] == "vehicle" for role in participant_roles):
            if len(subjects) == 2:
                return "Both vehicles gone"
            if len(subjects) == 3:
                return "All three vehicles gone"
        return " and ".join(subjects) + " exit"
    if len(leaves) > 1:
        return " and ".join(_predicate_sentence(leaf, roles) for leaf in leaves)
    return _predicate_sentence(leaves[0], roles) if leaves else expr.op.replace("_", " ").capitalize()


def _predicate_leaves(expr: Expr) -> list[Expr]:
    if expr.is_predicate:
        return [expr]
    return [leaf for child in expr.children for leaf in _predicate_leaves(child)]


def _role_label(role: str, roles: dict[str, str]) -> str:
    role_class = str(roles.get(role, "object")).capitalize()
    if role.endswith("_A"):
        return f"{role_class} 1"
    if role.endswith("_B"):
        return f"{role_class} 2"
    if role.endswith("_C"):
        return f"{role_class} 3"
    if role == "LEADER":
        return "Leader vehicle"
    if role == "FOLLOWER":
        return "Follower vehicle"
    return role.replace("_", " ").title()


def _predicate_subject(expr: Expr, roles: dict[str, str]) -> str:
    role = next((str(value) for value in expr.args.values() if isinstance(value, str) and value in roles), "Object")
    return _role_label(role, roles)


def _predicate_sentence(expr: Expr, roles: dict[str, str]) -> str:
    role_values = [str(value) for value in expr.args.values() if isinstance(value, str) and value in roles]
    subjects = [_role_label(value, roles) for value in role_values]
    if expr.op == "enters":
        return f"{subjects[0]} arrives"
    if expr.op == "exits":
        return f"{subjects[0]} exits"
    if expr.op == "present" and subjects:
        return f"{subjects[0]} present"
    if expr.op == "audio_event":
        return f'{str(expr.args.get("class", "audio event")).capitalize()} detected'
    if expr.op == "near" and len(subjects) == 2:
        return "Two vehicles together" if all(value.startswith("Vehicle ") for value in subjects) else f"{subjects[0]} approaches {subjects[1]}"
    if expr.op == "follows" and len(subjects) == 2:
        return f"{subjects[1]} follows {subjects[0]}"
    return _predicate_label(expr).replace("_", " ").capitalize()


def _pattern(expr: Expr, path: PatternPath) -> dict[str, Any]:
    if expr.is_predicate:
        return {
            "path": _path(path),
            "kind": "predicate",
            "op": expr.op,
            "label": _predicate_label(expr),
            "children": [],
        }
    details = []
    if expr.op == "k_of_n":
        details.append(f"{expr.k} of {len(expr.children)}")
    if expr.op in {"all", "k_of_n"} and expr.window_ms is not None:
        details.append(_duration(expr.window_ms))
    if expr.op == "within":
        if expr.min_ms is not None:
            details.append(f"{_duration(expr.min_ms)}–{_duration(expr.max_ms or 0)}")
        else:
            details.append(f"≤ {_duration(expr.max_ms or 0)}")
    if expr.op == "for":
        details.append(_duration(expr.duration_ms or 0))
    names = {"seq": "Sequence", "all": "All", "any": "Any", "k_of_n": "K of N", "within": "Within", "for": "For"}
    label = names.get(expr.op, expr.op)
    if details:
        label += " · " + " · ".join(details)
    return {
        "path": _path(path), "kind": "operator", "op": expr.op, "label": label,
        "children": [_pattern(child, path + (index,)) for index, child in enumerate(expr.children)],
    }


def _predicate_label(expr: Expr) -> str:
    visual = [str(value) for value in expr.args.values() if isinstance(value, str) and value.isupper()]
    operation = expr.op.replace("_", " ")
    if not visual:
        return operation
    if len(visual) == 1:
        return f"{visual[0]} {operation}"
    return f"{visual[0]} {operation} {' '.join(visual[1:])}"


def _path(path: PatternPath) -> str:
    return "root" if not path else ".".join(str(value) for value in path)


def _duration(milliseconds: int) -> str:
    seconds = milliseconds / 1000
    return f"{seconds:g}s"
