"""Shared arm / IK manipulation tolerances and move settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

MovePhase = Literal["default", "first_move_to_pose", "second_move_to_pose"]

_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "manipulation_config.json"
)

_cached_config: dict[str, Any] | None = None


def config_path() -> Path:
    return _CONFIG_PATH


def load_manipulation_config(path: Path | None = None) -> dict[str, Any]:
    global _cached_config
    cfg_path = path or _CONFIG_PATH
    if path is None and _cached_config is not None:
        return _cached_config
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if path is None:
        _cached_config = cfg
    return cfg


def _float(cfg: dict[str, Any], key: str, default: float | None = None) -> float:
    if key in cfg:
        return float(cfg[key])
    if default is not None:
        return float(default)
    raise KeyError(key)


def qpos_tol_for(phase: MovePhase = "default") -> float:
    cfg = load_manipulation_config()
    if phase == "first_move_to_pose":
        return _float(cfg, "qpos_tol_first_move_to_pose", cfg.get("qpos_tol"))
    if phase == "second_move_to_pose":
        return _float(cfg, "qpos_tol_second_move_to_pose", cfg.get("qpos_tol"))
    return _float(cfg, "qpos_tol")


def max_action_for(phase: MovePhase = "default") -> float:
    cfg = load_manipulation_config()
    if phase in ("first_move_to_pose", "second_move_to_pose"):
        return _float(cfg, "max_action_move_to_pose")
    return _float(cfg, "max_action_default")


def apply_move_settings(env: Any, phase: MovePhase = "default") -> None:
    """Set qpos_tol and max_action on an env for a move phase."""
    if phase in ("first_move_to_pose", "second_move_to_pose"):
        env.qpos_tol = qpos_tol_for(phase)
        env.max_action = max_action_for(phase)
    else:
        env.qpos_tol = qpos_tol_for("default")
        env.max_action = max_action_for("default")


def init_env_manipulation_defaults(env: Any) -> None:
    """Load manipulation config onto env instance fields used by IK moves."""
    cfg = load_manipulation_config()
    env.manip_cfg = cfg
    env.qpos_tol = qpos_tol_for("default")
    env.max_action = max_action_for("default")
