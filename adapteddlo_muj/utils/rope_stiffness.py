"""Shared stiffness and damping scaling for MuJoCo rope backend controllers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# dt^2 factor for shadow-solve backends (cosserat2, xpbd, geds stretch coupling).
COINTEGRATION_DT2_FACTOR = 1.0e2

_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "rope_model_config.json"
)


def config_path() -> Path:
    return _CONFIG_PATH


def load_rope_model_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or _CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def _scale_from_config(section: str, model_name: str) -> float:
    cfg = load_rope_model_config()
    scales = cfg.get(section, {})
    return float(scales.get(model_name, 1.0))


def material_scale_for(model_name: str) -> float:
    return _scale_from_config("material_scales", model_name)


def damping_scale_for(model_name: str) -> float:
    return _scale_from_config("damping_scales", model_name)


def scale_joint_damping(base_damping: float, model_name: str) -> float:
    return float(base_damping) * damping_scale_for(model_name)


def cointegration_stiff_scale(timestep: float) -> float:
    dt = float(timestep)
    return dt * dt * COINTEGRATION_DT2_FACTOR


def scale_material(
    alpha_bar: float,
    beta_bar: float,
    material_scale: float,
) -> tuple[float, float]:
    s = float(material_scale)
    return alpha_bar * s, beta_bar * s


def circular_section_inertia(radius: float) -> tuple[float, float]:
    r = float(radius)
    ix = np.pi * r**4 / 4.0
    j1 = np.pi * r**4 / 2.0
    return ix, j1


def youngs_torsion_moduli(
    alpha_bar: float,
    beta_bar: float,
    radius: float,
    timestep: float,
    material_scale: float = 1.0,
) -> tuple[float, float]:
    """Young's and torsion moduli for xpbd / geds backends."""
    alpha_s, beta_s = scale_material(alpha_bar, beta_bar, material_scale)
    ix, j1 = circular_section_inertia(radius)
    s = cointegration_stiff_scale(timestep)
    return (alpha_s / ix) * s, (beta_s / j1) * s


def cosserat2_moduli(
    alpha_bar: float,
    beta_bar: float,
    radius: float,
    timestep: float,
    material_scale: float = 1.0,
) -> tuple[float, float, float]:
    """Stretch, bend, and twist stiffness for cosserat2 shadow backend."""
    alpha_s, beta_s = scale_material(alpha_bar, beta_bar, material_scale)
    ix, _ = circular_section_inertia(radius)
    s = cointegration_stiff_scale(timestep)
    k_stretch = (alpha_s / ix) * s
    k_bend = alpha_s * s
    k_twist = beta_s * s
    return k_stretch, k_bend, k_twist


def backend_controller_stiffness(
    model_name: str,
    alpha_bar: float,
    beta_bar: float,
    *,
    radius: float,
    timestep: float,
) -> dict[str, float]:
    """Effective stiffness parameters each rope backend passes to C++/MuJoCo."""
    mat_scale = material_scale_for(model_name)
    alpha_s, beta_s = scale_material(alpha_bar, beta_bar, mat_scale)

    if model_name in ("adapt", "xfrc", "cosserat3", "cosserat4", "cosserat", "jpqder"):
        return {"k_bend": alpha_s, "k_twist": beta_s}
    if model_name == "massspring":
        return {"k_bend_x": alpha_s, "k_bend_y": alpha_s, "k_twist": beta_s}
    if model_name in ("xpbd", "geds"):
        ym, tm = youngs_torsion_moduli(
            alpha_bar, beta_bar, radius, timestep, mat_scale
        )
        return {"youngs_modulus": ym, "torsion_modulus": tm}
    if model_name == "cosserat2":
        ks, kb, kt = cosserat2_moduli(
            alpha_bar, beta_bar, radius, timestep, mat_scale
        )
        return {"k_stretch": ks, "k_bend": kb, "k_twist": kt}
    if model_name == "native":
        ix, j1 = circular_section_inertia(radius)
        return {"joint_bend": alpha_bar / ix, "joint_torsion": beta_bar / j1}

    raise ValueError(f"Unknown rope backend: {model_name!r}")
