"""cosserat3 real2sim_paramiden model spec and stiffness visualization."""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from adapteddlo_muj.envs.real2sim_paramiden.base import (
    R_THICKNESS,
    STIFF_VALS_DIR,
    search_limits,
    stiff_path,
)
from adapteddlo_muj.utils.rope_stiffness import (
    backend_controller_stiffness,
    material_scale_for,
)

MUJOCO_TIMESTEP = 0.0015
RADIUS = R_THICKNESS / 2.0
STIFF_SCALE_TO_ALPHA = 1.0 / (2.0 * np.pi) ** 3

# Backends in real2sim_paramiden (xfrc omitted; same mapping as adapt).
BACKEND_MODELS = [
    "adapt",
    "native",
    "massspring",
    "cosserat2",
    "cosserat3",
    "xpbd",
    "geds",
]


def get_model_spec():
    return {"name": "cosserat3"}


def stiff_scale_to_alpha_bar(stiff_scale: float) -> float:
    return float(stiff_scale) * STIFF_SCALE_TO_ALPHA


def controller_stiffness_table(
    stiff_scale: float,
    *,
    b_a_ratio: float = 1.0,
    models: Optional[Sequence[str]] = None,
    radius: float = RADIUS,
    timestep: float = MUJOCO_TIMESTEP,
) -> Dict[str, Dict[str, float]]:
    """Map stiff_scale (paramiden search variable) to per-backend stiffness dicts."""
    alpha_bar = stiff_scale_to_alpha_bar(stiff_scale)
    beta_bar = alpha_bar * b_a_ratio
    model_names = list(models) if models is not None else BACKEND_MODELS
    table: Dict[str, Dict[str, float]] = {}
    for model_name in model_names:
        table[model_name] = backend_controller_stiffness(
            model_name,
            alpha_bar,
            beta_bar,
            radius=radius,
            timestep=timestep,
        )
    return table


def _print_stiffness_table(
    table: Dict[str, Dict[str, float]],
    stiff_scale: float,
    alpha_bar: float,
    beta_bar: float,
) -> None:
    print(f"\nstiff_scale = {stiff_scale:.4f}  alpha_bar = {alpha_bar:.6e}  beta_bar = {beta_bar:.6e}")
    print(f"radius = {RADIUS}  dt = {MUJOCO_TIMESTEP}")
    print("-" * 72)
    for model_name, values in table.items():
        mat_scale = material_scale_for(model_name)
        parts = ", ".join(f"{k}={v:.6e}" for k, v in values.items())
        print(f"  {model_name:12s} (material_scale={mat_scale:g})  {parts}")


def _bend_twist_pairs(table: Dict[str, Dict[str, float]]) -> Dict[str, tuple[float, float]]:
    """Normalize backend-specific keys to (bend, twist) for plotting."""
    pairs: Dict[str, tuple[float, float]] = {}
    for model_name, values in table.items():
        if "k_bend" in values:
            pairs[model_name] = (values["k_bend"], values["k_twist"])
        elif "k_bend_x" in values:
            pairs[model_name] = (values["k_bend_x"], values["k_twist"])
        elif "youngs_modulus" in values:
            pairs[model_name] = (values["youngs_modulus"], values["torsion_modulus"])
        elif "joint_bend" in values:
            pairs[model_name] = (values["joint_bend"], values["joint_torsion"])
        else:
            bend = next(iter(values.values()))
            twist = list(values.values())[-1]
            pairs[model_name] = (bend, twist)
    return pairs


def visualize_stiffness(
    stiff_scales: Optional[Iterable[float]] = None,
    *,
    b_a_ratio: float = 1.0,
    models: Optional[Sequence[str]] = None,
    wire_color: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> str:
    """
    Plot effective bend/twist stiffness for rope backends used in paramiden.

    Returns the path to the saved figure (or the intended save path).
    """
    model_names = list(models) if models is not None else BACKEND_MODELS
    if stiff_scales is None:
        if wire_color is not None:
            pickle_path = stiff_path(wire_color, "cosserat3")
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    alpha_saved, b_a_saved = pickle.load(f)
                stiff_scale = alpha_saved * (2.0 * np.pi) ** 3
                b_a_ratio = b_a_saved
                stiff_scales = [stiff_scale]
            else:
                stiff_scales = np.linspace(0.0, 2.0, 41)
        else:
            stiff_scales = np.linspace(0.0, 2.0, 41)

    stiff_scales = np.asarray(list(stiff_scales), dtype=float)
    if stiff_scales.size == 1:
        table = controller_stiffness_table(
            float(stiff_scales[0]),
            b_a_ratio=b_a_ratio,
            models=model_names,
        )
        alpha_bar = stiff_scale_to_alpha_bar(float(stiff_scales[0]))
        beta_bar = alpha_bar * b_a_ratio
        _print_stiffness_table(table, float(stiff_scales[0]), alpha_bar, beta_bar)

        pairs = _bend_twist_pairs(table)
        names = list(pairs.keys())
        bend_vals = [pairs[n][0] for n in names]
        twist_vals = [pairs[n][1] for n in names]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width / 2, bend_vals, width, label="bend")
        ax.bar(x + width / 2, twist_vals, width, label="twist")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("effective stiffness")
        ax.set_title(
            f"Rope controller stiffness (stiff_scale={stiff_scales[0]:.4f}, b/a={b_a_ratio:.4f})"
        )
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        for model_name in model_names:
            bend_curve = []
            twist_curve = []
            for s in stiff_scales:
                pair = _bend_twist_pairs(
                    controller_stiffness_table(
                        float(s),
                        b_a_ratio=b_a_ratio,
                        models=[model_name],
                    )
                )[model_name]
                bend_curve.append(pair[0])
                twist_curve.append(pair[1])
            axes[0].plot(stiff_scales, bend_curve, label=model_name)
            axes[1].plot(stiff_scales, twist_curve, label=model_name)

        for ax, label in zip(axes, ("bend", "twist")):
            ax.set_yscale("log")
            ax.set_xlabel("stiff_scale")
            ax.set_ylabel(f"{label} stiffness")
            ax.set_title(f"Effective {label} stiffness vs stiff_scale")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")
        fig.suptitle(f"b/a ratio = {b_a_ratio:.4f}")
        fig.tight_layout()

    if save_path is None:
        os.makedirs(STIFF_VALS_DIR, exist_ok=True)
        save_path = os.path.join(STIFF_VALS_DIR, "cosserat3_stiffness_viz.png")
    fig.savefig(save_path, dpi=150)
    print(f"Saved figure: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Visualize effective stiffness values for real2sim_paramiden rope backends."
    )
    parser.add_argument(
        "--stiff-scale",
        type=float,
        default=None,
        help="Single stiff_scale to compare backends (default: sweep 0..2).",
    )
    parser.add_argument(
        "--b-a",
        type=float,
        default=1.0,
        help="beta/alpha ratio (default: 1.0).",
    )
    parser.add_argument(
        "--wire-color",
        type=str,
        default=None,
        help="Load saved cosserat3 stiffness pickle for this wire color.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated backend list (default: all paramiden backends).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figure without opening an interactive window.",
    )
    args = parser.parse_args(argv)

    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    stiff_scales = None
    if args.stiff_scale is not None:
        stiff_scales = [args.stiff_scale]
    elif args.wire_color is None:
        lo, hi = search_limits("cosserat3")[0]
        stiff_scales = np.linspace(lo, hi, 41)

    visualize_stiffness(
        stiff_scales=stiff_scales,
        b_a_ratio=args.b_a,
        models=models,
        wire_color=args.wire_color,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
