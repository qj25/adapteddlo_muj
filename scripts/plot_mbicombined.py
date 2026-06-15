"""Combined MBI plot for adapt, native, massspring, and cosserat."""

from __future__ import annotations

import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import matplotlib.pyplot as plt
import numpy as np

from validation_metrics import COMBINED_MODELS, load_result, mbi_avg_deviation

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_PATH = os.path.join(REPO_ROOT, "adapteddlo_muj", "data", "figs", "mbi_combined.pdf")
PLOT_EXCLUDED_MODELS = frozenset({"jpqder"})


def mbi_plot(b_a: np.ndarray, theta_crit: np.ndarray, *, color: str) -> float:
    b_a_base = b_a.copy()
    theta_crit_base = 2 * np.pi * np.sqrt(3) / b_a_base
    avg_deviation = float(np.linalg.norm(theta_crit_base - theta_crit) / len(theta_crit))
    max_devi_theta_crit = np.max(np.abs(theta_crit_base - theta_crit))
    print(f"max_devi_theta_crit = {max_devi_theta_crit}")
    print(f"avg_deviation = {avg_deviation}")
    plt.plot(b_a, theta_crit, alpha=0.7, color=color)
    return avg_deviation


def main() -> None:
    plot_models = [
        (model_name, legend_name)
        for model_name, legend_name in COMBINED_MODELS
        if model_name not in PLOT_EXCLUDED_MODELS
    ]
    plt.rcParams.update({"pdf.fonttype": 42})
    plt.style.use("seaborn-v0_8")
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [color_cycle[i % len(color_cycle)] for i in range(len(plot_models))]

    fig = plt.figure("Michell's Buckling Instability", figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"$\beta/\alpha$")
    ax.set_ylabel(r"$\theta^n$ (rad)")

    legend_labels = ["analytical"]
    b_a_ref: np.ndarray | None = None
    for (model_name, legend_name), color in zip(plot_models, colors):
        print(f"For {model_name}:")
        print("Loading MBI test...")
        result = load_result("mbi", model_name)
        valid = np.isfinite(result.theta_crit)
        if not np.any(valid):
            print(f"[{model_name}] No valid MBI points to plot.")
            continue
        b_a = result.b_a[valid]
        theta_crit = result.theta_crit[valid]
        if b_a_ref is None:
            b_a_ref = b_a.copy()
            theta_crit_ref = 2 * np.pi * np.sqrt(3) / b_a_ref
            ax.plot(
                b_a_ref,
                theta_crit_ref,
                c="k",
                linewidth="2",
                alpha=0.5,
                zorder=5,
            )
        mbi_plot(b_a=b_a, theta_crit=theta_crit, color=color)
        _ = mbi_avg_deviation(result)
        legend_labels.append(legend_name)

    ax.legend(legend_labels)
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    plt.savefig(FIG_PATH, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
