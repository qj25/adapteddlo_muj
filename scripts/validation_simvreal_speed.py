"""Print sim-v-real position error and speed metrics for combined validation models."""

from __future__ import annotations

import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from validation_metrics import COMBINED_MODELS
from validation_simvreal_speed_metrics import (
    format_simvreal_speed_latex_equations,
    format_simvreal_speed_latex_table,
    load_simvreal_avg_errors,
    load_simvreal_inverse_scaled_position_errors,
    load_speed_inverse_avg_percent_increases,
)


def _format_bracket(values: list[float]) -> str:
    inner = ", ".join(f"{value:.8f}" for value in values)
    return f"[{inner}]"


def main() -> None:
    model_names = [name for name, _ in COMBINED_MODELS]
    legend_names = [label for _, label in COMBINED_MODELS]

    print(f"Models (order): {model_names}")
    print(f"Legend labels: {legend_names}")
    print()

    pos_errors = load_simvreal_avg_errors(model_names)
    inv_pos_errors = load_simvreal_inverse_scaled_position_errors(model_names)
    inv_speeds = load_speed_inverse_avg_percent_increases(model_names)

    for model_name, pos_err, inv_pos, inv_speed in zip(
        model_names, pos_errors, inv_pos_errors, inv_speeds
    ):
        print(f"[{model_name}] simvreal_avg_norm_pos_error = {pos_err:.8f}")
        print(f"[{model_name}] inv_scaled_pos_error = {inv_pos:.8f}")
        print(f"[{model_name}] inv_avg_speed_pct_increase = {inv_speed:.8f}")
        print()

    print(f"simvreal_avg_norm_pos_error {_format_bracket(pos_errors)}")
    print(f"inv_scaled_pos_error {_format_bracket(inv_pos_errors)}")
    print(f"inv_avg_speed_pct_increase {_format_bracket(inv_speeds)}")
    print()
    print("LaTeX equations:")
    print(format_simvreal_speed_latex_equations())
    print()
    print("LaTeX table:")
    print(
        format_simvreal_speed_latex_table(
            model_names=model_names,
            legend_names=legend_names,
            inv_pos_errors=inv_pos_errors,
            inv_speeds=inv_speeds,
        )
    )


if __name__ == "__main__":
    main()
