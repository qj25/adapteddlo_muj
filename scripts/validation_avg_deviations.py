"""Print MBI and LHB average deviations for combined validation models."""

from __future__ import annotations

import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from validation_metrics import (
    COMBINED_MODELS,
    LHB_PIECES,
    LHB_S_MAX,
    LHB_S_MIN,
    format_lhb_latex_table,
    load_lhb_avg_deviations,
    load_lhb_piece_deviation,
    load_mbi_avg_deviations,
)


def _format_bracket(values: list[float]) -> str:
    inner = ", ".join(f"{value:.6g}" for value in values)
    return f"[{inner}]"


def main() -> None:
    model_names = [name for name, _ in COMBINED_MODELS]
    legend_names = [label for _, label in COMBINED_MODELS]

    print(f"Models (order): {model_names}")
    print(f"Legend labels: {legend_names}")
    print()

    mbi_values = load_mbi_avg_deviations(model_names)
    lhb_values = load_lhb_avg_deviations(model_names)

    for model_name, mbi_val, lhb_val in zip(model_names, mbi_values, lhb_values):
        print(f"[{model_name}] MBI avg_deviation = {mbi_val:.6g}")
        print(f"[{model_name}] LHB avg_deviation = {lhb_val:.6g}")

    print()
    print(f"MBI avg_deviation {_format_bracket(mbi_values)}")
    print(f"LHB avg_deviation {_format_bracket(lhb_values)}")
    print()
    print("LHB per-piece LaTeX table (native, adapt preset; massspring, cosserat5 computed):")
    for model_name in ("massspring", "cosserat5"):
        for r_pieces in LHB_PIECES:
            info = load_lhb_piece_deviation(model_name, r_pieces)
            if info is None or not info.is_low_confidence:
                continue
            print(
                f"[{model_name}] r_pieces={r_pieces}: using raw fallback "
                f"(trimmed {info.n_trimmed}/{info.n_total} samples in "
                f"({LHB_S_MIN}, {LHB_S_MAX}]; trimmed={info.trimmed_value:.6g}, raw={info.raw_value:.6g})"
            )
    print(format_lhb_latex_table())


if __name__ == "__main__":
    main()
