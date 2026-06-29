"""Shared loading and metrics for sim-v-real position error and speed tests."""

from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np

from validation_metrics import COMBINED_MODELS, REPO_ROOT

SIMVREAL_DIR = os.path.join(REPO_ROOT, "adapteddlo_muj", "data", "simvreal_test")
SPEED_TEST_DIR = os.path.join(REPO_ROOT, "adapteddlo_muj", "data", "speed_test")
SPEED_TEST_TYPE = "speedtest2"
SPEED_BASELINE_MODEL = "plain"
N_POSES = 8
POS_ERROR_SCALE = 10.0


def _simvreal_result_path(model_name: str) -> str:
    return os.path.join(SIMVREAL_DIR, f"simvreal2_{model_name}.json")


def _speed_result_path(model_name: str) -> str:
    return os.path.join(SPEED_TEST_DIR, f"{SPEED_TEST_TYPE}_{model_name}.json")


def load_simvreal_payload(model_name: str) -> dict:
    path = _simvreal_result_path(model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No sim-v-real results for {model_name}. "
            f"Run scripts/simvreal_dlomuj2.py first. Expected: {path}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_speed_payload(model_name: str) -> dict:
    path = _speed_result_path(model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No speed-test results for {model_name}. "
            f"Run scripts/speed_test.py --newstart 1 first. Expected: {path}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def simvreal_normalized_position_error(payload: dict) -> float:
    """Average normalized position error across the 8 poses.

    For each pose, wire-color errors are averaged first; then the mean is taken
    over all 8 poses. This matches the per-pose ``mean_error`` values written by
    ``scripts/simvreal_dlomuj2.py``.
    """
    mean_error = np.asarray(payload["mean_error"], dtype=float)
    if mean_error.ndim != 2:
        raise ValueError(
            f"[{payload.get('model', '?')}] Expected 2D mean_error, got shape {mean_error.shape}."
        )
    n_poses = mean_error.shape[1]
    if n_poses != N_POSES:
        raise ValueError(
            f"[{payload.get('model', '?')}] Expected {N_POSES} poses, got {n_poses}."
        )
    per_pose = np.nanmean(mean_error, axis=0)
    if not np.any(np.isfinite(per_pose)):
        raise ValueError(
            f"[{payload.get('model', '?')}] No valid normalized position errors."
        )
    return float(np.nanmean(per_pose))


def simvreal_inverse_scaled_position_error(avg_error: float) -> float:
    """Inverse of the average normalized position error scaled by 10."""
    scaled = POS_ERROR_SCALE * avg_error
    if np.isclose(scaled, 0.0):
        raise ValueError("Scaled average position error is zero; inverse is undefined.")
    return float(1.0 / scaled)


def speed_percent_increase(
    model_times: np.ndarray,
    plain_times: np.ndarray,
) -> np.ndarray:
    """Percentage increase in computational time relative to plain."""
    plain_arr = np.asarray(plain_times, dtype=float)
    model_arr = np.asarray(model_times, dtype=float)
    if plain_arr.shape != model_arr.shape:
        raise ValueError(
            "Model and plain speed-test piece counts do not match: "
            f"{plain_arr.shape} vs {model_arr.shape}."
        )
    ref_safe = np.where(np.abs(plain_arr) < 1e-12, 1e-12, plain_arr)
    return (model_arr - plain_arr) / ref_safe * 100.0


def speed_avg_percent_increase(
    model_times: np.ndarray,
    plain_times: np.ndarray,
) -> float:
    """Mean percentage time increase over all discrete piece counts."""
    pct = speed_percent_increase(model_times, plain_times)
    if pct.size == 0:
        raise ValueError("No speed-test piece counts available.")
    return float(np.mean(pct))


def speed_inverse_avg_percent_increase(
    model_times: np.ndarray,
    plain_times: np.ndarray,
) -> float:
    """Reciprocal of the mean percentage time increase vs plain."""
    avg_pct = speed_avg_percent_increase(model_times, plain_times)
    if np.isclose(avg_pct, 0.0):
        raise ValueError("Average percentage increase is zero; inverse is undefined.")
    return float(1.0 / avg_pct)


def load_simvreal_avg_errors(
    model_names: Optional[List[str]] = None,
) -> List[float]:
    names = model_names or [name for name, _ in COMBINED_MODELS]
    return [
        simvreal_normalized_position_error(load_simvreal_payload(model_name))
        for model_name in names
    ]


def load_simvreal_inverse_scaled_position_errors(
    model_names: Optional[List[str]] = None,
) -> List[float]:
    names = model_names or [name for name, _ in COMBINED_MODELS]
    return [
        simvreal_inverse_scaled_position_error(
            simvreal_normalized_position_error(load_simvreal_payload(model_name))
        )
        for model_name in names
    ]


def load_speed_inverse_avg_percent_increases(
    model_names: Optional[List[str]] = None,
    *,
    baseline_model: str = SPEED_BASELINE_MODEL,
) -> List[float]:
    names = model_names or [name for name, _ in COMBINED_MODELS]
    plain_payload = load_speed_payload(baseline_model)
    plain_times = np.asarray(plain_payload["times"], dtype=float)
    values: List[float] = []
    for model_name in names:
        payload = load_speed_payload(model_name)
        model_times = np.asarray(payload["times"], dtype=float)
        if payload["r_pieces_list"] != plain_payload["r_pieces_list"]:
            raise ValueError(
                f"[{model_name}] r_pieces_list does not match plain baseline."
            )
        values.append(speed_inverse_avg_percent_increase(model_times, plain_times))
    return values


def format_metric_value(val: float) -> str:
    text = f"{val:.8f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text


def format_simvreal_speed_latex_equations() -> str:
    return r"""\paragraph{Average normalized position error (sim-v-real, 8 poses).}
\begin{equation}
  \bar{e}_m = \frac{1}{P} \sum_{p=1}^{P} \left(
    \frac{1}{|W_p|} \sum_{w \in W_p} e_{m,w,p}
  \right)
\end{equation}
\begin{equation}
  e_{m,w,p} = \frac{1}{N_{w,p}} \sum_{i=1}^{N_{w,p}}
    \frac{\left\lVert \mathbf{s}_{m,w,p,i} - \mathbf{r}_{m,w,p,i}^{*} \right\rVert}{L_{m,w,p}}
\end{equation}
\begin{tabular}{cl}
  \hline
  Symbol & Meaning \\
  \hline
  $m$ & Model index \\
  $w$ & Wire color (black, red, white) \\
  $p$ & Pose index, $p \in \{1,\ldots,P\}$, $P=8$ \\
  $W_p$ & Wire colors with valid simulation data for pose $p$ \\
  $i$ & Discretized node index along the rope \\
  $N_{w,p}$ & Number of nodes for wire color $w$ and pose $p$ \\
  $\mathbf{s}_{m,w,p,i}$ & Simulated node position \\
  $\mathbf{r}_{m,w,p,i}^{*}$ & Reference position aligned to the simulation frame
    (line-standard for poses $0$--$3$, origin shift only for poses $4$--$7$) \\
  $L_{m,w,p}$ & Normalization length (rope length from pose $0$ for $p<4$;
    simulated rope length for $p \ge 4$) \\
  $e_{m,w,p}$ & Per-case normalized mean position error for model $m$,
    wire color $w$, and pose $p$ \\
  \hline
\end{tabular}
\begin{equation}
  \mathrm{InvPos}_m = \frac{1}{10\,\bar{e}_m}
\end{equation}
\begin{tabular}{cl}
  \hline
  Symbol & Meaning \\
  \hline
  $\mathrm{InvPos}_m$ & Inverse of the average normalized position error,
    scaled by a factor of $10$ \\
  \hline
\end{tabular}

\paragraph{Inverse of average percentage increase in computational time vs.\ plain.}
\begin{equation}
  \mathrm{InvSpeed}_m = \frac{1}{\overline{\Delta T}_m},
  \qquad
  \overline{\Delta T}_m = \frac{1}{K} \sum_{k=1}^{K} \Delta T_{m,k}
\end{equation}
\begin{equation}
  \Delta T_{m,k} = 100 \cdot \frac{t_{m,k} - t_{\mathrm{plain},k}}{t_{\mathrm{plain},k}}
\end{equation}
\begin{tabular}{cl}
  \hline
  Symbol & Meaning \\
  \hline
  $m$ & Model index \\
  $k$ & Discrete piece-count setting index \\
  $K$ & Number of piece-count settings ($K=6$) \\
  $r_k$ & Number of discrete rope pieces for setting $k$ \\
  $t_{m,k}$ & Wall-clock time to simulate $1$\,s of physics for model $m$
    at piece count $r_k$ \\
  $t_{\mathrm{plain},k}$ & Same timing for the plain baseline model \\
  $\Delta T_{m,k}$ & Percentage increase in computational time relative to plain \\
  $\overline{\Delta T}_m$ & Arithmetic mean of $\Delta T_{m,k}$ over all $K$ piece counts \\
  $\mathrm{InvSpeed}_m$ & Reciprocal of $\overline{\Delta T}_m$ \\
  \hline
\end{tabular}"""


def format_simvreal_speed_latex_table(
    model_names: Optional[List[str]] = None,
    legend_names: Optional[List[str]] = None,
    inv_pos_errors: Optional[List[float]] = None,
    inv_speeds: Optional[List[float]] = None,
    *,
    indent: str = "\t\t\t",
) -> str:
    names = model_names or [name for name, _ in COMBINED_MODELS]
    labels = legend_names or [label for _, label in COMBINED_MODELS]
    pos_scores = (
        inv_pos_errors
        if inv_pos_errors is not None
        else load_simvreal_inverse_scaled_position_errors(names)
    )
    speeds = (
        inv_speeds
        if inv_speeds is not None
        else load_speed_inverse_avg_percent_increases(names)
    )
    lines = [
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Model & $\mathrm{InvPos}_m$ & $\mathrm{InvSpeed}_m$ \\",
        r"\hline",
    ]
    for idx, (label, inv_pos, inv_speed) in enumerate(zip(labels, pos_scores, speeds)):
        prefix = "" if idx == 0 else indent
        suffix = " \\\\" if idx == len(labels) - 1 else " \\\\ \\hline"
        lines.append(
            f"{prefix}{label} & {format_metric_value(inv_pos)} & "
            f"{format_metric_value(inv_speed)}{suffix}"
        )
    lines.append(r"\end{tabular}")
    return "\n".join(lines)
