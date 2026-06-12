"""Compare sim vs reference across 8 poses (pts_all + pts_all2) for 5 models.

Prerequisite for poses 4-7:
  python scripts/tmp/gen_missing_all_svr2.py
Missing model pickle/JSON: warn and omit bar (NaN, not plotted).
"""
import json
import os
import pickle
import warnings
from datetime import datetime, timezone
import numpy as np

import adapteddlo_muj.utils.finddepth as fd1
from adapteddlo_muj.envs.simvreal_test.base import load_sim_pickle
from adapteddlo_muj.envs.test_shape_w_arm.base import (
    load_sim_case_json,
    simdata_json_path,
)
from adapteddlo_muj.envs.test_shape_w_arm.registry import (
    MODEL_REGISTRY,
    parse_models_arg,
)
from adapteddlo_muj.utils.argparse_utils import svr_parse
from adapteddlo_muj.utils.plotter import plot_bars_pose_panels

DEFAULT_MODELS = ["adapt", "jpqder", "native", "massspring", "cosserat5"]

parser = svr_parse()
_MODELS_HELP = ",".join(MODEL_REGISTRY.keys())
parser.add_argument(
    "--models",
    type=str,
    default=None,
    help=f"Comma-separated model names. Available: {_MODELS_HELP}. "
    f"Default: {','.join(DEFAULT_MODELS)}.",
)
args = parser.parse_args()

wc = args.wirecolor
mi = args.moveid
model_names = parse_models_arg(args.models, DEFAULT_MODELS)
unknown = [m for m in model_names if m not in MODEL_REGISTRY]
if unknown:
    raise ValueError(
        f"Unknown model(s): {unknown}. Available: {list(MODEL_REGISTRY.keys())}"
    )

wire_colors = ["black", "red", "white"]
move_ids = list(range(8))
n_wirecolors = len(wire_colors)
n_pos = len(move_ids)

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_data_dir = os.path.join(_repo_root, "adapteddlo_muj", "data")
_pts_all_path = os.path.join(_data_dir, "pts_all.pickle")
_pts_all2_path = os.path.join(_data_dir, "pts_all2.pickle")

with open(_pts_all_path, "rb") as f:
    pts_all = pickle.load(f)
with open(_pts_all2_path, "rb") as f:
    pts_all2 = pickle.load(f)
real_pos_arr_all = np.concatenate([pts_all, pts_all2], axis=1)
print("reference data loaded (pts_all + pts_all2)")

n_pieces = len(real_pos_arr_all[0][0]) - 1

json_dir = os.path.join(_data_dir, "simvreal_test")
os.makedirs(json_dir, exist_ok=True)

_simdata_plugin_dir = os.path.join(_data_dir, "simdata", "plugin")


def _plugin_picklename(wire_color: str, move_id: int, model_name: str) -> str:
    return os.path.join(
        _simdata_plugin_dir,
        f"simdata_{wire_color}{move_id}_{model_name}.pickle",
    )


def _has_pickle_case(wire_color: str, move_id: int, model_name: str) -> bool:
    return os.path.exists(_plugin_picklename(wire_color, move_id, model_name))


def _has_json_case(wire_color: str, move_id: int, model_name: str) -> bool:
    return os.path.exists(simdata_json_path(wire_color, move_id, model_name))


def _has_sim_case(wire_color: str, move_id: int, model_name: str) -> bool:
    return _has_pickle_case(wire_color, move_id, model_name) or _has_json_case(
        wire_color, move_id, model_name
    )


def _warn_missing_case(wire_color: str, move_id: int, model_name: str) -> None:
    pickle_path = _plugin_picklename(wire_color, move_id, model_name)
    json_path = simdata_json_path(wire_color, move_id, model_name)
    warnings.warn(
        f"Missing sim case for {model_name} {wire_color}{move_id} "
        f"(no {pickle_path} or {json_path}); bar omitted",
        stacklevel=2,
    )


def _align_reference_to_sim(
    sim_pts: np.ndarray,
    ref_pts: np.ndarray,
    move_id: int,
    r_len: float,
) -> np.ndarray:
    """Align reference points to sim frame before error computation.

    Poses 0-3: real camera coords (pts_all) — full line-standard alignment.
    Poses 4-7: synthetic sim coords (pts_all2) — origin shift only; both sides
    already share the same frame so adjust_linestandard / axis scaling would
    introduce artificial error.
    """
    ref_aligned = ref_pts.copy()
    sim_aligned = sim_pts.copy()
    ref_aligned -= ref_aligned[0]
    sim_aligned -= sim_aligned[0]
    if move_id < 4:
        sim_aligned[:, 0] *= -1.0
        sim_aligned[:, 1] *= -1.0
        se_axis = sim_aligned[-1] - sim_aligned[0]
        ref_aligned = fd1.adjust_linestandard(
            points_arr=ref_aligned,
            startend_axis=se_axis.copy(),
        )
        ref_aligned = fd1.optimize_through_axisscale(
            ref_aligned,
            se_axis,
            r_len,
        )
    return sim_aligned, ref_aligned


def load_case_for_move(wire_color: str, move_id: int, model_name: str):
    if _has_pickle_case(wire_color, move_id, model_name):
        return load_sim_pickle(
            wire_color=wire_color, pos_id=move_id, model_name=model_name
        )
    if _has_json_case(wire_color, move_id, model_name):
        return load_sim_case_json(wire_color, move_id, model_name)
    raise FileNotFoundError(
        f"Missing sim case for {wire_color}{move_id}_{model_name}"
    )


def run_single_model(model_name):
    node_pos_arr = np.zeros((n_wirecolors, n_pos, n_pieces + 1, 3))
    r_len = None
    for j in range(n_wirecolors):
        if wc is not None and wire_colors[j] != wc:
            continue
        for move_id in move_ids:
            if mi is not None and move_id != mi:
                continue
            if not _has_sim_case(wire_colors[j], move_id, model_name):
                _warn_missing_case(wire_colors[j], move_id, model_name)
                continue
            _, _, node_pos_arr_indiv, _ = load_case_for_move(
                wire_colors[j], move_id, model_name
            )
            if r_len is None:
                r_len = fd1.len_pts(node_pos_arr_indiv)
            node_pos_arr_indiv = fd1.split_lines2(node_pos_arr_indiv, n_pieces)
            node_pos_arr[j, move_id, :, :] = node_pos_arr_indiv[:, :].copy()

    real_pos_arr = real_pos_arr_all.copy()
    n_points = len(node_pos_arr[0][0])
    error_arr = np.full((n_wirecolors, n_pos), np.nan)
    rms_error_arr = np.full((n_wirecolors, n_pos), np.nan)

    print(f"Model: {model_name} | =======================================")
    for j in range(n_wirecolors):
        if wc is not None and wire_colors[j] != wc:
            continue
        print(f"wirecolor: {wire_colors[j]}")
        for move_id in move_ids:
            if mi is not None and move_id != mi:
                continue
            if not _has_sim_case(wire_colors[j], move_id, model_name):
                _warn_missing_case(wire_colors[j], move_id, model_name)
                continue
            sim_pts, ref_pts = _align_reference_to_sim(
                node_pos_arr[j, move_id],
                real_pos_arr[j, move_id],
                move_id,
                r_len,
            )
            norm_len = (
                fd1.len_pts(sim_pts)
                if move_id >= 4
                else r_len
            )

            diff = sim_pts - ref_pts
            dists = np.linalg.norm(diff, axis=1)
            error_arr[j, move_id] = np.sum(dists) / n_points / norm_len
            rms_error_arr[j, move_id] = np.sqrt(np.mean(dists**2)) / norm_len

    payload = {
        "model": model_name,
        "wire_colors": wire_colors,
        "pos_ids": move_ids,
        "mean_error": error_arr.tolist(),
        "rms_error": rms_error_arr.tolist(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = os.path.join(json_dir, f"simvreal2_{model_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {out_path}")
    return error_arr


all_errors = []
for model_name in model_names:
    all_errors.append(run_single_model(model_name))

if len(all_errors) > 1:
    plot_input = np.array(all_errors)[:, [2, 0, 1], :]
    plot_bars_pose_panels(plot_input, model_names=model_names, add_markers=True)
