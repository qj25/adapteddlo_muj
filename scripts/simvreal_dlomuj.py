import os
import pickle
import json
import numpy as np
from datetime import datetime, timezone
import adapteddlo_muj.utils.finddepth as fd1
from adapteddlo_muj.utils.plotter import plot_bars
from adapteddlo_muj.utils.argparse_utils import svr_parse
from adapteddlo_muj.envs.simvreal_test.registry import (
    DEFAULT_MODELS,
    get_model_specs,
    parse_models_arg,
)

parser = svr_parse()
parser.add_argument(
    "--models",
    type=str,
    default=None,
    help="Comma-separated model names to run (adapt,native,massspring).",
)
args = parser.parse_args()

wc = args.wirecolor
mi = args.moveid
model_names = parse_models_arg(args.models, DEFAULT_MODELS)
model_specs = get_model_specs(model_names)
wire_colors = ['black','red','white']
pos_type = ['0','1','2','3']
n_wirecolors = len(wire_colors)
n_pos = len(pos_type)

# init_qpos = np.array([
    # 7.40857786e-07,  6.38881793e-01,
    # 2.27269786e+00, -3.14160763e+00,
    # 1.34089981e+00,  3.14161037e+00
# ])
data_file = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "adapteddlo_muj/data/"
)
data_file = data_file + 'pts_all.pickle'

with open(data_file, 'rb') as f:
    real_pos_arr_all = pickle.load(f)
    print('real pos data loaded!')
n_pieces = len(real_pos_arr_all[0][0])-1

json_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "adapteddlo_muj/data/simvreal_test",
)
os.makedirs(json_dir, exist_ok=True)


def run_single_model(model_name, model_spec):
    node_pos_arr = np.zeros((n_wirecolors, n_pos, n_pieces + 1, 3))
    r_len = None
    for j in range(n_wirecolors):
        if wc is not None and wire_colors[j] != wc:
            continue
        for pos_id in range(n_pos):
            if mi is not None and pos_id != mi:
                continue
            _, _, node_pos_arr_indiv, _ = model_spec["load_case"](wire_colors[j], pos_id)
            if r_len is None:
                r_len = fd1.len_pts(node_pos_arr_indiv)
            node_pos_arr_indiv = fd1.split_lines2(node_pos_arr_indiv, n_pieces)
            node_pos_arr[j, pos_id, :, :] = node_pos_arr_indiv[:, :].copy()

    real_pos_arr = real_pos_arr_all.copy()
    n_points = len(node_pos_arr[0][0])
    error_arr = np.zeros((n_wirecolors, n_pos))
    rms_error_arr = np.zeros((n_wirecolors, n_pos))

    print(f"Model: {model_name} | =======================================")
    for j in range(n_wirecolors):
        if wc is not None and wire_colors[j] != wc:
            continue
        print(f"wirecolor: {wire_colors[j]}")
        for pos_id in range(n_pos):
            if mi is not None and pos_id != mi:
                continue
            node_pos_arr[j, pos_id] -= node_pos_arr[j, pos_id, 0]
            node_pos_arr[j, pos_id, :, 0] *= -1.0
            node_pos_arr[j, pos_id, :, 1] *= -1.0
            real_pos_arr[j, pos_id] -= real_pos_arr[j, pos_id, 0]
            se_axis = node_pos_arr[j, pos_id][-1] - node_pos_arr[j, pos_id][0]
            real_pos_arr[j, pos_id] = fd1.adjust_linestandard(
                points_arr=real_pos_arr[j, pos_id].copy(),
                startend_axis=se_axis.copy(),
            )
            real_pos_arr[j, pos_id] = fd1.optimize_through_axisscale(
                real_pos_arr[j, pos_id].copy(),
                se_axis,
                r_len,
            )

            diff = node_pos_arr[j, pos_id] - real_pos_arr[j, pos_id]
            dists = np.linalg.norm(diff, axis=1)
            error_arr[j, pos_id] = np.sum(dists) / n_points / r_len
            rms_error_arr[j, pos_id] = np.sqrt(np.mean(dists**2)) / r_len

    payload = {
        "model": model_name,
        "wire_colors": wire_colors,
        "pos_ids": [0, 1, 2, 3],
        "mean_error": error_arr.tolist(),
        "rms_error": rms_error_arr.tolist(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = os.path.join(json_dir, f"simvreal_{model_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {out_path}")
    return error_arr


all_errors = []
for model_name, model_spec in model_specs.items():
    all_errors.append(run_single_model(model_name, model_spec))

if len(all_errors) > 1:
    plot_input = np.array(all_errors)[:, [2, 0, 1]]
    plot_bars(plot_input, add_markers=True)