import os
import json
import numpy as np
from datetime import datetime, timezone

from adapteddlo_muj.envs.speed_test.base import build_result_payload
from adapteddlo_muj.envs.speed_test.registry import (
    DEFAULT_MODELS,
    MODEL_REGISTRY,
    get_model_specs,
    parse_models_arg,
)
from adapteddlo_muj.utils.argparse_utils import spdt_parse
from adapteddlo_muj.utils.plotter import plot_computetime

#======================| Settings |======================
parser = spdt_parse()
_MODELS_HELP = ",".join(MODEL_REGISTRY.keys())
parser.add_argument(
    "--models",
    type=str,
    default=None,
    help=f"Comma-separated model names to run. Available: {_MODELS_HELP}. "
    f"Default: {','.join(DEFAULT_MODELS)}.",
)
args = parser.parse_args()

new_start = bool(args.newstart)
model_names = parse_models_arg(args.models, DEFAULT_MODELS)
model_specs = get_model_specs(model_names)

test_type = 'speedtest2'

r_len = 9.29
r_thickness = 0.03
alpha_val = 1.345
beta_val = 0.789
if test_type == 'speedtest1':
    r_pieces_list = [20,30,40,50,60]
else:
    r_pieces_list = [40, 60, 80, 110, 140, 180]
    # r_pieces_list = [80]
# r_pieces_list = [80]
#======================| End Settings |======================
data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "adapteddlo_muj/data/speed_test"
)
os.makedirs(data_dir, exist_ok=True)


def output_path(model_name):
    return os.path.join(data_dir, f"{test_type}_{model_name}.json")


if new_start:
    settings = {
        "test_type": test_type,
        "r_len": r_len,
        "r_thickness": r_thickness,
        "alpha_val": alpha_val,
        "beta_val": beta_val,
    }
    for model_name, model_spec in model_specs.items():
        t_list = np.zeros((len(r_pieces_list),))
        for i, r_pieces in enumerate(r_pieces_list):
            print(f"[{model_name}] Testing speed for {r_pieces} pieces.. ..")
            run_cfg = dict(settings)
            run_cfg["r_pieces"] = r_pieces
            t_list[i] = model_spec["run"](run_cfg)

        payload = build_result_payload(
            model=model_name,
            test_type=test_type,
            r_pieces_list=[int(v) for v in r_pieces_list],
            times=t_list.tolist(),
            settings=settings,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with open(output_path(model_name), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved: {output_path(model_name)}")

else:
    series = []
    for model_name in model_names:
        model_path = output_path(model_name)
        with open(model_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not series:
            r_pieces_list = payload["r_pieces_list"]
        series.append(np.array(payload["times"], dtype=float))
        print(f"Loaded: {model_path}")
    plot_computetime(r_pieces_list, series)