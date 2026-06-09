import os
import shutil
from typing import Optional, Tuple

import numpy as np

from adapteddlo_muj.envs.realgrav_valid_test import TestRopeEnv

DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)
STIFF_VALS_DIR = os.path.join(DATA_ROOT, "dlo_muj_real", "stiff_vals")
REALDATA_DIR = os.path.join(DATA_ROOT, "dlo_muj_real")

ROPE_LEN = 1.5
R_PIECES = 52
R_THICKNESS = 0.01

# Search bounds: stiff_scale = alpha * (2*pi)^3 for bending; beta/alpha for twisting.
LEGACY_STIFF_LIM = np.array([0.0, 2.0])
BACKEND_STIFF_LIM = np.array([0.0, 20.0])
BACKEND_MODELS = frozenset({"massspring", "cosserat", "cosserat2", "xpbd", "geds"})


def parse_lim_arg(lim_arg: Optional[str], default: np.ndarray) -> np.ndarray:
    if lim_arg is None or lim_arg.strip() == "":
        return default.copy()
    parts = [float(v.strip()) for v in lim_arg.split(",") if v.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected two comma-separated values, got: {lim_arg!r}")
    lo, hi = parts
    if lo >= hi:
        raise ValueError(f"Invalid search range [{lo}, {hi}]: lower must be < upper.")
    return np.array([lo, hi])


def search_limits(model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    if model_name in BACKEND_MODELS:
        lim = BACKEND_STIFF_LIM.copy()
    else:
        lim = LEGACY_STIFF_LIM.copy()
    return lim, lim.copy()


def wire_params(wire_color: str) -> Tuple[float, np.ndarray]:
    if wire_color == "white":
        rgba_vals = np.concatenate((np.array([300, 300, 300]) / 300, [1]))
        massperlen = 0.087 / 5.0
    elif wire_color == "black":
        rgba_vals = np.concatenate((np.array([0, 0, 0]) / 300, [1]))
        massperlen = 0.081 / 2.98
    elif wire_color == "red":
        rgba_vals = np.concatenate((np.array([300, 0, 0]) / 300, [1]))
        massperlen = 0.043 / 2.0
    else:
        raise ValueError(f"Unknown wire color: {wire_color}")
    return massperlen, rgba_vals


def twisting_params(wire_color: str) -> Tuple[float, np.ndarray]:
    if wire_color == "white":
        crittwist_all = np.array([970.0, 970.0, 980.0, 975.0, 980.0])
        b_a_arr = np.array([0.8291015625, 0.8291015625, 0.8232421875, 0.8232421875, 0.8232421875])
    elif wire_color == "black":
        crittwist_all = np.array([470.0, 500.0, 480.0, 470.0, 490.0])
        b_a_arr = np.array([1.5205078125, 1.3857421875, 1.4736328125, 1.3857421875, 1.4267578125])
    elif wire_color == "red":
        crittwist_all = np.array([430.0, 470.0, 445.0, 445.0, 460.0])
        b_a_arr = np.array([1.7724609375, 1.5263671875, 1.6669921875, 1.6669921875, 1.5791015625])
    else:
        raise ValueError(f"Unknown wire color: {wire_color}")
    return float(np.mean(crittwist_all)), b_a_arr


def bendstiff_path(wire_color: str, model_name: str, test_id: str) -> str:
    return os.path.join(
        STIFF_VALS_DIR,
        f"{wire_color}_{model_name}_{test_id}_bendstiff.pickle",
    )


def stiff_path(wire_color: str, model_name: str) -> str:
    return os.path.join(STIFF_VALS_DIR, f"{wire_color}_{model_name}_stiff.pickle")


def realdata_path(wire_color: str, test_id: str) -> str:
    return os.path.join(REALDATA_DIR, f"{wire_color}{test_id}_data.pickle")


def effective_rope_len(rope_len: float = ROPE_LEN, r_pieces: int = R_PIECES) -> float:
    return rope_len * r_pieces / (r_pieces - 2)


def mbi_pickle_path(model_name: str, grav_on: bool = True) -> str:
    grav_folder = "real/grav" if grav_on else "real/nograv"
    return os.path.join(
        DATA_ROOT,
        "mbi",
        grav_folder,
        model_name,
        "mbitest1.pickle",
    )


def ensure_mbi_pickle(model_name: str, grav_on: bool = True) -> str:
    picklename = mbi_pickle_path(model_name, grav_on=grav_on)
    if os.path.exists(picklename):
        return picklename
    fallback = mbi_pickle_path("adapt", grav_on=grav_on)
    if model_name != "adapt" and os.path.exists(fallback):
        os.makedirs(os.path.dirname(picklename), exist_ok=True)
        shutil.copy2(fallback, picklename)
        print(f"Bootstrapped MBI pickle for {model_name} from adapt: {picklename}")
        return picklename
    return picklename


def create_mbi_env(
    model_name: str,
    *,
    alpha_bar: float,
    beta_bar: float,
    rgba_vals: np.ndarray,
    massperlen: float,
    overall_rot: float = 0.0,
    rope_len: float = ROPE_LEN,
    grav_on: bool = True,
    do_render: bool = False,
    new_start: bool = False,
) -> TestRopeEnv:
    if not new_start:
        ensure_mbi_pickle(model_name, grav_on=grav_on)
    r_len = effective_rope_len(rope_len)
    r_mass = massperlen * r_len
    env = TestRopeEnv(
        overall_rot=overall_rot,
        do_render=do_render,
        r_pieces=R_PIECES,
        r_len=r_len,
        r_thickness=R_THICKNESS,
        test_type="mbi",
        alpha_bar=alpha_bar,
        beta_bar=beta_bar,
        r_mass=r_mass,
        new_start=new_start,
        stifftorqtype=model_name,
        grav_on=grav_on,
        rgba_vals=rgba_vals,
    )
    if do_render:
        env.set_viewer_details(
            dist=1.5,
            azi=90.0,
            elev=0.0,
            lookat=np.array([-0.81, 0.0, 0.15]),
        )
    return env


def sim_pos_error(env: TestRopeEnv, real_pos: np.ndarray, r_len: float = ROPE_LEN) -> float:
    sim_pos = env.observations["rope_pose"][1:-1].copy()[:, [0, 2]]
    sim_pos -= sim_pos[0]
    sim_pos *= -1.0
    return float(
        np.sum(np.linalg.norm(sim_pos - real_pos, axis=1)) / len(sim_pos) / r_len
    )
