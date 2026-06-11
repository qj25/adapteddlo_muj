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
BACKEND_STIFF_LIM = np.array([0.0, 2.0])
MASSSPRING_B_A_LIM = np.array([0.0, 2.0])
BACKEND_MODELS = frozenset({"massspring", "cosserat", "xpbd", "geds"})

# Models backed by adapteddlo_muj/controllers/*_cpp (use manual_rot in MBI circle test).
CPP_BACKEND_MODELS = frozenset({
    "adapt",
    "xfrc",
    "massspring",
    "cosserat",
    "cosserat3",
    "cosserat5",
    "xpbd",
    "geds",
})

# Models that reuse another model's real/grav MBI pickle (TestRopeEnv only).
REALGRAV_MBI_PICKLE_ALIASES = {}

# Plugin-based models use adapt/plgn/<plugin>/mbitest1.pickle (TestPluginEnv).
PLUGIN_MBI_MODELS = frozenset({"jpqder"})


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
        stiff_lim = BACKEND_STIFF_LIM.copy()
    else:
        stiff_lim = LEGACY_STIFF_LIM.copy()
    if model_name == "massspring":
        b_a_lim = MASSSPRING_B_A_LIM.copy()
    else:
        b_a_lim = stiff_lim.copy()
    return stiff_lim, b_a_lim


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


def mbi_pickle_source(model_name: str) -> str:
    return REALGRAV_MBI_PICKLE_ALIASES.get(model_name, model_name)


def mbi_pickle_path(model_name: str, grav_on: bool = True) -> str:
    if model_name in PLUGIN_MBI_MODELS:
        return os.path.join(
            DATA_ROOT,
            "mbi",
            "adapt",
            "plgn",
            "wire",
            "mbitest1.pickle",
        )
    source = mbi_pickle_source(model_name)
    grav_folder = "real/grav" if grav_on else "real/nograv"
    return os.path.join(
        DATA_ROOT,
        "mbi",
        grav_folder,
        source,
        "mbitest1.pickle",
    )


def ensure_mbi_pickle(model_name: str, grav_on: bool = True) -> str:
    picklename = mbi_pickle_path(model_name, grav_on=grav_on)
    if os.path.exists(picklename):
        return picklename
    if model_name in PLUGIN_MBI_MODELS:
        return picklename
    fallback = mbi_pickle_path("adapt", grav_on=grav_on)
    if mbi_pickle_source(model_name) != "adapt" and os.path.exists(fallback):
        os.makedirs(os.path.dirname(picklename), exist_ok=True)
        shutil.copy2(fallback, picklename)
        print(
            f"Bootstrapped MBI pickle for {mbi_pickle_source(model_name)} "
            f"from adapt: {picklename}"
        )
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
    if model_name == "jpqder":
        from adapteddlo_muj.envs.validitytest_env import TestPluginEnv

        env = TestPluginEnv(
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
            plugin_name="wire",
        )
    else:
        env = TestRopeEnv(
            overall_rot=overall_rot,
            manual_rot=model_name in CPP_BACKEND_MODELS,
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
