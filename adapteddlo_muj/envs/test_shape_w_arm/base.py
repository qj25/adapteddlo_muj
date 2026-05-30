import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np

import adapteddlo_muj.utils.transform_utils as T
from adapteddlo_muj.envs.rnrvalid2 import ValidRnR2Env
from adapteddlo_muj.envs.rnrvalid3_plugin import ValidRnR3Env

PIECE_MULTI = 5
R_LEN = 0.40
R_PIECES = PIECE_MULTI * 10
R_THICKNESS = 0.006

DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)


def stiff_pickle_path(wire_color: str, stiff_key: str) -> str:
    return os.path.join(DATA_ROOT, "dlo_muj_real", "stiff_vals", f"{wire_color}_{stiff_key}_stiff.pickle")


def load_stiffness(wire_color: str, stiff_key: str) -> Tuple[float, float]:
    picklename = stiff_pickle_path(wire_color, stiff_key)
    if not os.path.exists(picklename) and stiff_key == "massspring":
        picklename = stiff_pickle_path(wire_color, "adapt")
    with open(picklename, "rb") as f:
        alpha_glob, b_a_glob = pickle.load(f)
    beta_glob = b_a_glob * alpha_glob
    print(f"alpha_glob = {alpha_glob}")
    print(f"beta_glob = {beta_glob}")
    return alpha_glob, beta_glob


def wire_params(wire_color: str) -> Tuple[float, np.ndarray]:
    if wire_color == "white":
        massperlen = 0.087 / 5.0
        rgba_vals = np.concatenate((np.array([300, 300, 300]) / 300, [1]))
    elif wire_color == "black":
        rgba_vals = np.concatenate((np.array([0, 0, 0]) / 300, [1]))
        massperlen = 0.079 / 2.98
    elif wire_color == "red":
        rgba_vals = np.concatenate((np.array([300, 0, 0]) / 300, [1]))
        massperlen = 0.043 / 2.0
    else:
        raise ValueError(f"Unknown wire color: {wire_color}")
    return massperlen, rgba_vals


def create_rnr2_env(
    wire_color: str,
    rope_type: str,
    overall_rot,
    do_render: bool,
    stiff_key: Optional[str] = None,
) -> ValidRnR2Env:
    stiff_key = stiff_key or rope_type
    alpha_glob, beta_glob = load_stiffness(wire_color, stiff_key)
    massperlen, rgba_vals = wire_params(wire_color)
    return ValidRnR2Env(
        alpha_bar=alpha_glob,
        beta_bar=beta_glob,
        r_len=R_LEN,
        r_mass=R_LEN * massperlen,
        r_thickness=R_THICKNESS,
        r_pieces=R_PIECES,
        overall_rot=overall_rot,
        rope_type=rope_type,
        rgba_vals=rgba_vals,
        do_render=do_render,
    )


def create_jpqder_env(wire_color: str, overall_rot, do_render: bool) -> ValidRnR3Env:
    alpha_glob, beta_glob = load_stiffness(wire_color, "adapt")
    massperlen, rgba_vals = wire_params(wire_color)
    return ValidRnR3Env(
        alpha_bar=alpha_glob,
        beta_bar=beta_glob,
        r_len=R_LEN,
        r_mass=R_LEN * massperlen,
        r_thickness=R_THICKNESS,
        r_pieces=R_PIECES,
        overall_rot=overall_rot,
        plugin_name="wire",
        rgba_vals=rgba_vals,
        do_render=do_render,
    )


def run_manipulation(
    env,
    move_pos: np.ndarray,
    move_quat: np.ndarray,
    z_rot: np.ndarray,
    pos_id: int,
    getting_jointpos: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    desired_pos = env.init_pos + move_pos[pos_id]
    desired_quat = T.quat_multiply(env.init_quat, move_quat[pos_id])
    env.max_action = 0.02
    env.move_to_pose(desired_pos, desired_quat)
    if not getting_jointpos:
        env.rot_x_rads(z_rot[pos_id])
        env.move_to_pose(desired_pos, desired_quat)
        print("holding_pos")
        env.hold_pos(10.0)
        print("held_pos")
    joint_pos = env._jd.copy()
    nodes_pos = env.observations["rope_pose"]
    return joint_pos, nodes_pos


def simdata_path(wire_color: str, pos_id: int, model_name: str, use_plugin: bool) -> str:
    subdir = "plugin" if use_plugin else ""
    return os.path.join(DATA_ROOT, "simdata", subdir, f"simdata_{wire_color}{pos_id}_{model_name}.pickle")
