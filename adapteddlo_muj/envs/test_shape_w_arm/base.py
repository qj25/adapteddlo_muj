import json
import os
import pickle
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import adapteddlo_muj.utils.transform_utils as T
from adapteddlo_muj.utils.manipulation_config import (
    apply_move_settings,
    load_manipulation_config,
)
from adapteddlo_muj.envs.real2sim_paramiden.base import stiff_path
from adapteddlo_muj.envs.rnrvalid2 import ValidRnR2Env
from adapteddlo_muj.envs.rnrvalid3_plugin import ValidRnR3Env
from adapteddlo_muj.utils.wire_plugin import COSSERAT_WIRE_PLUGIN_CONFIGS

PIECE_MULTI = 5
R_LEN = 0.40
R_PIECES = PIECE_MULTI * 10
R_THICKNESS = 0.006

DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)
TEST_SHAPE_DATA_DIR = os.path.join(DATA_ROOT, "test_shape_w_arm")
LEGACY_SIMDATA_DIRS = (
    os.path.join(DATA_ROOT, "simdata", "plugin"),
    os.path.join(DATA_ROOT, "simdata", "normal"),
    os.path.join(DATA_ROOT, "simdata"),
)
LEGACY_MODEL_ALIASES = {"adapt2": "jpqder"}
SIMDATA_PICKLE_RE = re.compile(
    r"^simdata_(?P<wire>black|red|white)(?P<moveid>\d+)_(?P<model>.+)\.pickle$"
)


def load_stiffness(wire_color: str, model_name: str) -> Tuple[float, float]:
    picklename = stiff_path(wire_color, model_name)
    if not os.path.exists(picklename):
        if model_name != "adapt":
            adapt_picklename = stiff_path(wire_color, "adapt")
            if os.path.exists(adapt_picklename):
                print(
                    f"[{wire_color}/{model_name}] stiffness not found at {picklename}, "
                    f"using adapt: {adapt_picklename}"
                )
                picklename = adapt_picklename
            else:
                raise FileNotFoundError(
                    f"Stiffness file not found for {model_name} or adapt: "
                    f"{picklename}, {adapt_picklename}"
                )
        else:
            raise FileNotFoundError(f"Stiffness file not found: {picklename}")
    with open(picklename, "rb") as f:
        alpha_glob, b_a_glob = pickle.load(f)
    beta_glob = b_a_glob * alpha_glob
    print(f"[{wire_color}/{model_name}] alpha_glob = {alpha_glob}")
    print(f"[{wire_color}/{model_name}] beta_glob = {beta_glob}")
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
    model_name: Optional[str] = None,
) -> ValidRnR2Env:
    model_name = model_name or rope_type
    alpha_glob, beta_glob = load_stiffness(wire_color, model_name)
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


def create_wire_plugin_env(
    wire_color: str,
    overall_rot,
    do_render: bool,
    model_name: str,
    extra_plugin_configs: Optional[Dict[str, str]] = None,
) -> ValidRnR3Env:
    alpha_glob, beta_glob = load_stiffness(wire_color, model_name)
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
        config_model_name=model_name,
        rgba_vals=rgba_vals,
        do_render=do_render,
        extra_plugin_configs=extra_plugin_configs,
    )


def create_jpqder_env(
    wire_color: str,
    overall_rot,
    do_render: bool,
    model_name: Optional[str] = None,
) -> ValidRnR3Env:
    model_name = model_name or "jpqder"
    return create_wire_plugin_env(
        wire_color, overall_rot, do_render, model_name=model_name
    )


def create_cosserat_env(
    wire_color: str,
    overall_rot,
    do_render: bool,
    model_name: Optional[str] = None,
) -> ValidRnR3Env:
    model_name = model_name or "cosserat"
    return create_wire_plugin_env(
        wire_color,
        overall_rot,
        do_render,
        model_name=model_name,
        extra_plugin_configs=COSSERAT_WIRE_PLUGIN_CONFIGS,
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
    apply_move_settings(env, "first_move_to_pose")
    env.move_to_pose(desired_pos, desired_quat)
    if not getting_jointpos:
        env.rot_x_rads(z_rot[pos_id])
        apply_move_settings(env, "second_move_to_pose")
        env.move_to_pose(desired_pos, desired_quat)
        print("holding_pos")
        env.hold_pos(load_manipulation_config()["hold_time_after_pose_s"])
        print("held_pos")
    joint_pos = env._jd.copy()
    nodes_pos = env.observations["rope_pose"]
    return joint_pos, nodes_pos


def simdata_path(wire_color: str, pos_id: int, model_name: str, use_plugin: bool) -> str:
    subdir = "plugin" if use_plugin else ""
    return os.path.join(DATA_ROOT, "simdata", subdir, f"simdata_{wire_color}{pos_id}_{model_name}.pickle")


def simdata_json_path(wire_color: str, move_id: int, model_name: str) -> str:
    return os.path.join(
        TEST_SHAPE_DATA_DIR,
        f"sim_{wire_color}{move_id}_{model_name}.json",
    )


def save_sim_case_json(
    wire_color: str,
    move_id: int,
    model_name: str,
    init_qpos: np.ndarray,
    z_rot_rad: float,
    joint_pos: np.ndarray,
    nodes_pos: np.ndarray,
    created_at: Optional[str] = None,
    source_pickle: Optional[str] = None,
) -> str:
    os.makedirs(TEST_SHAPE_DATA_DIR, exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model_name,
        "wire_color": wire_color,
        "moveid": int(move_id),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "init_qpos": np.asarray(init_qpos, dtype=float).tolist(),
        "z_rot_rad": float(z_rot_rad),
        "joint_pos": np.asarray(joint_pos, dtype=float).tolist(),
        "nodes_pos": np.asarray(nodes_pos, dtype=float).tolist(),
        "r_pieces": int(R_PIECES),
        "r_len": float(R_LEN),
    }
    if source_pickle is not None:
        payload["source_pickle"] = source_pickle
    out_path = simdata_json_path(wire_color, move_id, model_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def canonical_model_name(legacy_model: str) -> str:
    return LEGACY_MODEL_ALIASES.get(legacy_model, legacy_model)


def parse_simdata_pickle_name(filename: str) -> Tuple[str, int, str]:
    match = SIMDATA_PICKLE_RE.match(os.path.basename(filename))
    if match is None:
        raise ValueError(f"Unrecognized legacy simdata pickle name: {filename}")
    legacy_model = match.group("model")
    return (
        match.group("wire"),
        int(match.group("moveid")),
        canonical_model_name(legacy_model),
    )


def load_legacy_sim_pickle(pickle_path: str) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, (list, tuple)) or len(data) != 4:
        raise ValueError(f"Expected legacy pickle list of length 4 in {pickle_path}")
    init_qpos, z_rot_rad, nodes_pos, joint_pos = data
    return (
        np.asarray(init_qpos, dtype=float),
        float(z_rot_rad),
        np.asarray(nodes_pos, dtype=float),
        np.asarray(joint_pos, dtype=float),
    )


def iter_legacy_simdata_pickles() -> List[str]:
    seen = set()
    pickle_paths: List[str] = []
    for root in LEGACY_SIMDATA_DIRS:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if not name.endswith(".pickle") or not name.startswith("simdata_"):
                continue
            path = os.path.join(root, name)
            if path in seen:
                continue
            seen.add(path)
            pickle_paths.append(path)
    return pickle_paths


def convert_legacy_pickle_to_json(
    pickle_path: str,
    skip_existing: bool = False,
) -> Optional[str]:
    wire_color, move_id, model_name = parse_simdata_pickle_name(pickle_path)
    out_path = simdata_json_path(wire_color, move_id, model_name)
    if skip_existing and os.path.exists(out_path):
        return None
    init_qpos, z_rot_rad, nodes_pos, joint_pos = load_legacy_sim_pickle(pickle_path)
    mtime = os.path.getmtime(pickle_path)
    created_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    rel_source = os.path.relpath(pickle_path, DATA_ROOT)
    return save_sim_case_json(
        wire_color=wire_color,
        move_id=move_id,
        model_name=model_name,
        init_qpos=init_qpos,
        z_rot_rad=z_rot_rad,
        joint_pos=joint_pos,
        nodes_pos=nodes_pos,
        created_at=created_at,
        source_pickle=rel_source,
    )


def convert_all_legacy_simdata_pickles(skip_existing: bool = False) -> List[str]:
    ordered_paths: List[str] = []
    seen_keys = set()
    for root in LEGACY_SIMDATA_DIRS:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if not name.endswith(".pickle") or not name.startswith("simdata_"):
                continue
            path = os.path.join(root, name)
            try:
                key = parse_simdata_pickle_name(path)
            except ValueError:
                continue
            if key in seen_keys:
                continue
            seen_keys.add(key)
            ordered_paths.append(path)

    saved_paths: List[str] = []
    for pickle_path in ordered_paths:
        out_path = convert_legacy_pickle_to_json(pickle_path, skip_existing=skip_existing)
        if out_path is not None:
            saved_paths.append(out_path)
    return saved_paths
