#!/usr/bin/env python3
"""Validate stability fixes through abbreviated/full moveid 2."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import adapteddlo_muj.utils.transform_utils as T
from adapteddlo_muj.envs.test_shape_w_arm.base import run_manipulation
from adapteddlo_muj.envs.test_shape_w_arm.registry import get_model_specs

MOVE_POS = np.array([[0.165, 0.1, 0.1]])
MOVE_AA = np.array([[50.0, 10.0, 0.0]]) * np.pi / 180
Z_ROT = np.array([360.0]) * np.pi / 180
MOVE_QUAT = np.array([T.axisangle2quat(MOVE_AA[0])])
FAIL_QVEL = 100.0


def patch_xpbd_k_torque(env, k_torque=0.001):
    dlo = env.dlo_sim
    dlo.k_torque = k_torque
    dlo.xpbd_math.setTorqueGain(k_torque)


def patch_geds_torque_limit(env, t_limit=1.0):
    dlo = env.dlo_sim
    orig = dlo._forces_to_torques

    def limited():
        orig()
        for i in range(len(dlo.torq_node)):
            mag = np.linalg.norm(dlo.torq_node[i])
            if mag > t_limit:
                dlo.torq_node[i] *= t_limit / mag

    dlo._forces_to_torques = limited


def patch_geds_lower_stiff(env, scale=0.5):
    dlo = env.dlo_sim
    dlo.youngs_modulus *= scale
    dlo.torsion_modulus *= scale
    dlo.geds_math.setMaterial(dlo.youngs_modulus, dlo.torsion_modulus)


def run_with_monitor(model_name, patch_fn=None, label=None):
    label = label or model_name
    print(f"\n=== {label} ===")
    env = get_model_specs([model_name])[model_name]["create_env"]("white", None, False)
    env.max_action = 0.02
    if patch_fn:
        patch_fn(env)
    fail_step = None
    orig_step = env.step

    def monitored_step(action=np.zeros(6)):
        nonlocal fail_step
        obs, r, done, trunc, info = orig_step(action)
        qv = float(np.max(np.abs(env.data.qvel[env.joint_qveladdr_full])))
        if fail_step is None and (qv > FAIL_QVEL or np.any(np.isnan(env.data.qpos))):
            fail_step = env.env_steps
            print(f"  FAIL at step {fail_step} qvel={qv:.4g}")
        return obs, r, done, trunc, info

    env.step = monitored_step
    try:
        run_manipulation(env, MOVE_POS, MOVE_QUAT, Z_ROT, 0, getting_jointpos=False)
        status = "OK" if fail_step is None else f"FAIL@{fail_step}"
        print(f"  finished steps={env.env_steps} status={status}")
    except Exception as exc:
        print(f"  EXCEPTION at step {env.env_steps}: {exc}")
    return fail_step


def main():
    run_with_monitor("adapt")
    run_with_monitor("xpbd", label="xpbd baseline")
    run_with_monitor("xpbd", lambda e: patch_xpbd_k_torque(e, 0.001), label="xpbd k_torque=0.001")
    run_with_monitor("xpbd", lambda e: patch_xpbd_k_torque(e, 0.005), label="xpbd k_torque=0.005")
    run_with_monitor("geds", label="geds baseline")
    run_with_monitor("geds", patch_geds_torque_limit, label="geds torque_limit=1")
    run_with_monitor("geds", patch_geds_lower_stiff, label="geds stiff x0.5")


if __name__ == "__main__":
    main()
