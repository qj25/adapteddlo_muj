#!/usr/bin/env python3
"""GEDS phase-specific failure analysis."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import adapteddlo_muj.utils.transform_utils as T
from adapteddlo_muj.envs.test_shape_w_arm.base import run_manipulation
from adapteddlo_muj.envs.test_shape_w_arm.registry import get_model_specs

FAIL_QVEL = 80.0


def monitor_env(env, label):
    fail = {"step": None, "phase": None, "qv": None}
    orig = env.step
    phase = ["init"]

    def wrapped(action=np.zeros(6)):
        obs, r, d, t, i = orig(action)
        qv = float(np.max(np.abs(env.data.qvel[env.joint_qveladdr_full])))
        if fail["step"] is None and qv > FAIL_QVEL:
            fail["step"] = env.env_steps
            fail["phase"] = phase[0]
            fail["qv"] = qv
            dlo = env.dlo_sim
            print(
                f"  {label} FAIL step={fail['step']} phase={fail['phase']} qv={qv:.3g} "
                f"max_f={np.max(np.linalg.norm(dlo.force_node,axis=1)):.4g} "
                f"max_torq={np.max(np.linalg.norm(dlo.torq_node,axis=1)):.4g}"
            )
        return obs, r, d, t, i

    env.step = wrapped
    orig_mtp = env.move_to_pose
    orig_rot = env.rot_x_rads
    orig_hold = env.hold_pos

    def mtp(*a, **k):
        phase[0] = "move_to_pose"
        return orig_mtp(*a, **k)

    def rot(*a, **k):
        phase[0] = "rot_x_rads"
        return orig_rot(*a, **k)

    def hold(*a, **k):
        phase[0] = "hold_pos"
        return orig_hold(*a, **k)

    env.move_to_pose = mtp
    env.rot_x_rads = rot
    env.hold_pos = hold
    return fail


def run_case(label, patch=None):
    print(f"\n## {label}")
    env = get_model_specs(["geds"])["geds"]["create_env"]("white", None, False)
    env.max_action = 0.02
    if patch:
        patch(env)
    fail = monitor_env(env, label)
    move_pos = np.array([[0.165, 0.1, 0.1]])
    move_quat = np.array([T.axisangle2quat(np.array([50, 10, 0]) * np.pi / 180)])
    z_rot = np.array([360.0]) * np.pi / 180
    run_manipulation(env, move_pos, move_quat, z_rot, 0, getting_jointpos=False)
    print(f"  done steps={env.env_steps} fail={fail['step']} phase={fail['phase']}")
    return fail


def patch_torque_limit(env, lim=0.5):
    dlo = env.dlo_sim
    orig = dlo._forces_to_torques

    def lim_fn():
        orig()
        for i in range(len(dlo.torq_node)):
            m = np.linalg.norm(dlo.torq_node[i])
            if m > lim:
                dlo.torq_node[i] *= lim / m

    dlo._forces_to_torques = lim_fn


def patch_stiff_scale(env, scale=0.25):
    dlo = env.dlo_sim
    dlo.youngs_modulus *= scale
    dlo.torsion_modulus *= scale
    dlo.geds_math.setMaterial(dlo.youngs_modulus, dlo.torsion_modulus)


def patch_f_limit(env, lim=0.1):
    env.dlo_sim.f_limit = lim


def main():
    run_case("geds baseline")
    run_case("geds torque_limit=0.5", lambda e: patch_torque_limit(e, 0.5))
    run_case("geds stiff x0.25", lambda e: patch_stiff_scale(e, 0.25))
    run_case("geds f_limit=0.1", lambda e: patch_f_limit(e, 0.1))


if __name__ == "__main__":
    main()
