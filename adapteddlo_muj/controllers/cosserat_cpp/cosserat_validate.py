#!/usr/bin/env python3
"""Smoke and validation checks for Cosserat rod controller."""

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from adapteddlo_muj.controllers.cosserat_cpp import RodCosserat
from adapteddlo_muj.envs.our_rope_valid_test import TestRopeEnv


def test_cpp_rest_zero_torque():
    n = 8
    pts = np.zeros((n, 3))
    pts[:, 0] = np.linspace(0.0, 0.7, n)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    rod = RodCosserat.RodCosserat(n, 0.1, 1.0, 0.5)
    rod.reinitRest(pts.flatten(), quat.flatten())
    torque = np.zeros(3 * n)
    rod.computeWrenches(pts.flatten(), quat.flatten(), 0.002, torque)
    assert np.linalg.norm(torque) < 1e-4


def test_twist_torque_on_quat_perturbation():
    n = 6
    pts = np.zeros((n, 3))
    pts[:, 0] = np.linspace(0.0, 0.5, n)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    twisted = quat.copy()
    angle = 0.35
    for i in range(1, n - 1):
        twisted[i] = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
    rod = RodCosserat.RodCosserat(n, 0.1, 1.0, 0.5)
    rod.reinitRest(pts.flatten(), quat.flatten())
    torque = np.zeros(3 * n)
    rod.computeWrenches(pts.flatten(), twisted.flatten(), 0.002, torque)
    assert np.linalg.norm(torque) > 1e-2, "expected twist restoring torque"


def test_mujoco_cosserat_step():
    env = TestRopeEnv(
        do_render=False,
        r_pieces=20,
        r_len=0.9,
        r_thickness=0.03,
        test_type="speedtest1",
        model_name="cosserat",
        new_start=True,
    )
    for _ in range(25):
        env.step()
    assert env.env_steps == 25
    assert np.all(np.isfinite(env.data.qpos))


def main():
    tests = [
        ("cpp_rest_zero_torque", test_cpp_rest_zero_torque),
        ("twist_torque", test_twist_torque_on_quat_perturbation),
        ("mujoco_cosserat_step", test_mujoco_cosserat_step),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
