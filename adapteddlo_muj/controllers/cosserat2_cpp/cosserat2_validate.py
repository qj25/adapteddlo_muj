#!/usr/bin/env python3
"""Smoke and validation checks for Stable Cosserat2 rod controller."""

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from adapteddlo_muj.controllers.cosserat2_cpp import RodCosserat2
from adapteddlo_muj.envs.our_rope_valid_test import TestRopeEnv


def _quat_mul_wxyz(qa, qb):
    wa, xa, ya, za = qa
    wb, xb, yb, zb = qb
    return np.array([
        wa * wb - xa * xb - ya * yb - za * zb,
        wa * xb + xa * wb + ya * zb - za * yb,
        wa * yb - xa * zb + ya * wb + za * xb,
        wa * zb + xa * yb - ya * xb + za * wb,
    ])


def _quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_to_rotvec(q):
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    v = q[1:]
    vn = np.linalg.norm(v)
    if vn < 1e-12:
        return 2 * v
    ang = 2 * np.arctan2(vn, q[0])
    return (ang / vn) * v


def _joint_violation(qcur, qrest, i):
    q_rel = _quat_mul_wxyz(_quat_inv(qcur[i - 1]), qcur[i])
    q_rel0 = _quat_mul_wxyz(_quat_inv(qrest[i - 1]), qrest[i])
    q_dev = _quat_mul_wxyz(_quat_inv(q_rel0), q_rel)
    return _quat_to_rotvec(q_dev)


def _total_joint_violation(qcur, qrest):
    return sum(np.linalg.norm(_joint_violation(qcur, qrest, i)) for i in range(1, len(qcur)))


def _apply_child_torque_step(quat, torque, dt_step=0.002):
    out = quat.copy()
    torque = torque.reshape(len(quat), 3)
    for i in range(1, len(quat)):
        t = torque[i]
        ang = dt_step * np.linalg.norm(t)
        if ang < 1e-12:
            continue
        axis = t / np.linalg.norm(t)
        dq_local = np.array([np.cos(ang / 2), *(np.sin(ang / 2) * axis)])
        out[i] = _quat_mul_wxyz(out[i], dq_local)
        out[i] /= np.linalg.norm(out[i])
    return out


def test_cpp_rest_zero_wrench():
    n = 8
    pts = np.zeros((n, 3))
    pts[:, 0] = np.linspace(0.0, 0.7, n)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    inv_mass = np.ones(n)
    inv_inertia = np.tile(np.eye(3).flatten(), n)
    rod = RodCosserat2.RodCosserat2(n, pts.flatten(), quat.flatten(), 0.1, 0.01, 1e3, 1.0, 0.5, True)
    force = np.zeros(3 * n)
    torque = np.zeros(3 * n)
    rod.computeWrenches(pts.flatten(), quat.flatten(), inv_mass, inv_inertia, 0.002, force, torque)
    assert np.linalg.norm(force) < 1e-2
    assert np.linalg.norm(torque) < 1e-2


def test_stiffness_scaling():
    n = 6
    pts = np.zeros((n, 3))
    pts[:, 0] = np.linspace(0.0, 0.5, n)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    bent = quat.copy()
    for i in range(1, n - 1):
        bent[i] = [np.cos(0.1), 0, np.sin(0.1), 0]
    inv_mass = np.ones(n)
    inv_inertia = np.tile(np.eye(3).flatten(), n)
    norms = []
    for kb in [0.5, 2.0, 8.0]:
        rod = RodCosserat2.RodCosserat2(n, pts.flatten(), quat.flatten(), 0.1, 0.01, 1e3, kb, 0.5, True)
        force = np.zeros(3 * n)
        torque = np.zeros(3 * n)
        rod.computeWrenches(pts.flatten(), bent.flatten(), inv_mass, inv_inertia, 0.002, force, torque)
        norms.append(np.linalg.norm(torque))
    assert norms[1] > norms[0]
    assert norms[2] > norms[1]


def test_position_perturbation_restoring_force():
    n = 6
    rest_pts = np.zeros((n, 3))
    rest_pts[:, 0] = np.linspace(0.0, 0.5, n)
    perturbed = rest_pts.copy()
    perturbed[3, 1] = 0.05
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    inv_mass = np.ones(n)
    inv_inertia = np.tile(np.eye(3).flatten(), n)
    rod = RodCosserat2.RodCosserat2(n, rest_pts.flatten(), quat.flatten(), 0.1, 0.01, 1e3, 2.0, 0.5, True)
    rod.setForceGain(0.05)
    force = np.zeros(3 * n)
    torque = np.zeros(3 * n)
    dt = 0.002
    rod.computeWrenches(perturbed.flatten(), quat.flatten(), inv_mass, inv_inertia, dt, force, torque)
    force = force.reshape(n, 3)
    assert force[3, 1] < -1e-3, f"expected restoring Y force, got {force[3]}"
    assert np.linalg.norm(force) > 1e-2


def test_bent_quat_produces_torque():
    n = 6
    pts = np.zeros((n, 3))
    pts[:, 0] = np.linspace(0.0, 0.5, n)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    bent = quat.copy()
    for i in range(1, n - 1):
        bent[i] = [np.cos(0.15), 0, np.sin(0.15), 0]
    inv_mass = np.ones(n)
    inv_inertia = np.tile(np.eye(3).flatten(), n)
    rod = RodCosserat2.RodCosserat2(n, pts.flatten(), quat.flatten(), 0.1, 0.01, 1e3, 2.0, 0.5, True)
    rod.setTorqueGain(0.05)
    force = np.zeros(3 * n)
    torque = np.zeros(3 * n)
    rod.computeWrenches(pts.flatten(), bent.flatten(), inv_mass, inv_inertia, 0.002, force, torque)
    assert np.linalg.norm(torque) > 1e-1, "expected bend/twist restoring torque"


def test_mujoco_cosserat2_step():
    env = TestRopeEnv(
        do_render=False,
        r_pieces=20,
        r_len=0.9,
        r_thickness=0.03,
        test_type="speedtest1",
        model_name="cosserat2",
        new_start=True,
    )
    for _ in range(25):
        env.step()
    assert env.env_steps == 25
    assert np.all(np.isfinite(env.data.qpos))


def main():
    tests = [
        ("cpp_rest_zero_wrench", test_cpp_rest_zero_wrench),
        ("stiffness_scaling", test_stiffness_scaling),
        ("position_restoring_force", test_position_perturbation_restoring_force),
        ("bent_quat_torque", test_bent_quat_produces_torque),
        ("mujoco_cosserat2_step", test_mujoco_cosserat2_step),
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
