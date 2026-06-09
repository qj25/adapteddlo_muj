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
    """Apply child-node torques in body frame (matches MassSpring / qfrc_passive)."""
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


def test_cpp_rest_zero_torque():
    n = 8
    pts = np.zeros((n, 3))
    pts[:, 0] = np.linspace(0.0, 0.7, n)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    rod = RodCosserat.RodCosserat(n, 0.1, 1.0, 0.5)
    rod.reinitRest(pts.flatten(), quat.flatten())
    force = np.zeros(3 * n)
    torque = np.zeros(3 * n)
    rod.computeWrenches(pts.flatten(), quat.flatten(), 0.002, force, torque)
    assert np.linalg.norm(torque) < 1e-4
    assert np.linalg.norm(force) < 1e-4


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
    rod.setTorqueGain(1.0)
    rod.reinitRest(pts.flatten(), quat.flatten())
    force = np.zeros(3 * n)
    torque = np.zeros(3 * n)
    rod.computeWrenches(pts.flatten(), twisted.flatten(), 0.002, force, torque)
    assert np.linalg.norm(torque) > 1e-2, "expected twist restoring torque"


def test_torque_reduces_joint_violation():
    n = 6
    pts = np.zeros((n, 3))
    pts[:, 0] = np.linspace(0.0, 0.5, n)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    bent = quat.copy()
    angle = 0.2
    for i in range(1, n - 1):
        bent[i] = [np.cos(angle / 2), 0, np.sin(angle / 2), 0]

    rod = RodCosserat.RodCosserat(n, 0.1, 1.0, 0.5)
    rod.setTorqueGain(1.0)
    rod.reinitRest(pts.flatten(), quat.flatten())
    force = np.zeros(3 * n)
    torque = np.zeros(3 * n)
    dt = 0.002
    rod.computeWrenches(pts.flatten(), bent.flatten(), dt, force, torque)

    viol_before = _total_joint_violation(bent, quat)
    bent_after = _apply_child_torque_step(bent, torque, dt_step=0.05)
    viol_after = _total_joint_violation(bent_after, quat)
    assert viol_after < viol_before, (
        f"torque should reduce joint violation ({viol_before:.6f} -> {viol_after:.6f})"
    )


def test_stretch_force_on_elongated_segment():
    n = 3
    seg_len = 0.1
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    rest_pts = np.zeros((n, 3))
    rest_pts[:, 0] = np.linspace(0.0, seg_len * (n - 1), n)
    stretched = rest_pts.copy()
    stretched[1, 0] += 0.05
    rod = RodCosserat.RodCosserat(n, seg_len, 1.0, 0.5)
    rod.setStretchStiffness(10.0)
    rod.reinitRest(rest_pts.flatten(), quat.flatten())
    force = np.zeros(3 * n)
    torque = np.zeros(3 * n)
    rod.computeWrenches(stretched.flatten(), quat.flatten(), 0.002, force, torque)
    force = force.reshape(n, 3)
    # middle node pulled toward neighbors when one segment is elongated
    assert force[1, 0] < 0.0
    assert np.linalg.norm(force) > 1e-3


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
        ("torque_reduces_violation", test_torque_reduces_joint_violation),
        ("stretch_force", test_stretch_force_on_elongated_segment),
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
