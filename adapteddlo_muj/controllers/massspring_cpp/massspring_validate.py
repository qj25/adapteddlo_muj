#!/usr/bin/env python3
"""Toy validation for MassSpring bending torque (mirrors MassSpring.cpp math)."""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    qn = np.asarray(q, dtype=np.float64).copy()
    n = np.linalg.norm(qn)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    qn /= n
    if qn[0] < 0.0:
        qn = -qn
    return qn


def invert_quat(q: np.ndarray) -> np.ndarray:
    qi = -np.asarray(q, dtype=np.float64)
    qi[0] = -qi[0]
    return qi / np.dot(qi, qi)


def multiply_quat(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    wa, xa, ya, za = qa
    wb, xb, yb, zb = qb
    out = np.array(
        [
            wa * wb - xa * xb - ya * yb - za * zb,
            wa * xb + xa * wb + ya * zb - za * yb,
            wa * yb - xa * zb + ya * wb + za * xb,
            wa * zb + xa * yb - ya * xb + za * wb,
        ],
        dtype=np.float64,
    )
    return normalize_quat(out)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    qn = normalize_quat(q)
    w = float(np.clip(qn[0], -1.0, 1.0))
    v = qn[1:4]
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return 2.0 * v
    angle = 2.0 * np.arctan2(v_norm, w)
    return (angle / v_norm) * v


def rot_vec_quat(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    quat = np.asarray(quat, dtype=np.float64)
    if np.dot(vec, vec) < 1e-24:
        return np.zeros(3)
    if abs(quat[0] - 1.0) < 1e-12 and np.dot(quat[1:4], quat[1:4]) < 1e-12:
        return vec.copy()
    q_xyz = quat[1:4]
    tmp = np.array(
        [
            quat[0] * vec[0] + quat[2] * vec[2] - quat[3] * vec[1],
            quat[0] * vec[1] + quat[3] * vec[0] - quat[1] * vec[2],
            quat[0] * vec[2] + quat[1] * vec[1] - quat[2] * vec[0],
        ]
    )
    return vec + 2.0 * np.cross(q_xyz, tmp)


def axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    half = 0.5 * angle
    return normalize_quat(
        np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float64)
    )


def compute_torque_ref(
    positions: np.ndarray,
    quats: np.ndarray,
    neutral_quats: np.ndarray,
    k_bend: float,
    *,
    tangent_frame: str = "world",
) -> np.ndarray:
    """Reference port of MassSpring::computeTorque."""
    n = len(quats)
    node_torque = np.zeros((n, 3))
    fallback = np.array([1.0, 0.0, 0.0])

    for i in range(1, n - 1):
        q_prev = normalize_quat(quats[i - 1])
        q_cur = normalize_quat(quats[i])
        q_rel = multiply_quat(invert_quat(q_prev), q_cur)
        q_rel0 = multiply_quat(
            invert_quat(normalize_quat(neutral_quats[i - 1])),
            normalize_quat(neutral_quats[i]),
        )
        q_dev = multiply_quat(invert_quat(q_rel0), q_rel)
        dev = quat_to_rotvec(q_dev)

        seg = positions[i] - positions[i - 1]
        seg_n = np.linalg.norm(seg)
        tangent_world = seg / seg_n if seg_n > 1e-12 else fallback

        if tangent_frame == "world":
            tangent = tangent_world
        elif tangent_frame == "parent":
            tangent = rot_vec_quat(tangent_world, invert_quat(q_prev))
            t_n = np.linalg.norm(tangent)
            tangent = tangent / t_n if t_n > 1e-12 else fallback
        else:
            raise ValueError(tangent_frame)

        bend_vec = dev - np.dot(dev, tangent) * tangent
        tau_parent = -k_bend * bend_vec
        tau_child = rot_vec_quat(tau_parent, invert_quat(q_rel))
        node_torque[i] = tau_child
    return node_torque


def straight_rope(n_nodes: int, seg_len: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
  pos = np.zeros((n_nodes, 3))
  for i in range(1, n_nodes):
      pos[i, 0] = pos[i - 1, 0] + seg_len
  quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_nodes, 1))
  return pos, quat


def constant_y_bend(
    n_nodes: int,
    seg_len: float,
    theta_per_joint: float,
) -> tuple[np.ndarray, np.ndarray]:
    pos, quat = straight_rope(n_nodes, seg_len)
    axis = np.array([0.0, 1.0, 0.0])
    cumulative = axis_angle_quat(axis, 0.0)
    for i in range(1, n_nodes):
        cumulative = multiply_quat(cumulative, axis_angle_quat(axis, theta_per_joint))
        quat[i] = cumulative
        # rotate segment direction for positions (visual centerline bend)
        seg = rot_vec_quat(np.array([seg_len, 0.0, 0.0]), cumulative)
        pos[i] = pos[i - 1] + seg
    return pos, quat


def helical_bend(n_nodes: int, seg_len: float, theta: float) -> tuple[np.ndarray, np.ndarray]:
    pos, quat = straight_rope(n_nodes, seg_len)
    for i in range(1, n_nodes):
        axis = np.array([0.2, 1.0, 0.3 * (i % 2)])
        q_step = axis_angle_quat(axis, theta)
        quat[i] = multiply_quat(quat[i - 1], q_step)
        seg = rot_vec_quat(np.array([seg_len, 0.0, 0.0]), quat[i - 1])
        pos[i] = pos[i - 1] + seg
    return pos, quat


def test_helical_frame_mismatch():
    k_bend = 2.0
    theta = np.deg2rad(15.0)
    n_nodes = 6
    pos, quat = helical_bend(n_nodes, 0.03, theta)
    neutral = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_nodes, 1))

    torq_world = compute_torque_ref(pos, quat, neutral, k_bend, tangent_frame="world")
    torq_parent = compute_torque_ref(pos, quat, neutral, k_bend, tangent_frame="parent")

    print("\nhelical bend, theta = {:.3f} rad".format(theta))
    for i in range(1, n_nodes - 1):
        tw = torq_world[i]
        tp = torq_parent[i]
        angle = np.degrees(
            np.arccos(
                np.clip(
                    np.dot(tw, tp) / (np.linalg.norm(tw) * np.linalg.norm(tp) + 1e-12),
                    -1.0,
                    1.0,
                )
            )
        )
        print(
            "  joint {:d}: |w|={:.4f} |p|={:.4f} angle(w,p)={:.1f}deg".format(
                i, np.linalg.norm(tw), np.linalg.norm(tp), angle
            )
        )


def test_constant_bend_uniformity():
    k_bend = 2.0
    theta = np.deg2rad(8.0)
    n_nodes = 6  # nv+2 with nv=4 joints
    pos, quat = constant_y_bend(n_nodes, 0.03, theta)
    neutral = quat.copy()
    neutral[:] = np.array([1.0, 0.0, 0.0, 0.0])

    torq_world = compute_torque_ref(pos, quat, neutral, k_bend, tangent_frame="world")
    torq_parent = compute_torque_ref(pos, quat, neutral, k_bend, tangent_frame="parent")

    joint_idx = list(range(1, n_nodes - 1))
    mags_world = [np.linalg.norm(torq_world[i]) for i in joint_idx]
    mags_parent = [np.linalg.norm(torq_parent[i]) for i in joint_idx]

    print("constant Y bend, theta_per_joint = {:.3f} rad".format(theta))
    print("  joint indices:", joint_idx)
    print("  |tau| world tangent:  ", ["{:.4f}".format(m) for m in mags_world])
    print("  |tau| parent tangent:", ["{:.4f}".format(m) for m in mags_parent])

    # Interior joints should match for uniform bend; last joint included.
    interior_world = mags_world[1:-1]
    spread_world = max(interior_world) - min(interior_world) if interior_world else 0.0
    interior_parent = mags_parent[1:-1]
    spread_parent = max(interior_parent) - min(interior_parent) if interior_parent else 0.0
    print("  interior spread world:  {:.4f}".format(spread_world))
    print("  interior spread parent: {:.4f}".format(spread_parent))
    print("  last joint |tau| world:  {:.4f}".format(mags_world[-1]))
    print("  last joint |tau| parent: {:.4f}".format(mags_parent[-1]))

    expected = k_bend * theta
    print("  expected ~k*theta = {:.4f}".format(expected))
    return torq_world, torq_parent


def test_cpp_module_if_available():
    try:
        from adapteddlo_muj.controllers.massspring_cpp import MassSpring as MS
    except ImportError:
        print("MassSpring extension not built; skipping C++ comparison")
        return

    k_bend = 2.0
    theta = np.deg2rad(8.0)
    n_nodes = 6
    pos, quat = constant_y_bend(n_nodes, 0.03, theta)
    neutral = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_nodes, 1))

    ms = MS.MassSpring(neutral.flatten(), k_bend, k_bend)
    out = np.zeros(n_nodes * 3)
    ms.computeTorque(pos.flatten(), quat.flatten(), out)
    torq_cpp = out.reshape(n_nodes, 3)
    torq_ref = compute_torque_ref(pos, quat, neutral, k_bend, tangent_frame="world")

    print("\nC++ vs python reference (world tangent):")
    for i in range(1, n_nodes - 1):
        diff = np.linalg.norm(torq_cpp[i] - torq_ref[i])
        print("  joint {:d}: |diff|={:.6e}, |cpp|={:.4f}".format(i, diff, np.linalg.norm(torq_cpp[i])))


if __name__ == "__main__":
    test_constant_bend_uniformity()
    test_helical_frame_mismatch()
    test_cpp_module_if_available()
