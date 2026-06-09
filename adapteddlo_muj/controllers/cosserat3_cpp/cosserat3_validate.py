#!/usr/bin/env python3
"""Validation checks for JTill2017 Kirchhoff rod (cosserat3) direct solver."""

import os
import sys

import numpy as np
import mujoco

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from adapteddlo_muj.controllers.cosserat3_cpp import RodCosserat3
from adapteddlo_muj.envs.our_rope_valid_test import TestRopeEnv


def _init_bf_from_x(x):
    e0 = x[1] - x[0]
    e_bar = np.linalg.norm(e0)
    bf0 = np.zeros((3, 3))
    bf0[0, :] = e0 / e_bar
    bf0[1, :] = np.cross(bf0[0, :], np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(bf0[1, :]) < 1e-6:
        bf0[1, :] = np.cross(bf0[0, :], np.array([0.0, 1.0, 0.0]))
    bf0[1, :] /= np.linalg.norm(bf0[1, :])
    bf0[2, :] = np.cross(bf0[0, :], bf0[1, :])
    return bf0


def _make_straight_rod(n=8, length=0.7, alpha=1.345 / 10, beta=0.789 / 10, radius=0.01):
    x = np.zeros((n, 3))
    x[:, 0] = np.linspace(0.0, length, n)
    bf0 = _init_bf_from_x(x)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    return x, bf0, quat, alpha, beta, radius


def test_cpp_rest_zero_wrench():
    n = 8
    x, bf0, _, alpha, beta, radius = _make_straight_rod(n=n)
    rod = RodCosserat3.RodCosserat3(
        x.flatten(), bf0.flatten(), 0.0, 0.0, alpha, beta, radius
    )
    bf_end = np.zeros(9)
    rod.updateVars(x.flatten(), bf0.flatten(), bf_end)
    force = np.zeros(3 * n)
    torque = np.zeros(3 * n)
    rod.calculateCenterlineF2(force)
    rod.calculateCenterlineTorq(torque, np.tile([1, 0, 0, 0], n), 0)
    assert np.linalg.norm(force) < 1e-5, f"rest force {np.linalg.norm(force)}"
    assert np.linalg.norm(torque) < 1e-4, f"rest torque {np.linalg.norm(torque)}"


def test_cpp_bent_restoring_force():
    n = 8
    x, bf0, quat, alpha, beta, radius = _make_straight_rod(n=n)
    bent = x.copy()
    bent[3, 2] = 0.08
    bf0_bent = _init_bf_from_x(bent)
    rod = RodCosserat3.RodCosserat3(
        x.flatten(), bf0.flatten(), 0.0, 0.0, alpha, beta, radius
    )
    bf_end = np.zeros(9)
    rod.updateVars(bent.flatten(), bf0_bent.flatten(), bf_end)
    force = np.zeros(3 * n)
    rod.calculateCenterlineF2(force)
    f_norm = np.linalg.norm(force)
    assert f_norm > 1e-4, f"expected restoring force, got {f_norm}"
    assert force[3 * 3 + 2] < 0.0, "bend node z-force should restore toward rest"


def test_stiffness_scaling():
    n = 8
    x, bf0, quat, alpha, beta, radius = _make_straight_rod(n=n)
    bent = x.copy()
    bent[4, 2] = 0.06
    bf0_bent = _init_bf_from_x(bent)

    def _force_mag(a_scale):
        rod = RodCosserat3.RodCosserat3(
            x.flatten(), bf0.flatten(), 0.0, 0.0, alpha * a_scale, beta, radius
        )
        bf_end = np.zeros(9)
        rod.updateVars(bent.flatten(), bf0_bent.flatten(), bf_end)
        force = np.zeros(3 * n)
        rod.calculateCenterlineF2(force)
        return np.linalg.norm(force)

    f1 = _force_mag(1.0)
    f2 = _force_mag(2.0)
    ratio = f2 / f1 if f1 > 1e-12 else 0.0
    assert 1.5 < ratio < 2.5, f"stiffness ratio {ratio}, f1={f1}, f2={f2}"


def test_energy_positive_definite():
    n = 8
    x, bf0, _, alpha, beta, radius = _make_straight_rod(n=n)
    bent = x.copy()
    bent[3, 2] = 0.05
    bf0_bent = _init_bf_from_x(bent)
    rod = RodCosserat3.RodCosserat3(
        x.flatten(), bf0.flatten(), 0.0, 0.0, alpha, beta, radius
    )
    bf_end = np.zeros(9)
    rod.updateVars(bent.flatten(), bf0_bent.flatten(), bf_end)
    e_bent = rod.calculateEnergy()
    rod.updateVars(x.flatten(), bf0.flatten(), bf_end)
    e_rest = rod.calculateEnergy()
    assert e_bent > e_rest + 1e-10


def test_mujoco_smoke():
    env = TestRopeEnv(
        do_render=False,
        r_pieces=12,
        r_len=0.5,
        r_thickness=0.02,
        alpha_bar=1.345 / 10,
        beta_bar=0.789 / 10,
        test_type="speedtest1",
        model_name="cosserat3",
        new_start=True,
    )
    for _ in range(20):
        env.step()
    assert env.env_steps == 20
    assert np.all(np.isfinite(env.data.qpos))


def main():
    tests = [
        test_cpp_rest_zero_wrench,
        test_cpp_bent_restoring_force,
        test_stiffness_scaling,
        test_energy_positive_definite,
        test_mujoco_smoke,
    ]
    for t in tests:
        print(f"running {t.__name__}...")
        t()
        print(f"  OK")
    print("All cosserat3 validation tests passed.")


if __name__ == "__main__":
    main()
