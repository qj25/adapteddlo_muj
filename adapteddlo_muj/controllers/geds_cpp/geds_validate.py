#!/usr/bin/env python3
"""Smoke and validation checks for Tier-A GEDS (plan section 5)."""

import os
import sys

import numpy as np
import mujoco

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from adapteddlo_muj.assets.genrope.gdv_O import GenKin_O
from adapteddlo_muj.controllers.geds_cpp import RodGeds
from adapteddlo_muj.controllers.ropekin_controller_geds import DLORopeGeds
from adapteddlo_muj.envs.our_rope_valid_test import TestRopeEnv
from adapteddlo_muj.utils.xml_utils import XMLWrapper


def _cr_pos(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _make_free_rope_env(r_pieces=15, r_len=0.6, r_thickness=0.02):
    assets = os.path.join(ROOT, "adapteddlo_muj/assets")
    world_path = os.path.join(assets, "world_test.xml")
    rope_path = os.path.join(assets, "dlorope1dkin.xml")
    box_path = os.path.join(assets, "anchorbox.xml")
    init_pos = np.array([r_len / 2.0, 0.0, 0.5])
    init_quat = np.array([1.0, 0.0, 0.0, 0.0])
    GenKin_O(
        r_len=r_len,
        r_thickness=r_thickness,
        r_pieces=r_pieces,
        j_stiff=0.0,
        j_damp=0.5,
        init_pos=init_pos,
        init_quat=init_quat,
        coll_on=False,
        d_small=0.0,
        rope_type="capsule",
        vis_subcyl=False,
        obj_path=rope_path,
    )
    xml = XMLWrapper(world_path)
    dlorope = XMLWrapper(rope_path)
    anchorbox = XMLWrapper(box_path)
    xml.merge_multiple(anchorbox, ["worldbody", "equality", "contact"])
    xml.merge_multiple(dlorope, ["worldbody"])
    model = mujoco.MjModel.from_xml_string(xml.get_xml_string())
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0.0
    mujoco.mj_forward(model, data)
    seg_len = r_len / float(r_pieces)
    ctrl = DLORopeGeds(
        model=model,
        data=data,
        n_link=r_pieces,
        segment_length=seg_len,
        radius=r_thickness / 2.0,
        alpha_bar=1.345 / 10,
        beta_bar=0.789 / 10,
        bothweld=False,
    )
    return model, data, ctrl


def test_catmull_rom_endpoints():
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([2.0, 0.5, 0.0])
    p3 = np.array([3.0, 0.0, 0.0])
    at_start = _cr_pos(p0, p1, p2, p3, 0.0)
    at_end = _cr_pos(p0, p1, p2, p3, 1.0)
    assert np.allclose(at_start, p1, atol=1e-12), f"start mismatch {at_start}"
    assert np.allclose(at_end, p2, atol=1e-12), f"end mismatch {at_end}"


def test_cpp_rest_zero_force():
    n = 8
    pts = np.zeros((n, 3))
    pts[:, 0] = np.linspace(0.0, 0.7, n)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    rod = RodGeds.RodGeds(n, 0.1, 0.02, 1e8, 1e7)
    rod.reinitRest(pts.flatten(), quat.flatten())
    force = np.zeros(3 * n)
    torque = np.zeros(3 * n)
    rod.computeElasticWrenches(pts.flatten(), quat.flatten(), force, torque)
    assert np.linalg.norm(force) < 1e-6
    assert np.linalg.norm(torque) < 1e-6


def test_cpp_bent_restoring_force():
    n = 8
    pts = np.zeros((n, 3))
    pts[:, 0] = np.linspace(0.0, 0.7, n)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    bent = pts.copy()
    bent[3, 2] = 0.08
    rod = RodGeds.RodGeds(n, 0.1, 0.02, 1e9, 1e8)
    rod.reinitRest(pts.flatten(), quat.flatten())
    force = np.zeros(3 * n)
    torque = np.zeros(3 * n)
    rod.computeElasticWrenches(bent.flatten(), quat.flatten(), force, torque)
    f_norm = np.linalg.norm(force)
    assert f_norm > 1.0, f"expected restoring force, got {f_norm}"
    assert force[3 * 3 + 2] < 0.0, "bend node force should restore toward rest"


def test_twist_torque_on_quat_perturbation():
    n = 6
    pts = np.zeros((n, 3))
    pts[:, 0] = np.linspace(0.0, 0.5, n)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    twisted = quat.copy()
    angle = 0.35
    for i in range(1, n - 1):
        twisted[i] = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
    rod = RodGeds.RodGeds(n, 0.1, 0.02, 1e10, 1e9)
    rod.reinitRest(pts.flatten(), quat.flatten())
    force = np.zeros(3 * n)
    torque = np.zeros(3 * n)
    rod.computeElasticWrenches(pts.flatten(), twisted.flatten(), force, torque)
    assert np.linalg.norm(torque) > 1e-2, "expected twist restoring torque"


def test_geds_restoring_direction():
    n = 8
    pts = np.zeros((n, 3))
    pts[:, 0] = np.linspace(0.0, 0.7, n)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    bent = pts.copy()
    bent[3, 1] = 0.05
    bent[5, 2] = -0.04
    rod = RodGeds.RodGeds(n, 0.1, 0.02, 1e9, 1e8)
    rod.reinitRest(pts.flatten(), quat.flatten())
    force = np.zeros(3 * n)
    torque = np.zeros(3 * n)
    rod.computeElasticWrenches(bent.flatten(), quat.flatten(), force, torque)
    for idx in (3, 5):
        disp = bent[idx] - pts[idx]
        f = force[idx * 3 : idx * 3 + 3]
        assert np.dot(f, disp) < 0.0, f"node {idx} force should oppose displacement"


def test_mujoco_geds_step():
    env = TestRopeEnv(
        do_render=False,
        r_pieces=20,
        r_len=0.9,
        r_thickness=0.03,
        test_type="speedtest1",
        model_name="geds",
        new_start=True,
    )
    for _ in range(25):
        env.step()
    assert env.env_steps == 25
    assert np.all(np.isfinite(env.data.qpos))


def test_endpoint_position_hold():
    model, data, ctrl = _make_free_rope_env()
    bid = ctrl.vec_bodyid[0]
    target = data.xpos[bid].copy()
    max_err = 0.0
    for _ in range(80):
        ctrl.update_torque(start_pos=target, end_pos=None)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        max_err = max(max_err, np.linalg.norm(data.xpos[bid] - target))
    assert max_err < 0.02, f"endpoint hold error too large: {max_err}"


def main():
    tests = [
        ("catmull_rom_endpoints", test_catmull_rom_endpoints),
        ("cpp_rest_zero_force", test_cpp_rest_zero_force),
        ("cpp_bent_restoring_force", test_cpp_bent_restoring_force),
        ("twist_torque", test_twist_torque_on_quat_perturbation),
        ("geds_restoring_direction", test_geds_restoring_direction),
        ("mujoco_geds_step", test_mujoco_geds_step),
        ("endpoint_position_hold", test_endpoint_position_hold),
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
