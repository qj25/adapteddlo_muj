#!/usr/bin/env python3
"""MuJoCo validation: per-joint bending torques from DLORopeMassSpring."""

from __future__ import annotations

import os
import sys

import mujoco
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from adapteddlo_muj.assets.genrope.gdv_O import GenKin_O
from adapteddlo_muj.controllers.ropekin_controller_massspring import DLORopeMassSpring
from adapteddlo_muj.utils.xml_utils import (
    XMLWrapper,
    allocate_genrope_xml_paths,
    release_genrope_xml_paths,
)


def make_env(r_pieces=12, r_len=0.36, r_thickness=0.02, bothweld=False):
    assets = os.path.join(ROOT, "adapteddlo_muj", "assets")
    world_path = os.path.join(assets, "world_test.xml")
    gen_paths = allocate_genrope_xml_paths("dlorope1dkin.xml")
    init_pos = np.array([r_len / 2.0, 0.0, 0.5])
    GenKin_O(
        r_len=r_len,
        r_thickness=r_thickness,
        r_pieces=r_pieces,
        j_stiff=0.0,
        j_damp=0.01,
        init_pos=init_pos,
        init_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        coll_on=False,
        d_small=0.0,
        rope_type="capsule",
        vis_subcyl=False,
        obj_path=gen_paths.rope,
    )
    xml = XMLWrapper(world_path)
    dlorope = XMLWrapper(gen_paths.rope)
    anchorbox = XMLWrapper(gen_paths.anchorbox)
    xml.merge_multiple(anchorbox, ["worldbody", "equality", "contact"])
    xml.merge_multiple(dlorope, ["worldbody"])
    model = mujoco.MjModel.from_xml_string(xml.get_xml_string())
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0.0
    mujoco.mj_forward(model, data)
    ctrl = DLORopeMassSpring(
        model=model,
        data=data,
        n_link=r_pieces,
        radius=r_thickness / 2.0,
        alpha_bar=2.0,
        beta_bar=0.0,
        bothweld=bothweld,
    )
    release_genrope_xml_paths(gen_paths)
    return model, data, ctrl


def apply_uniform_y_bend(model, ctrl, data, angle_per_joint):
    """Each ball joint gets the same relative rotation (not cumulative world twist)."""
    axis = np.array([0.0, 1.0, 0.0])
    half = 0.5 * angle_per_joint
    q_step = np.array([np.cos(half), *(np.sin(half) * axis)])

    for jnt_id in ctrl.dlo_joint_ids:
        adr = model.jnt_qposadr[jnt_id]
        data.qpos[adr : adr + 4] = q_step
    mujoco.mj_forward(model, data)


def report_torques(ctrl, label):
    ctrl._calc_centerline_torq()
    joint_torq = ctrl.torq_node[1 : ctrl.nv + 1]
    mags = np.linalg.norm(joint_torq, axis=1)
    print(label)
    print("  nv={}, n joints={}, torq_node shape={}".format(ctrl.nv, len(mags), ctrl.torq_node.shape))
    print("  |tau| per joint:", np.array2string(mags, precision=4, separator=", "))
    print("  last joint |tau|={:.4f}, first joint |tau|={:.4f}".format(mags[-1], mags[0]))
    print("  spread interior={:.4f}".format(mags[1:-1].max() - mags[1:-1].min() if len(mags) > 2 else 0.0))


if __name__ == "__main__":
    model, data, ctrl = make_env(r_pieces=12, bothweld=False)
    angle = np.deg2rad(6.0)

    ctrl.reset_neutral()
    mujoco.mj_forward(model, data)
    report_torques(ctrl, "straight (neutral captured)")

    apply_uniform_y_bend(model, ctrl, data, angle)
    report_torques(ctrl, "uniform Y bend, {:.1f} deg/joint".format(np.degrees(angle)))
