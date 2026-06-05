#!/usr/bin/env python3
"""Multi-step XPBD/GEDS ablations."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapteddlo_muj.envs.test_shape_w_arm.registry import get_model_specs

FAIL_QVEL = 50.0


def run_steps(model_name, n_steps=30, patch_fn=None, label=""):
    env = get_model_specs([model_name])[model_name]["create_env"]("white", None, False)
    dlo = env.dlo_sim
    if patch_fn:
        patch_fn(env, dlo)
    peak = {"force": 0, "tau": 0, "torq": 0, "qvel": 0}
    fail_at = None
    for step in range(1, n_steps + 1):
        env.grav_comp()
        env.data.ctrl[env.joint_ids] = env.observations["qpos"]
        if model_name == "xpbd":
            dlo._calc_wrenches()
            dlo._forces_to_torques()
            env.data.qfrc_passive[dlo.qvel0_addr : dlo.qvellast_addr + 1] += dlo.torq_node[1:-1].flatten()
            peak["force"] = max(peak["force"], float(np.max(np.linalg.norm(dlo.force_node, axis=1))))
            peak["tau"] = max(peak["tau"], float(np.max(np.linalg.norm(dlo.tau_direct, axis=1))))
            peak["torq"] = max(peak["torq"], float(np.max(np.linalg.norm(dlo.torq_node, axis=1))))
        elif model_name == "geds":
            dlo._calc_elastic_wrenches()
            dlo._forces_to_torques()
            env.data.qfrc_passive[dlo.qvel0_addr : dlo.qvellast_addr + 1] += dlo.torq_node[1:-1].flatten()
            peak["force"] = max(peak["force"], float(np.max(np.linalg.norm(dlo.force_node, axis=1))))
            peak["tau"] = max(peak["tau"], float(np.max(np.linalg.norm(dlo.tau_roll, axis=1))))
            peak["torq"] = max(peak["torq"], float(np.max(np.linalg.norm(dlo.torq_node, axis=1))))
        env.sim.step()
        env.sim.forward()
        env._get_observations()
        qv = float(np.max(np.abs(env.data.qvel[env.joint_qveladdr_full])))
        peak["qvel"] = max(peak["qvel"], qv)
        if qv > FAIL_QVEL and fail_at is None:
            fail_at = step
    print(f"  {label:40s} fail@{fail_at} peak_qv={peak['qvel']:.3g} peak_f={peak['force']:.3g} peak_tau={peak['tau']:.3g} peak_torq={peak['torq']:.3g}")
    return fail_at


def patch_site_xpos(env, dlo):
    orig = dlo._calc_wrenches

    def _calc_wrenches_site():
        dlo.x_flat = env.data.site_xpos[dlo.vec_siteid[:]].flatten()
        dlo.quat_flat = env.data.xquat[dlo.vec_bodyid[:]].flatten()
        dlo._fill_inv_mass_inertia()
        dlo.xpbd_math.computeWrenches(
            dlo.x_flat,
            dlo.quat_flat,
            dlo.inv_mass,
            dlo.inv_inertia_w_flat,
            dlo.model.opt.timestep,
            dlo.force_node_flat,
            dlo.tau_direct_flat,
        )
        dlo.force_node = dlo.force_node_flat.reshape((dlo.nv + 2, 3))
        dlo.tau_direct = dlo.tau_direct_flat.reshape((dlo.nv + 2, 3))
        dlo._limit_forces()

    dlo._calc_wrenches = _calc_wrenches_site
    # rest pose from sites too
    dlo.x_flat = env.data.site_xpos[dlo.vec_siteid[:]].flatten().copy()
    dlo.quat_flat = env.data.xquat[dlo.vec_bodyid[:]].flatten().copy()
    dlo.xpbd_math.reinitRestPose(dlo.x_flat, dlo.quat_flat)


def main():
    print("XPBD ablations (30 idle steps):")
    run_steps("xpbd", 30, label="baseline")
    run_steps("xpbd", 30, lambda e, d: (d.xpbd_math.setForceGain(0.0), setattr(d, "k_force", 0.0)), label="k_force=0")
    run_steps("xpbd", 30, lambda e, d: (d.xpbd_math.setTorqueGain(0.01), setattr(d, "k_torque", 0.01)), label="k_torque=0.01")
    run_steps("xpbd", 30, lambda e, d: (d.xpbd_math.setForceGain(0.01), d.xpbd_math.setTorqueGain(0.01), setattr(d, "k_force", 0.01), setattr(d, "k_torque", 0.01)), label="k_f=k_t=0.01")
    run_steps("xpbd", 30, lambda e, d: (d.xpbd_math.setTorqueGain(0.001), setattr(d, "k_torque", 0.001)), label="k_torque=0.001")
    run_steps("xpbd", 30, lambda e, d: d.xpbd_math.setNumIterations(1), label="num_iters=1")
    run_steps("xpbd", 30, patch_site_xpos, label="site_xpos read")

    print("\nGEDS ablations (30 idle steps):")
    run_steps("geds", 30, label="baseline")
    run_steps("geds", 30, lambda e, d: setattr(d, "k_p_endpoint", 0.0), label="k_p_endpoint=0")
    run_steps("geds", 30, lambda e, d: setattr(d, "samples_per_span", 2), label="samples_per_span=2")

    # softer material for xpbd
    print("\nXPBD softer material:")
    def softer(e, d):
        d.youngs_modulus *= 0.1
        d.torsion_modulus *= 0.1
        d.xpbd_math.setMaterial(d.youngs_modulus, d.torsion_modulus)
    run_steps("xpbd", 30, softer, label="E,G x0.1")


if __name__ == "__main__":
    main()
