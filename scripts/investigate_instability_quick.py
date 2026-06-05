#!/usr/bin/env python3
"""Quick instability probe: first steps + abbreviated moveid 2 phases."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import adapteddlo_muj.utils.transform_utils as T
from adapteddlo_muj.envs.test_shape_w_arm.registry import get_model_specs

FAIL_QVEL = 50.0


def site_body_diff(env, dlo):
    return float(
        np.max(
            np.linalg.norm(
                env.data.xpos[dlo.vec_bodyid[:]] - env.data.site_xpos[dlo.vec_siteid[:]],
                axis=1,
            )
        )
    )


def rope_metrics(env):
    qv = env.data.qvel[env.joint_qveladdr_full]
    qf = env.data.qfrc_passive[env.joint_qveladdr_full]
    return float(np.max(np.abs(qv))), float(np.max(np.abs(qf))), bool(np.any(np.isnan(qv)))


def apply_elastic(env):
    dlo = env.dlo_sim
    if env.rope_type == "xpbd":
        dlo._calc_wrenches()
        dlo._forces_to_torques()
        env.data.qfrc_passive[dlo.qvel0_addr : dlo.qvellast_addr + 1] += dlo.torq_node[1:-1].flatten()
        return {
            "force": float(np.max(np.linalg.norm(dlo.force_node, axis=1))),
            "tau_direct": float(np.max(np.linalg.norm(dlo.tau_direct, axis=1))),
            "torq": float(np.max(np.linalg.norm(dlo.torq_node, axis=1))),
        }
    if env.rope_type == "geds":
        dlo._calc_elastic_wrenches()
        dlo._forces_to_torques()
        env.data.qfrc_passive[dlo.qvel0_addr : dlo.qvellast_addr + 1] += dlo.torq_node[1:-1].flatten()
        return {
            "force": float(np.max(np.linalg.norm(dlo.force_node, axis=1))),
            "tau_roll": float(np.max(np.linalg.norm(dlo.tau_roll, axis=1))),
            "torq": float(np.max(np.linalg.norm(dlo.torq_node, axis=1))),
        }
    if env.rope_type == "adapt":
        dlo._update_dlo_cpp()
        dlo._calc_centerlineTorq(excl_joints=0)
        env.data.qfrc_passive[dlo.qvel0_addr : dlo.qvellast_addr + 1] += dlo.torq_node[1:-1].flatten()
        return {"torq": float(np.max(np.linalg.norm(dlo.torq_node, axis=1)))}
    return {}


def sim_step(env, action=None):
    if action is None:
        action = env.observations["qpos"]
    env.grav_comp()
    env.data.ctrl[env.joint_ids] = action
    w = apply_elastic(env)
    env.sim.step()
    env.sim.forward()
    env.env_steps += 1
    env.cur_time += env.dt
    env._get_observations()
    qv, qf, nan = rope_metrics(env)
    w["qvel"] = qv
    w["qfrc"] = qf
    w["nan"] = nan
    w["site_diff"] = site_body_diff(env, env.dlo_sim)
    w["step"] = env.env_steps
    return w


def probe_idle(model_name, n_steps=20):
    print(f"\n## {model_name} idle ({n_steps} steps)")
    env = get_model_specs([model_name])[model_name]["create_env"]("white", None, False)
    env.max_action = 0.02
    dlo = env.dlo_sim
    print(f"  dt={env.dt} stiff_scale material: E={getattr(dlo, 'youngs_modulus', 'n/a')} G={getattr(dlo, 'torsion_modulus', 'n/a')}")
    print(f"  k_force={getattr(dlo, 'k_force', 'n/a')} k_torque={getattr(dlo, 'k_torque', 'n/a')} bothweld={dlo.bothweld}")
    for i in range(n_steps):
        m = sim_step(env)
        if i < 5 or m["qvel"] > 1.0 or m["nan"]:
            print(f"  step{m['step']}: {m}")
        if m["nan"] or m["qvel"] > FAIL_QVEL:
            print(f"  FAIL at idle step {m['step']}")
            return env, m["step"]
    return env, None


def probe_move_to_pose(model_name, max_iters=500):
    print(f"\n## {model_name} move_to_pose (cap {max_iters} ik iterations)")
    env = get_model_specs([model_name])[model_name]["create_env"]("white", None, False)
    env.max_action = 0.02
    desired_pos = env.init_pos + np.array([0.165, 0.1, 0.1])
    desired_quat = T.quat_multiply(env.init_quat, T.axisangle2quat(np.array([50, 10, 0]) * np.pi / 180))
    goal = np.concatenate((desired_pos, desired_quat))
    targ_qpos = env.ik_arm.calc_ik(goal, env.observations["qpos"]).copy()

    fail_step = None
    for it in range(max_iters):
        qpos_diff = env.joint_sum(targ_qpos, -env.observations["qpos"])
        if np.linalg.norm(qpos_diff) <= env.qpos_tol:
            print(f"  reached target at iter {it}")
            break
        move_dir = qpos_diff
        j0 = env._jd
        action = env.scale_action(move_dir, out_max=env.max_action)
        ctrl_ts = 1 / 40
        n_sub = int(np.ceil(ctrl_ts / env.dt))
        for s in range(1, n_sub + 1):
            jd = j0 + action * s / n_sub
            m = sim_step(env, jd)
            if m["nan"] or m["qvel"] > FAIL_QVEL:
                print(f"  FAIL move_to_pose iter={it} sub={s}: {m}")
                fail_step = m["step"]
                return fail_step
            if m["step"] <= 3:
                print(f"  early step{m['step']}: {m}")
        env._jd = j0 + action
    return fail_step


def probe_rot(model_name, n_rot_steps=30):
    print(f"\n## {model_name} ropeend_rot x{n_rot_steps}")
    env = get_model_specs([model_name])[model_name]["create_env"]("white", None, False)
    env.max_action = 0.02
    rot_step = np.pi / 180
    for i in range(n_rot_steps):
        rot_arr = np.zeros(3)
        rot_arr[0] = rot_step
        rot_quat = T.axisangle2quat(rot_arr)
        bid = env.ropeend_body_id
        env.model.body_quat[bid] = T.quat_multiply(rot_quat, env.model.body_quat[bid])
        # abbreviated hold: 5 sim steps
        for _ in range(5):
            m = sim_step(env, env.observations["qpos"])
            if m["nan"] or m["qvel"] > FAIL_QVEL:
                print(f"  FAIL rot step {i}: {m}")
                return m["step"]
        if i % 10 == 0:
            m = sim_step(env, env.observations["qpos"])
            print(f"  rot {i}: {m}")
    return None


def ablate_xpbd():
    print("\n## XPBD ablations")
    base = get_model_specs(["xpbd"])["xpbd"]["create_env"]("white", None, False)
    dlo = base.dlo_sim

    cases = [
        ("baseline", {}),
        ("k_force=0", {"k_force": 0.0}),
        ("k_torque=0.01", {"k_torque": 0.01}),
        ("k_force=k_torque=0.01", {"k_force": 0.01, "k_torque": 0.01}),
        ("num_iterations=1", {"num_iterations": 1}),
    ]
    for name, kwargs in cases:
        env = get_model_specs(["xpbd"])["xpbd"]["create_env"]("white", None, False)
        dlo = env.dlo_sim
        for k, v in kwargs.items():
            setattr(dlo, k, v)
            if k == "k_force":
                dlo.xpbd_math.setForceGain(v)
            if k == "k_torque":
                dlo.xpbd_math.setTorqueGain(v)
            if k == "num_iterations":
                dlo.xpbd_math.setNumIterations(v)
        m = sim_step(env)
        print(f"  {name}: step1 {m}")


def main():
    models = sys.argv[1].split(",") if len(sys.argv) > 1 else ["adapt", "xpbd", "geds"]
    for m in models:
        probe_idle(m, 10)
        probe_move_to_pose(m, 200)
        probe_rot(m, 20)
    if "xpbd" in models or len(sys.argv) == 1:
        ablate_xpbd()


if __name__ == "__main__":
    main()
