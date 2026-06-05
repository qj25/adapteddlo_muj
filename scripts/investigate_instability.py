#!/usr/bin/env python3
"""Headless diagnostics: adapt vs xpbd vs geds instability on moveid 2."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import adapteddlo_muj.utils.transform_utils as T
from adapteddlo_muj.envs.test_shape_w_arm.base import R_PIECES, run_manipulation
from adapteddlo_muj.envs.test_shape_w_arm.registry import get_model_specs

MOVE_POS = np.array([[0.165, 0.1, 0.1]])
MOVE_AA = np.array([[50.0, 10.0, 0.0]]) * np.pi / 180
Z_ROT = np.array([360.0]) * np.pi / 180
MOVE_QUAT = np.array([T.axisangle2quat(MOVE_AA[0])])

FAIL_QVEL = 50.0
FAIL_QFRCPASSIVE = 1e4
FAIL_QPOS = 100.0


class PhaseTracker:
    def __init__(self):
        self.phase = "init"
        self.step_in_phase = 0

    def set_phase(self, name):
        self.phase = name
        self.step_in_phase = 0


def _rope_qvel_slice(env):
    return env.data.qvel[env.joint_qveladdr_full]


def _rope_qfrc_passive_slice(env):
    return env.data.qfrc_passive[env.joint_qveladdr_full]


def _site_body_diff(env, dlo):
    xpos = env.data.xpos[dlo.vec_bodyid[:]]
    site = env.data.site_xpos[dlo.vec_siteid[:]]
    return np.max(np.linalg.norm(xpos - site, axis=1))


def _log_wrenches(dlo, rope_type):
    out = {}
    if hasattr(dlo, "force_node"):
        out["max_force_node"] = float(np.max(np.linalg.norm(dlo.force_node, axis=1)))
    if hasattr(dlo, "tau_direct"):
        out["max_tau_direct"] = float(np.max(np.linalg.norm(dlo.tau_direct, axis=1)))
    if hasattr(dlo, "tau_roll"):
        out["max_tau_roll"] = float(np.max(np.linalg.norm(dlo.tau_roll, axis=1)))
    if hasattr(dlo, "torq_node"):
        out["max_torq_node"] = float(np.max(np.linalg.norm(dlo.torq_node, axis=1)))
    return out


def instrument_env(env, model_name):
    tracker = PhaseTracker()
    logs = []
    orig_step = env.step
    orig_hold_pos = env.hold_pos
    orig_move_to_pose = env.move_to_pose
    orig_rot_x_rads = env.rot_x_rads
    orig_ropeend_rot = env.ropeend_rot
    orig_move_to_qpos_good = env.move_to_qpos_good
    dlo = env.dlo_sim

    def check_fail(step, phase):
        qvel = _rope_qvel_slice(env)
        qfrc = _rope_qfrc_passive_slice(env)
        qpos = env.data.qpos[env.joint_qposids]
        max_qvel = float(np.max(np.abs(qvel))) if len(qvel) else 0.0
        max_qfrc = float(np.max(np.abs(qfrc))) if len(qfrc) else 0.0
        max_qpos = float(np.max(np.abs(qpos)))
        nan = bool(np.any(np.isnan(qvel)) or np.any(np.isnan(qfrc)) or np.any(np.isnan(qpos)))
        fail = (
            nan
            or max_qvel > FAIL_QVEL
            or max_qfrc > FAIL_QFRCPASSIVE
            or max_qpos > FAIL_QPOS
        )
        return fail, max_qvel, max_qfrc, max_qpos, nan

    def wrapped_step(action=np.zeros(6)):
        env.grav_comp()
        env.data.ctrl[env.joint_ids] = action
        wrenches_before = {}
        if env.rope_type in ("adapt", "geds", "xpbd"):
            if env.rope_type == "xpbd":
                dlo._calc_wrenches()
                wrenches_before = _log_wrenches(dlo, model_name)
                dlo._forces_to_torques()
                if dlo.bothweld:
                    env.data.qfrc_passive[dlo.qvel0_addr - 3 : dlo.qvellast_addr + 1] += dlo.torq_node[:-1].flatten()
                else:
                    env.data.qfrc_passive[dlo.qvel0_addr : dlo.qvellast_addr + 1] += dlo.torq_node[1:-1].flatten()
            elif env.rope_type == "geds":
                dlo._calc_elastic_wrenches()
                wrenches_before = _log_wrenches(dlo, model_name)
                dlo._forces_to_torques()
                dlo._apply_endpoint_pd(None, None)
                env.data.qfrc_passive[dlo.qvel0_addr : dlo.qvellast_addr + 1] += dlo.torq_node[1:-1].flatten()
            elif env.rope_type == "adapt":
                dlo._update_dlo_cpp()
                dlo._calc_centerlineTorq(excl_joints=0)
                wrenches_before = _log_wrenches(dlo, model_name)
                env.data.qfrc_passive[dlo.qvel0_addr : dlo.qvellast_addr + 1] += dlo.torq_node[1:-1].flatten()
        else:
            orig_step(action)
            return env._get_observations(), 0, env.env_steps > env.max_env_steps, False, 0

        env.sim.step()
        env.sim.forward()
        env.cur_time += env.dt
        env.env_steps += 1
        env._get_observations()

        site_diff = _site_body_diff(env, dlo)
        fail, max_qvel, max_qfrc, max_qpos, nan = check_fail(env.env_steps, tracker.phase)
        entry = {
            "step": env.env_steps,
            "phase": tracker.phase,
            "step_in_phase": tracker.step_in_phase,
            "site_body_diff": site_diff,
            "max_qvel": max_qvel,
            "max_qfrc_passive": max_qfrc,
            "max_qpos_arm": max_qpos,
            "nan": nan,
            "fail": fail,
            **wrenches_before,
        }
        logs.append(entry)
        tracker.step_in_phase += 1

        if fail and not getattr(env, "_fail_logged", False):
            env._fail_logged = True
            env._fail_step = env.env_steps
            env._fail_phase = tracker.phase
            print(f"  FAIL at step {env.env_steps} phase={tracker.phase}: qvel={max_qvel:.3g} qfrc={max_qfrc:.3g} nan={nan}")

        done = env.env_steps > env.max_env_steps
        return env._get_observations(), 0, done, False, 0

    def wrapped_move_to_pose(targ_pos=None, targ_quat=None):
        tracker.set_phase("move_to_pose")
        return orig_move_to_pose(targ_pos, targ_quat)

    def wrapped_rot_x_rads(x_rads):
        tracker.set_phase("rot_x_rads")
        return orig_rot_x_rads(x_rads)

    def wrapped_hold_pos(hold_time=2.0):
        tracker.set_phase("hold_pos")
        return orig_hold_pos(hold_time)

    def wrapped_ropeend_rot(rot_a=np.pi / 180, rot_axis=0):
        tracker.set_phase("ropeend_rot")
        return orig_ropeend_rot(rot_a, rot_axis)

    env.step = wrapped_step
    env.move_to_pose = wrapped_move_to_pose
    env.rot_x_rads = wrapped_rot_x_rads
    env.hold_pos = wrapped_hold_pos
    env.ropeend_rot = wrapped_ropeend_rot
    return logs, tracker


def summarize_logs(model_name, logs, env):
    if not logs:
        print(f"{model_name}: no logs")
        return
    fail_entries = [e for e in logs if e["fail"]]
    first = logs[0]
    print(f"\n=== {model_name} ===")
    print(f"  steps logged: {len(logs)}")
    print(f"  step0: force={first.get('max_force_node', 'n/a')} tau_direct={first.get('max_tau_direct', 'n/a')} "
          f"tau_roll={first.get('max_tau_roll', 'n/a')} torq={first.get('max_torq_node', 'n/a')} "
          f"site_diff={first['site_body_diff']:.6f} qvel={first['max_qvel']:.4g}")
    if fail_entries:
        fe = fail_entries[0]
        print(f"  FIRST FAIL: step={fe['step']} phase={fe['phase']} qvel={fe['max_qvel']:.4g} "
              f"qfrc={fe['max_qfrc_passive']:.4g} force={fe.get('max_force_node', 'n/a')} "
              f"torq={fe.get('max_torq_node', 'n/a')}")
    else:
        print("  completed without fail threshold")
    # peak per phase
    phases = sorted(set(e["phase"] for e in logs))
    for ph in phases:
        ph_logs = [e for e in logs if e["phase"] == ph]
        peak_qvel = max(e["max_qvel"] for e in ph_logs)
        peak_force = max(e.get("max_force_node", 0) for e in ph_logs)
        peak_torq = max(e.get("max_torq_node", 0) for e in ph_logs)
        print(f"  phase {ph}: n={len(ph_logs)} peak_qvel={peak_qvel:.4g} peak_force={peak_force:.4g} peak_torq={peak_torq:.4g}")


def run_model(model_name):
    print(f"\n--- Running {model_name} ---")
    spec = get_model_specs([model_name])[model_name]
    env = spec["create_env"]("white", None, False)
    env.max_action = 0.02
    logs, tracker = instrument_env(env, model_name)

    # log step 0 (before manipulation)
    env.grav_comp()
    if model_name == "xpbd":
        env.dlo_sim._calc_wrenches()
        w0 = _log_wrenches(env.dlo_sim, model_name)
        print(f"  pre-step0 wrenches: {w0}")
        print(f"  pre-step0 site_body_diff: {_site_body_diff(env, env.dlo_sim):.6f}")

    try:
        run_manipulation(env, MOVE_POS, MOVE_QUAT, Z_ROT, 0, getting_jointpos=False)
    except Exception as exc:
        print(f"  EXCEPTION: {exc}")
    summarize_logs(model_name, logs, env)
    return logs


def main():
    models = ["adapt", "xpbd", "geds"]
    if len(sys.argv) > 1:
        models = sys.argv[1].split(",")
    all_logs = {}
    for m in models:
        all_logs[m] = run_model(m)
    print("\n=== COMPARISON (first 5 steps) ===")
    for m in models:
        logs = all_logs[m]
        if logs:
            for e in logs[:5]:
                print(f"  {m} step{e['step']}: ph={e['phase']} f={e.get('max_force_node',0):.4g} "
                      f"td={e.get('max_tau_direct',0):.4g} tr={e.get('max_tau_roll',0):.4g} "
                      f"tq={e.get('max_torq_node',0):.4g} qv={e['max_qvel']:.4g}")


if __name__ == "__main__":
    main()
