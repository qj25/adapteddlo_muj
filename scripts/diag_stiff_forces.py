"""Diagnose stiffness forces for test_shape_w_arm moveid 7 with stall watchdog."""
import sys
import time
import traceback

import numpy as np

import adapteddlo_muj.utils.transform_utils as T
from adapteddlo_muj.envs.test_shape_w_arm.registry import get_model_specs
from adapteddlo_muj.utils.manipulation_config import (
    apply_move_settings,
    load_manipulation_config,
)

WIRE_COLOR = "black"
MOVE_ID = 7
STALL_SEC = 10.0
MODELS = sys.argv[1:] if len(sys.argv) > 1 else ["geds", "xpbd", "adapt"]

move_pos = np.array([[0.20, 0.0, 0.0]])
move_pos[:, 0] -= 0.035
move_aa = np.array([[0.0, 0.0, 0.0]])
z_rot = np.array([720.0 * np.pi / 180.0])
move_quat = np.array([T.axisangle2quat(move_aa[0])])

model_specs = get_model_specs(MODELS)


class StallError(RuntimeError):
    pass


class ProgressWatchdog:
    """Abort if a single step is slow or IK/error plateaus for STALL_SEC seconds."""

    def __init__(self, stall_sec=STALL_SEC):
        self.stall_sec = stall_sec
        self._best_ik_error = float("inf")
        self._best_ik_time = time.monotonic()
        self._last_env_steps = -1
        self._last_env_steps_time = time.monotonic()

    def touch_ik(self, ik_error):
        now = time.monotonic()
        if ik_error < self._best_ik_error - 1e-6:
            self._best_ik_error = ik_error
            self._best_ik_time = now
            return
        if now - self._best_ik_time > self.stall_sec:
            raise StallError(
                f"IK error plateaued at ~{ik_error:.4f} "
                f"(best={self._best_ik_error:.4f}) for {self.stall_sec:.0f}s"
            )

    def touch_env_step(self, env_steps):
        now = time.monotonic()
        if env_steps != self._last_env_steps:
            self._last_env_steps = env_steps
            self._last_env_steps_time = now
            return
        if now - self._last_env_steps_time > self.stall_sec:
            raise StallError(
                f"env_steps stuck at {env_steps} for {self.stall_sec:.0f}s"
            )

    def check_step_duration(self, dt, env_steps):
        if dt > self.stall_sec:
            raise StallError(
                f"single env.step at env_steps={env_steps} took {dt:.1f}s"
            )


def _install_monitors(env, stats, watchdog):
    dlo = env.dlo_sim
    orig_update = dlo.update_torque
    orig_step = env.step

    def monitored_update_torque(*args, **kwargs):
        out = orig_update(*args, **kwargs)
        forces = dlo.force_node.copy()
        torques_direct = getattr(
            dlo, "tau_roll", getattr(dlo, "tau_direct", np.zeros_like(forces))
        )
        torques_joint = dlo.torq_node.copy()

        force_norms = np.linalg.norm(forces, axis=1)
        tau_d_norms = np.linalg.norm(torques_direct, axis=1)
        tau_j_norms = np.linalg.norm(torques_joint, axis=1)
        total_force = float(np.linalg.norm(forces))

        qv = env.data.qvel[dlo.dlo_joint_qveladdr_full]
        qvel_n = float(np.linalg.norm(qv))
        rope_pos = env.data.site_xpos[dlo.vec_siteid[:]]
        pos_spread = float(np.max(rope_pos) - np.min(rope_pos))
        has_nan = bool(np.isnan(env.data.qpos).any() or np.isnan(env.data.qvel).any())
        limited = total_force > dlo.f_limit

        stats["steps"] += 1
        stats["max_force_node"] = max(stats["max_force_node"], float(force_norms.max()))
        stats["max_tau_direct"] = max(stats["max_tau_direct"], float(tau_d_norms.max()))
        stats["max_tau_joint"] = max(stats["max_tau_joint"], float(tau_j_norms.max()))
        stats["max_total_force"] = max(stats["max_total_force"], total_force)
        stats["limit_hits"] += int(limited)
        stats["max_qvel"] = max(stats["max_qvel"], qvel_n)
        stats["max_pos_spread"] = max(stats["max_pos_spread"], pos_spread)
        stats["nan"] = stats["nan"] or has_nan

        watchdog.touch_env_step(env.env_steps)

        if limited or has_nan or qvel_n > 20.0 or force_norms.max() > 50.0:
            stats["events"].append(
                {
                    "phase": stats.get("phase", "?"),
                    "step": env.env_steps,
                    "max_force": float(force_norms.max()),
                    "worst_node": int(force_norms.argmax()),
                    "total_force": total_force,
                    "limited": limited,
                    "max_tau_direct": float(tau_d_norms.max()),
                    "max_tau_joint": float(tau_j_norms.max()),
                    "qvel_norm": qvel_n,
                    "nan": has_nan,
                }
            )
        return out

    def monitored_step(action=np.zeros(6)):
        t0 = time.monotonic()
        out = orig_step(action)
        watchdog.check_step_duration(time.monotonic() - t0, env.env_steps)
        return out

    dlo.update_torque = monitored_update_torque
    env.step = monitored_step

    orig_move = env.move_to_qpos_good

    def monitored_move_to_qpos_good(targ_qpos):
        outer = 0
        last_err = None
        while True:
            qpos_diff = env.joint_sum(targ_qpos, -env.observations["qpos"])
            err = float(np.linalg.norm(qpos_diff))
            stats["ik_error"] = err
            if err <= env.qpos_tol:
                return
            outer += 1
            watchdog.touch_ik(err)
            last_err = err

            move_dir = qpos_diff
            j0 = env._jd
            action = env.scale_action(move_dir, out_max=env.max_action)
            ctrl_ts = 1 / load_manipulation_config()["control_freq_hz"]
            dyn_ts = env.model.opt.timestep
            steps = 0
            print(f"error = {err}")
            while steps < ctrl_ts / dyn_ts:
                steps += 1
                jd = j0 + (action * steps / np.ceil(ctrl_ts / dyn_ts))
                env.step(action=jd)

    env.move_to_qpos_good = monitored_move_to_qpos_good


def run_with_monitor(model_name):
    spec = model_specs[model_name]
    env = spec["create_env"](WIRE_COLOR, None, do_render=False)
    apply_move_settings(env, "first_move_to_pose")
    watchdog = ProgressWatchdog(STALL_SEC)

    stats = {
        "model": model_name,
        "steps": 0,
        "max_force_node": 0.0,
        "max_tau_direct": 0.0,
        "max_tau_joint": 0.0,
        "max_total_force": 0.0,
        "limit_hits": 0,
        "max_qvel": 0.0,
        "max_pos_spread": 0.0,
        "nan": False,
        "events": [],
        "f_limit": env.dlo_sim.f_limit,
        "phase": "init",
        "ik_error": -1.0,
        "stalled": False,
        "stall_reason": None,
    }
    _install_monitors(env, stats, watchdog)

    desired_pos = env.init_pos + move_pos[0]
    desired_quat = T.quat_multiply(env.init_quat, move_quat[0])

    try:
        stats["phase"] = "move_to_pose_1"
        env.move_to_pose(desired_pos, desired_quat)

        stats["phase"] = "rot_x_rads"
        env.rot_x_rads(z_rot[0])

        stats["phase"] = "move_to_pose_2"
        env.move_to_pose(desired_pos, desired_quat)

        stats["phase"] = "hold_pos"
        env.hold_pos(5.0)
    except StallError as exc:
        stats["stalled"] = True
        stats["stall_reason"] = str(exc)
        stats["stall_env_steps"] = env.env_steps
        stats["stall_ik_error"] = stats.get("ik_error")
        print(f"  [STALL] {exc}")

    return stats


def _print_summary(stats):
    print(f"  f_limit:              {stats['f_limit']}")
    print(f"  final phase:          {stats['phase']}")
    print(f"  stalled:              {stats['stalled']}")
    if stats["stalled"]:
        print(f"  stall reason:         {stats['stall_reason']}")
        print(f"  stall env_steps:      {stats.get('stall_env_steps')}")
        print(f"  stall ik_error:       {stats.get('stall_ik_error')}")
    print(f"  steps monitored:      {stats['steps']}")
    print(f"  max per-node force:   {stats['max_force_node']:.4f} N")
    print(f"  max direct torque:    {stats['max_tau_direct']:.4f} N·m")
    print(f"  max joint torque:     {stats['max_tau_joint']:.4f} N·m")
    print(f"  max total force norm: {stats['max_total_force']:.4f} N")
    print(f"  limiter hits:         {stats['limit_hits']}")
    print(f"  max rope qvel norm:   {stats['max_qvel']:.4f}")
    print(f"  max pos spread:       {stats['max_pos_spread']:.4f}")
    print(f"  NaN detected:         {stats['nan']}")
    if stats["events"]:
        print(f"  notable events ({len(stats['events'])}):")
        for ev in stats["events"][:10]:
            print(
                f"    step {ev['step']:5d} [{ev['phase']:16s}] "
                f"F_max={ev['max_force']:8.2f}@node{ev['worst_node']} "
                f"F_tot={ev['total_force']:8.2f} lim={ev['limited']} "
                f"tau_d={ev['max_tau_direct']:8.2f} tau_j={ev['max_tau_joint']:8.2f} "
                f"qvel={ev['qvel_norm']:8.2f} nan={ev['nan']}"
            )


if __name__ == "__main__":
    all_stats = []
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model}  wire={WIRE_COLOR}  moveid={MOVE_ID}")
        print(f"{'='*60}")
        try:
            stats = run_with_monitor(model)
            _print_summary(stats)
            all_stats.append(stats)
        except Exception:
            print(f"  [CRASH] unhandled exception for {model}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    for s in all_stats:
        print(
            f"  {s['model']:8s} stalled={s['stalled']} "
            f"phase={s['phase']:16s} "
            f"F_max={s['max_force_node']:8.2f} "
            f"tau_j_max={s['max_tau_joint']:8.2f} "
            f"qvel_max={s['max_qvel']:8.2f} "
            f"limit_hits={s['limit_hits']}"
        )
