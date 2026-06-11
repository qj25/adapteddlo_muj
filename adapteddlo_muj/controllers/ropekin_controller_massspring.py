import numpy as np
import mujoco

import adapteddlo_muj.utils.mjc2_utils as mjc2
import adapteddlo_muj.controllers.massspring_cpp.MassSpring as MassSpring


class DLORopeMassSpring:
    def __init__(
        self,
        model,
        data,
        n_link,
        radius,
        overall_rot=0.0,
        alpha_bar=1.345 / 10,
        beta_bar=0.789 / 10,
        bothweld=True,
        f_limit=1000.0,
    ):
        self.model = model
        self.data = data
        self.bothweld = bothweld
        self.f_limit = f_limit
        self.radius = radius

        self.nv = n_link - 1
        self.vec_bodyid = np.zeros(self.nv + 2, dtype=int)
        self._init_sitebody()

        self.torq_node = np.zeros((self.nv + 2, 3))
        self.torq_node_flat = self.torq_node.flatten()
        self.neutral_quat_flat = np.zeros((self.nv + 2) * 4)

        self.reset_rot = overall_rot
        self.overall_rot = overall_rot
        self.p_thetan = 0.0

        self.alpha_bar = alpha_bar * 10.0
        self.beta_bar = beta_bar
        self._stiffness_from_alpha_beta()

        self.dlo_joint_ids = []
        self.dlo_joint_qveladdr = []
        for i in range(1, n_link):
            fj_str = f"J_{i}"
            if i == (n_link - 1):
                fj_str = "J_last"
            self.dlo_joint_ids.append(mjc2.obj_name2id(self.model, "joint", fj_str))
            self.dlo_joint_qveladdr.append(self.model.jnt_dofadr[self.dlo_joint_ids[-1]])
        self.dlo_joint_qveladdr = np.array(self.dlo_joint_qveladdr)
        self.qvel0_addr = np.min(self.dlo_joint_qveladdr)
        self.dlo_joint_qveladdr_full = [
            n for n in range(self.dlo_joint_qveladdr[0], self.dlo_joint_qveladdr[-1] + 3)
        ]
        self.qvellast_addr = np.max(self.dlo_joint_qveladdr_full)
        self.rotx_qveladdr = self.dlo_joint_qveladdr[:] + 2
        self.rxqva_len = len(self.rotx_qveladdr)

        self.n_cable = self.nv + 1
        self.cable_omega0 = np.zeros((self.n_cable, 3), dtype=np.float64)
        self.cable_stiffness = np.zeros((self.n_cable, 4), dtype=np.float64)
        self.k_twist = self._cable_twist_stiffness(self.beta_bar)
        self.k_twist = beta_bar
        self._init_resetbody_vars()
        self._init_massspring_cpp()

    def _stiffness_from_alpha_beta(self):
        self.k_bend_x = self.alpha_bar
        self.k_bend_y = self.alpha_bar

    def _init_sitebody(self):
        for i in range(self.nv + 2):
            ii_b = i
            if ii_b == (self.nv + 1):
                ii_b = "last2"
            if ii_b == self.nv:
                ii_b = "last"
            if ii_b == 0:
                ii_b = "first"
            self.vec_bodyid[i] = mjc2.obj_name2id(self.model, "body", f"B_{ii_b}")
        self.ropestart_bodyid = mjc2.obj_name2id(self.model, "body", "stiffrope")

    def _init_resetbody_vars(self):
        self.xpos_reset = self.model.body_pos[self.vec_bodyid[:]].copy()
        self.xquat_reset = self.model.body_quat[self.vec_bodyid[:]].copy()

    def set_resetbody_vars(self):
        self._init_resetbody_vars()

    def _cable_twist_stiffness(self, twist_g):
        """Match cable.cc stiffness[4*b+0] = J * G for capsule cross-section."""
        j_torsion = np.pi * self.radius ** 4 / 2.0
        return j_torsion * twist_g

    def _joint_quat_adr(self, body_id):
        jnt_id = self.model.body_jntadr[body_id]
        if jnt_id < 0:
            return None
        dofnum = self.model.body_dofnum[body_id]
        if dofnum < 3:
            return None
        return self.model.jnt_qposadr[jnt_id] + dofnum - 3

    def _cable_body_id(self, b):
        return int(self.vec_bodyid[b])

    def _quat_diff(self, quat, body_quat, joint_quat):
        """Match cable.cc QuatDiff with pullback=false."""
        mujoco.mju_mulQuat(quat, body_quat, joint_quat)

    def _local_stress(self, stress, stiffness, quat, omega0, pullback=False):
        """Match cable.cc LocalStress; only twist (index 0) is active."""
        omega = np.zeros(3, dtype=np.float64)
        tmp = np.zeros(3, dtype=np.float64)
        mujoco.mju_quat2Vel(omega, quat, 1.0)
        if stiffness[3] > 1e-12:
            tmp[0] = -stiffness[0] * (omega[0] - omega0[0]) / stiffness[3]
        if pullback:
            invquat = np.zeros(4, dtype=np.float64)
            mujoco.mju_negQuat(invquat, quat)
            mujoco.mju_rotVecQuat(stress, tmp, invquat)
        else:
            stress[:] = tmp

    def _capture_cable_seg_lens(self):
        """Match cable.cc constructor: stiffness[4*b+3] = dist(xpos[i], xpos[i-1])."""
        seg_len = np.zeros(self.n_cable, dtype=np.float64)
        for b in range(1, self.n_cable):
            i = self._cable_body_id(b)
            prev_i = self._cable_body_id(b - 1)
            seg_len[b] = np.linalg.norm(self.data.xpos[i] - self.data.xpos[prev_i])
        return seg_len

    def _capture_cable_twist_rest(self):
        """Match cable.cc constructor with flat=true: all omega0 zero (no mju_subQuat)."""
        self.cable_omega0[:] = 0.0
        for b in range(self.n_cable):
            self.cable_stiffness[b, 1] = 0.0
            self.cable_stiffness[b, 2] = 0.0
            self.cable_stiffness[b, 0] = self.k_twist
            self.cable_stiffness[b, 3] = (
                self._cable_rest_seg_len[b] if b > 0 else 0.0
            )

    def _apply_cable_twist_torques(self):
        """Match cable.cc Compute (twist component only)."""
        quat = np.zeros(4, dtype=np.float64)
        lfrc = np.zeros(3, dtype=np.float64)
        stress = np.zeros(3, dtype=np.float64)
        xfrc = np.zeros(3, dtype=np.float64)
        zero_force = np.zeros(3, dtype=np.float64)

        for b in range(self.n_cable):
            i = self._cable_body_id(b)
            stiffness_b = self.cable_stiffness[b]
            if stiffness_b[0] == 0.0:
                continue

            lfrc[:] = 0.0

            if b > 0:
                qadr = self._joint_quat_adr(i)
                if qadr is not None:
                    body_quat = self.model.body_quat[i]
                    joint_quat = self.data.qpos[qadr : qadr + 4]
                    self._quat_diff(quat, body_quat, joint_quat)
                    self._local_stress(
                        stress,
                        stiffness_b,
                        quat,
                        self.cable_omega0[b],
                        pullback=True,
                    )
                    lfrc += stress

            if b < self.n_cable - 1:
                bn = b + 1
                in_id = self._cable_body_id(bn)
                qadr = self._joint_quat_adr(in_id)
                if qadr is not None:
                    body_quat = self.model.body_quat[in_id]
                    joint_quat = self.data.qpos[qadr : qadr + 4]
                    self._quat_diff(quat, body_quat, joint_quat)
                    self._local_stress(
                        stress,
                        self.cable_stiffness[bn],
                        quat,
                        self.cable_omega0[bn],
                        pullback=False,
                    )
                    lfrc -= stress

            if np.linalg.norm(lfrc) < 1e-12:
                continue

            mujoco.mju_rotVecQuat(xfrc, lfrc, self.data.xquat[i])
            mujoco.mj_applyFT(
                self.model,
                self.data,
                zero_force,
                xfrc,
                self.data.xpos[i],
                i,
                self.data.qfrc_passive,
            )

    def _capture_neutral_quat(self):
        self.neutral_quat_flat = self.data.xquat[self.vec_bodyid[:]].flatten().copy()

    def _init_massspring_cpp(self):
        self._capture_neutral_quat()
        self.massspring_math = MassSpring.MassSpring(
            self.neutral_quat_flat,
            self.k_bend_x,
            self.k_bend_y,
        )
        self._cable_rest_seg_len = self._capture_cable_seg_lens()
        self._capture_cable_twist_rest()

    def reset_neutral(self):
        self._capture_neutral_quat()
        self.massspring_math.setNeutralQuat(self.neutral_quat_flat)

    def get_dlosim(self):
        ropestart_pos = self.model.body_pos[self.ropestart_bodyid, :].copy()
        ropestart_quat = self.model.body_quat[self.ropestart_bodyid, :].copy()
        return ropestart_pos, ropestart_quat, self.overall_rot, self.p_thetan

    def set_dlosim(self, ropestart_pos, ropestart_quat, overall_rot, p_thetan):
        self.model.body_pos[self.ropestart_bodyid, :] = ropestart_pos
        self.model.body_quat[self.ropestart_bodyid, :] = ropestart_quat
        self.overall_rot = overall_rot
        self.p_thetan = p_thetan
        mujoco.mj_forward(self.model, self.data)
        self.reset_neutral()
        self._cable_rest_seg_len = self._capture_cable_seg_lens()
        self._capture_cable_twist_rest()

    def reset_body(self):
        self.model.body_pos[self.vec_bodyid[:], :] = self.xpos_reset.copy()
        self.model.body_quat[self.vec_bodyid[:], :] = self.xquat_reset.copy()
        self._reset_vel()
        mujoco.mj_forward(self.model, self.data)
        self.reset_neutral()
        self._cable_rest_seg_len = self._capture_cable_seg_lens()
        self._capture_cable_twist_rest()

    def reset_sim(self):
        self.overall_rot = self.reset_rot
        self.p_thetan = 0.0
        self.reset_neutral()
        self._capture_cable_twist_rest()

    def change_ropestiffness(self, alpha_bar, beta_bar):
        self.alpha_bar = alpha_bar
        self.beta_bar = beta_bar
        self._stiffness_from_alpha_beta()
        self.k_twist = self._cable_twist_stiffness(self.beta_bar)
        self.k_twist = beta_bar
        self.massspring_math.setStiffness(self.k_bend_x, self.k_bend_y)
        self._capture_cable_twist_rest()

    def _calc_centerline_torq(self):
        body_xpos_flat = self.data.xpos[self.vec_bodyid[:]].flatten()
        body_quats_flat = self.data.xquat[self.vec_bodyid[:]].flatten()
        self.massspring_math.computeTorque(
            body_xpos_flat,
            body_quats_flat,
            self.torq_node_flat,
        )
        self.torq_node = self.torq_node_flat.reshape((self.nv + 2, 3))

    def _update_xvecs(self):
        # Compatibility shim for env state-reset paths that expect this call.
        return None

    def _reset_vel(self):
        self.data.qvel[self.dlo_joint_qveladdr_full] = np.zeros(self.rxqva_len * 3)

    def reset_qvel_rotx(self):
        self.data.qvel[self.rotx_qveladdr] = np.zeros(self.rxqva_len)

    def update_torque(self):
        self._calc_centerline_torq()
        # torq_node has one entry per body (nv+2); only indices 1..nv map to ball joints.
        # Index nv+1 is the rigid tip site B_last2 and is excluded from qfrc_passive.
        joint_torq = self.torq_node[1 : self.nv + 1]
        if self.bothweld:
            self.data.qfrc_passive[self.qvel0_addr - 3 : self.qvellast_addr + 1] += np.concatenate(
                (self.torq_node[0:1], joint_torq)
            ).flatten()
        else:
            self.data.qfrc_passive[self.qvel0_addr : self.qvellast_addr + 1] += joint_torq.flatten()
        self._apply_cable_twist_torques()
