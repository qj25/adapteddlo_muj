"""
Kirchhoff rod controller (JTill2017) with cosserat3 bending and cable-style twist.

Bending uses the Kirchhoff C++ evaluator; twist stiffness follows the MuJoCo
cable plugin (quaternion joint error + mju_quat2Vel + mj_applyFT).
"""

import numpy as np
import mujoco

import adapteddlo_muj.utils.transform_utils as T
import adapteddlo_muj.controllers.cosserat5_cpp.RodCosserat5 as RodCosserat5
import adapteddlo_muj.utils.mjc2_utils as mjc2


class DLORopeCosserat5:
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
        incl_prevf=False,
    ):
        self.model = model
        self.data = data
        self.bothweld = bothweld
        self.radius = radius

        self.d_vec = 0
        self.nv = n_link - 1 - self.d_vec * 2
        self.vec_siteid = np.zeros(self.nv + 2, dtype=int)
        self.vec_bodyid = np.zeros(self.nv + 2, dtype=int)
        self.link_bodyid = np.zeros(self.nv + 1, dtype=int)

        self._init_sitebody()

        self.x = np.zeros((self.nv + 2, 3))
        self.force_node = np.zeros((self.nv + 2, 3))
        self.torq_node = np.zeros((self.nv + 2, 3))
        self.incl_prevf = incl_prevf
        if self.incl_prevf:
            self.force_node_prev = np.zeros((self.nv + 2, 3))
            self.prevf_ratio = 0.1
        self.force_node_flat = self.force_node.flatten()
        self.torq_node_flat = self.torq_node.flatten()

        self.reset_rot = overall_rot
        self.overall_rot = self.reset_rot
        self.p_thetan = self.reset_rot % (2.0 * np.pi)
        if self.p_thetan > np.pi:
            self.p_thetan -= 2 * np.pi
        self.theta_displace = self.p_thetan

        self.e_bar = np.zeros(self.nv + 1)
        self.alpha_bar = alpha_bar
        self.beta_bar = beta_bar
        self.f_limit = f_limit

        self.bf0_bar = np.zeros((3, 3))
        self.bf_end = np.zeros((3, 3))
        self.bf_end_flat = self.bf_end.flatten()

        self.dlo_joint_ids = []
        self.dlo_joint_qveladdr = []
        for i in range(1, n_link):
            fj_str = f"J_{i}"
            if i == (n_link - 1):
                fj_str = "J_last"
            self.dlo_joint_ids.append(
                mjc2.obj_name2id(self.model, "joint", fj_str)
            )
            self.dlo_joint_qveladdr.append(
                self.model.jnt_dofadr[self.dlo_joint_ids[-1]]
            )
        self.dlo_joint_qveladdr = np.array(self.dlo_joint_qveladdr)
        self.qvel0_addr = np.min(self.dlo_joint_qveladdr)
        self.dlo_joint_qveladdr_full = [
            n for n in range(
                self.dlo_joint_qveladdr[0],
                self.dlo_joint_qveladdr[-1] + 3,
            )
        ]
        self.qvellast_addr = np.max(self.dlo_joint_qveladdr_full)
        self.rotx_qveladdr = self.dlo_joint_qveladdr[:] + 2
        self.rxqva_len = len(self.rotx_qveladdr)

        n_nodes = self.nv + 2
        self.n_cable = self.nv + 1
        self.cable_omega0 = np.zeros((self.n_cable, 3), dtype=np.float64)
        self.cable_stiffness = np.zeros((self.n_cable, 4), dtype=np.float64)
        self.k_twist = beta_bar
        self._init_cosserat5_cpp()

    def _init_sitebody(self):
        for i in range(self.d_vec, self.nv + 2 + self.d_vec):
            ii = i
            ii_s = ii
            ii_b = ii
            if ii == (self.nv + 1):
                ii_s = "last"
                ii_b = "last2"
            if ii == self.nv:
                ii_b = "last"
            if ii == 0:
                ii_b = "first"
            self.vec_siteid[i - self.d_vec] = mjc2.obj_name2id(
                self.model, "site", "S_{}".format(ii_s)
            )
            self.vec_bodyid[i - self.d_vec] = mjc2.obj_name2id(
                self.model, "body", "B_{}".format(ii_b)
            )
        self.ropestart_bodyid = mjc2.obj_name2id(self.model, "body", "stiffrope")
        self.startsec_site = mjc2.obj_name2id(self.model, "site", "S_first")
        self.endsec_site = mjc2.obj_name2id(self.model, "site", "S_last")

    def _update_xvecs(self):
        self.x = self.data.site_xpos[self.vec_siteid[:]].copy()

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
        # print(self.data.qfrc_passive.reshape((self.nv, 3)))

    def _init_cosserat5_cpp(self):
        self._update_xvecs()
        self._init_resetbody_vars()
        self._x2e()
        self._init_bf()
        self.dlo_math = RodCosserat5.RodCosserat5(
            self.x.flatten(),
            self.bf0_bar.flatten(),
            self.p_thetan,
            self.overall_rot,
            self.alpha_bar,
            self.beta_bar,
            self.radius,
        )
        self._init_o2m()
        self._cable_rest_seg_len = self._capture_cable_seg_lens()
        self._capture_cable_twist_rest()

    def _update_cosserat5_cpp(self):
        self._update_xvecs()
        self._update_bishf_S()
        bf_align = self.dlo_math.updateVars(
            self.x.flatten(),
            self.bf0_bar.flatten(),
            self.bf_end_flat,
        )
        self.bf_end = self.bf_end_flat.reshape((3, 3))
        self.p_thetan = self._get_thetan()
        self.overall_rot = self.dlo_math.updateTheta(self.p_thetan)
        return bf_align

    def get_dlosim(self):
        ropestart_pos = self.model.body_pos[self.ropestart_bodyid, :].copy()
        ropestart_quat = self.model.body_quat[self.ropestart_bodyid, :].copy()
        return ropestart_pos, ropestart_quat, self.overall_rot, self.p_thetan

    def recapture_rest(self):
        """Capture Kirchhoff rest curvature and flat cable rest (omega0=0)."""
        mujoco.mj_forward(self.model, self.data)
        self._update_xvecs()
        self.dlo_math.captureRestCurvature()
        self._cable_rest_seg_len = self._capture_cable_seg_lens()
        self._capture_cable_twist_rest()

    def set_dlosim(self, ropestart_pos, ropestart_quat, overall_rot, p_thetan):
        self.overall_rot = overall_rot
        self.p_thetan = p_thetan
        self.dlo_math.resetTheta(self.p_thetan, self.overall_rot)

    def reset_body(self):
        self.model.body_pos[self.vec_bodyid[:], :] = self.xpos_reset.copy()
        self.model.body_quat[self.vec_bodyid[:], :] = self.xquat_reset.copy()
        self._reset_vel()
        self.recapture_rest()

    def reset_sim(self):
        self.overall_rot = self.reset_rot
        self.p_thetan = self.reset_rot % (2.0 * np.pi)
        if self.p_thetan > np.pi:
            self.p_thetan -= 2 * np.pi
        self.dlo_math.resetTheta(self.p_thetan, self.overall_rot)
        self.recapture_rest()

    def change_ropestiffness(self, alpha_bar, beta_bar):
        self.alpha_bar = alpha_bar
        self.beta_bar = beta_bar
        self.k_twist = beta_bar
        self.dlo_math.changeAlphaBeta(self.alpha_bar, self.beta_bar)
        self._capture_cable_twist_rest()

    def _x2e(self):
        self.e = np.zeros((self.nv + 1, 3))
        self.e[0] = self.x[1] - self.x[0]
        self.e_bar[0] = np.linalg.norm(self.e[0])
        for i in range(1, self.nv + 1):
            self.e[i] = self.x[i + 1] - self.x[i]
            self.e_bar[i] = np.linalg.norm(self.e[i])

    def _init_bf(self):
        parll_tol = 1e-6
        self.bf0_bar[0, :] = self.e[0] / self.e_bar[0]
        self.bf0_bar[1, :] = np.cross(self.bf0_bar[0, :], np.array([0, 0, 1.0]))
        if np.linalg.norm(self.bf0_bar[1, :]) < parll_tol:
            self.bf0_bar[1, :] = np.cross(self.bf0_bar[0, :], np.array([0, 1.0, 0]))
        self.bf0_bar[1, :] /= np.linalg.norm(self.bf0_bar[1, :])
        self.bf0_bar[2, :] = np.cross(self.bf0_bar[0, :], self.bf0_bar[1, :])

    def _update_bishf_S(self):
        mat_res = np.zeros(9)
        self.dlo_math.calculateOf2Mf(
            self.data.site_xmat[self.startsec_site],
            mat_res,
        )
        self.bf0_bar = np.transpose(mat_res.reshape((3, 3)))

    def _init_loc_rotframe(self, q1, q2):
        qe = T.axisangle2quat(T.quat_error(q1, q2))
        q1_inv = T.quat_inverse(q1)
        return T.quat_multiply(T.quat_multiply(q1_inv, qe), q1)

    def _init_o2m(self):
        q_o0 = T.mat2quat(
            self.data.site_xmat[self.startsec_site].reshape((3, 3))
        )
        q_b0 = T.mat2quat(np.transpose(self.bf0_bar))
        self.qe_o2m_loc = self._init_loc_rotframe(q_o0, q_b0)
        self.qe_m2o_loc = self._init_loc_rotframe(q_b0, q_o0)
        self.dlo_math.initQe_o2m_loc(self.qe_o2m_loc)

    def _get_thetan(self):
        mat_on = self.data.site_xmat[self.endsec_site]
        mat_bn = np.transpose(self.bf_end)
        mat_mn = np.zeros(9)
        self.dlo_math.calculateOf2Mf(mat_on, mat_mn)
        mat_mn = mat_mn.reshape((3, 3))
        theta_n = (
            self.dlo_math.angBtwn3(mat_bn[:, 1], mat_mn[:, 1], mat_bn[:, 0])
            + self.theta_displace
        )
        return theta_n

    def _calc_centerlineTorq(self, excl_joints):
        body_quats_flat = self.data.xquat[self.vec_bodyid[:]].flatten()
        self.dlo_math.calculateCenterlineTorq(
            self.torq_node_flat,
            body_quats_flat,
            excl_joints,
        )
        self.torq_node = self.torq_node_flat.reshape((self.nv + 2, 3))

    def _reset_vel(self):
        self.data.qvel[self.dlo_joint_qveladdr_full] = np.zeros(self.rxqva_len * 3)

    def reset_qvel_rotx(self):
        self.data.qvel[self.rotx_qveladdr] = np.zeros(self.rxqva_len)

    def update_torque(self):
        self._update_cosserat5_cpp()
        excl_joints = 0
        self._calc_centerlineTorq(excl_joints=excl_joints)
        if self.bothweld:
            self.data.qfrc_passive[
                self.qvel0_addr - 3 : self.qvellast_addr + 1
            ] += self.torq_node[:-1].flatten()
        else:
            self.data.qfrc_passive[
                self.qvel0_addr : self.qvellast_addr + 1
            ] += self.torq_node[1:-1].flatten()

        self._apply_cable_twist_torques()
