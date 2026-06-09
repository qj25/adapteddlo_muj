"""
Kirchhoff rod controller (JTill2017) with direct quasi-static elastic wrenches.

No shadow solve; MuJoCo integrates the articulated chain while this module
supplies bending/twist torques via qfrc_passive (adapt path).
"""

import numpy as np
import mujoco

import adapteddlo_muj.utils.transform_utils as T
import adapteddlo_muj.controllers.cosserat3_cpp.RodCosserat3 as RodCosserat3
import adapteddlo_muj.utils.mjc2_utils as mjc2


class DLORopeCosserat3:
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

        self._init_cosserat3_cpp()

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

    def _init_cosserat3_cpp(self):
        self._update_xvecs()
        self._init_resetbody_vars()
        self._x2e()
        self._init_bf()
        self.dlo_math = RodCosserat3.RodCosserat3(
            self.x.flatten(),
            self.bf0_bar.flatten(),
            self.p_thetan,
            self.overall_rot,
            self.alpha_bar,
            self.beta_bar,
            self.radius,
        )
        self._init_o2m()

    def _update_cosserat3_cpp(self):
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

    def set_dlosim(self, ropestart_pos, ropestart_quat, overall_rot, p_thetan):
        self.model.body_pos[self.ropestart_bodyid, :] = ropestart_pos
        self.model.body_quat[self.ropestart_bodyid, :] = ropestart_quat
        self.overall_rot = overall_rot
        self.p_thetan = p_thetan
        self.dlo_math.resetTheta(self.p_thetan, self.overall_rot)

    def reset_body(self):
        self.model.body_pos[self.vec_bodyid[:], :] = self.xpos_reset.copy()
        self.model.body_quat[self.vec_bodyid[:], :] = self.xquat_reset.copy()
        self._reset_vel()
        mujoco.mj_forward(self.model, self.data)

    def reset_sim(self):
        self.overall_rot = self.reset_rot
        self.p_thetan = self.reset_rot % (2.0 * np.pi)
        if self.p_thetan > np.pi:
            self.p_thetan -= 2 * np.pi
        self.dlo_math.resetTheta(self.p_thetan, self.overall_rot)
        self.dlo_math.captureRestCurvature()

    def change_ropestiffness(self, alpha_bar, beta_bar):
        self.alpha_bar = alpha_bar
        self.beta_bar = beta_bar
        self.dlo_math.changeAlphaBeta(self.alpha_bar, self.beta_bar)

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
        self._update_cosserat3_cpp()
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
