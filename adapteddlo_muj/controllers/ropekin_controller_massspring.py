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

        self.nv = n_link - 1
        self.vec_bodyid = np.zeros(self.nv + 2, dtype=int)
        self._init_sitebody()

        self.torq_node = np.zeros((self.nv + 2, 3))
        self.torq_node_flat = self.torq_node.flatten()
        self.neutral_quat_flat = np.zeros((self.nv + 2) * 4)

        self.reset_rot = overall_rot
        self.overall_rot = overall_rot
        self.p_thetan = 0.0

        self.alpha_bar = alpha_bar
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

        self._init_resetbody_vars()
        self._init_massspring_cpp()

    def _stiffness_from_alpha_beta(self):
        self.k_bend_x = self.alpha_bar
        self.k_bend_y = self.alpha_bar
        self.k_twist = self.beta_bar

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

    def _capture_neutral_quat(self):
        self.neutral_quat_flat = self.data.xquat[self.vec_bodyid[:]].flatten().copy()

    def _init_massspring_cpp(self):
        self._capture_neutral_quat()
        self.massspring_math = MassSpring.MassSpring(
            self.neutral_quat_flat,
            self.k_bend_x,
            self.k_bend_y,
            self.k_twist,
        )

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

    def reset_body(self):
        self.model.body_pos[self.vec_bodyid[:], :] = self.xpos_reset.copy()
        self.model.body_quat[self.vec_bodyid[:], :] = self.xquat_reset.copy()
        self._reset_vel()
        mujoco.mj_forward(self.model, self.data)
        self.reset_neutral()

    def reset_sim(self):
        self.overall_rot = self.reset_rot
        self.p_thetan = 0.0
        self.reset_neutral()

    def change_ropestiffness(self, alpha_bar, beta_bar):
        self.alpha_bar = alpha_bar
        self.beta_bar = beta_bar
        self._stiffness_from_alpha_beta()
        self.massspring_math.setStiffness(self.k_bend_x, self.k_bend_y, self.k_twist)

    def _calc_centerline_torq(self):
        body_quats_flat = self.data.xquat[self.vec_bodyid[:]].flatten()
        self.massspring_math.computeTorque(body_quats_flat, self.torq_node_flat)
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
        if self.bothweld:
            self.data.qfrc_passive[self.qvel0_addr - 3 : self.qvellast_addr + 1] += self.torq_node[
                :-1
            ].flatten()
        else:
            self.data.qfrc_passive[self.qvel0_addr : self.qvellast_addr + 1] += self.torq_node[
                1:-1
            ].flatten()
