import numpy as np
import mujoco

import adapteddlo_muj.utils.mjc2_utils as mjc2
import adapteddlo_muj.controllers.cosserat_cpp.RodCosserat as RodCosserat


class DLORopeCosserat:
    def __init__(
        self,
        model,
        data,
        n_link,
        segment_length,
        radius,
        overall_rot=0.0,
        alpha_bar=1.345 / 10,
        beta_bar=0.789 / 10,
        bothweld=True,
        f_limit=1000.0,
        num_iters=2,
        k_torque=1.0,
    ):
        self.model = model
        self.data = data
        self.bothweld = bothweld
        self.f_limit = f_limit
        self.segment_length = segment_length
        self.radius = radius
        self.num_iters = num_iters
        self.k_torque = k_torque

        self.nv = n_link - 1
        self.vec_siteid = np.zeros(self.nv + 2, dtype=int)
        self.vec_bodyid = np.zeros(self.nv + 2, dtype=int)
        self._init_sitebody()

        self.torq_node = np.zeros((self.nv + 2, 3))
        self.torq_node_flat = self.torq_node.flatten()
        self.x_flat = np.zeros((self.nv + 2) * 3)
        self.quat_flat = np.zeros((self.nv + 2) * 4)

        self.reset_rot = overall_rot
        self.overall_rot = overall_rot
        self.p_thetan = 0.0

        self.alpha_bar = alpha_bar
        self.beta_bar = beta_bar
        self._material_from_alpha_beta()

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
        self._init_cosserat_cpp()

    def _material_from_alpha_beta(self):
        # Match mass-spring scaling: alpha_bar/beta_bar directly as bend/twist moduli.
        self.k_bend = self.alpha_bar
        self.k_twist = self.beta_bar

    def _init_sitebody(self):
        for i in range(self.nv + 2):
            ii_s = i
            ii_b = i
            if ii_s == (self.nv + 1):
                ii_s = "last"
                ii_b = "last2"
            elif ii_s == self.nv:
                ii_s = "last"
                ii_b = "last"
            elif ii_s == 0:
                ii_s = "first"
                ii_b = "first"
            self.vec_siteid[i] = mjc2.obj_name2id(self.model, "site", f"S_{ii_s}")
            self.vec_bodyid[i] = mjc2.obj_name2id(self.model, "body", f"B_{ii_b}")
        self.ropestart_bodyid = mjc2.obj_name2id(self.model, "body", "stiffrope")

    def _init_resetbody_vars(self):
        self.xpos_reset = self.model.body_pos[self.vec_bodyid[:]].copy()
        self.xquat_reset = self.model.body_quat[self.vec_bodyid[:]].copy()

    def set_resetbody_vars(self):
        self._init_resetbody_vars()

    def _capture_neutral_state(self):
        self.x_flat = self.data.site_xpos[self.vec_siteid[:]].flatten().copy()
        self.quat_flat = self.data.xquat[self.vec_bodyid[:]].flatten().copy()

    def _init_cosserat_cpp(self):
        self._capture_neutral_state()
        self.cosserat_math = RodCosserat.RodCosserat(
            self.nv + 2,
            self.segment_length,
            self.k_bend,
            self.k_twist,
        )
        self.cosserat_math.setNumIterations(self.num_iters)
        self.cosserat_math.setTorqueGain(self.k_torque)
        self.cosserat_math.reinitRest(
            self.x_flat,
            self.quat_flat,
        )

    def reset_neutral(self):
        self._capture_neutral_state()
        self.cosserat_math.reinitRest(
            self.x_flat,
            self.quat_flat,
        )

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
        self._material_from_alpha_beta()
        self.cosserat_math.setMaterial(self.k_bend, self.k_twist)

    def _limit_torques(self):
        torq_mag = np.linalg.norm(self.torq_node)
        if torq_mag > self.f_limit:
            self.torq_node *= self.f_limit / torq_mag

    def _calc_centerline_torq(self):
        self.x_flat = self.data.site_xpos[self.vec_siteid[:]].flatten()
        self.quat_flat = self.data.xquat[self.vec_bodyid[:]].flatten()
        dt = float(self.model.opt.timestep)
        self.cosserat_math.computeWrenches(
            self.x_flat,
            self.quat_flat,
            dt,
            self.torq_node_flat,
        )
        self.torq_node = self.torq_node_flat.reshape((self.nv + 2, 3))
        self._limit_torques()

    def _update_xvecs(self):
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
