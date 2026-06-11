import numpy as np
import mujoco

import adapteddlo_muj.utils.transform_utils as T
import adapteddlo_muj.utils.mjc2_utils as mjc2
import adapteddlo_muj.controllers.cosserat2_cpp.RodCosserat2 as RodCosserat2
from adapteddlo_muj.utils.dlo_utils import force2torq
from adapteddlo_muj.utils.rope_stiffness import (
    cosserat2_moduli,
    material_scale_for,
    scale_joint_damping,
)


class DLORopeCosserat2:
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
        num_iterations=4,
        k_force=0.05,
        k_torque=0.05,
        joint_damping=scale_joint_damping(0.01, "cosserat2"),
    ):
        self.model = model
        self.data = data
        self.bothweld = bothweld
        self.f_limit = f_limit
        self.segment_length = segment_length
        self.radius = radius
        self.num_iterations = num_iterations
        self.k_force = k_force
        self.k_torque = k_torque
        self.joint_damping = joint_damping

        self.nv = n_link - 1
        self.vec_siteid = np.zeros(self.nv + 2, dtype=int)
        self.vec_bodyid = np.zeros(self.nv + 2, dtype=int)
        self._init_sitebody()

        self.force_node = np.zeros((self.nv + 2, 3))
        self.torq_node = np.zeros((self.nv + 2, 3))
        self.tau_direct = np.zeros((self.nv + 2, 3))
        self.force_node_flat = self.force_node.flatten()
        self.torq_node_flat = self.torq_node.flatten()
        self.tau_direct_flat = self.tau_direct.flatten()

        self.x_flat = np.zeros((self.nv + 2) * 3)
        self.quat_flat = np.zeros((self.nv + 2) * 4)
        self.inv_mass = np.zeros(self.nv + 2)
        self.inv_inertia_w_flat = np.zeros((self.nv + 2) * 9)

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
        self._apply_joint_damping()
        self._init_cosserat2_cpp()

    def _apply_joint_damping(self):
        """Override MuJoCo ball-joint damping for rope links (XML default is 0.01)."""
        for dof in self.dlo_joint_qveladdr_full:
            self.model.dof_damping[dof] = self.joint_damping

    def _material_from_alpha_beta(self):
        self.k_stretch, self.k_bend, self.k_twist = cosserat2_moduli(
            self.alpha_bar,
            self.beta_bar,
            self.radius,
            self.model.opt.timestep,
            material_scale_for("cosserat2"),
        )

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

    def _capture_rest_pose(self):
        self.x_flat = self.data.xpos[self.vec_bodyid[:]].flatten().copy()
        self.quat_flat = self.data.xquat[self.vec_bodyid[:]].flatten().copy()

    def _init_cosserat2_cpp(self):
        self._capture_rest_pose()
        self.cosserat2_math = RodCosserat2.RodCosserat2(
            self.nv + 2,
            self.x_flat,
            self.quat_flat,
            self.segment_length,
            self.radius,
            self.k_stretch,
            self.k_bend,
            self.k_twist,
            self.bothweld,
        )
        self.cosserat2_math.setForceGain(self.k_force)
        self.cosserat2_math.setTorqueGain(self.k_torque)
        self.cosserat2_math.setNumIterations(self.num_iterations)

    def reset_neutral(self):
        self._capture_rest_pose()
        self.cosserat2_math.reinitRest(self.x_flat, self.quat_flat)

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
        self.cosserat2_math.setMaterial(self.k_stretch, self.k_bend, self.k_twist)

    def _update_xvecs(self):
        return None

    def _reset_vel(self):
        self.data.qvel[self.dlo_joint_qveladdr_full] = np.zeros(self.rxqva_len * 3)

    def reset_qvel_rotx(self):
        self.data.qvel[self.rotx_qveladdr] = np.zeros(self.rxqva_len)

    def _fill_inv_mass_inertia(self):
        n = self.nv + 2
        for i in range(n):
            bid = self.vec_bodyid[i]
            if self.bothweld and (i == 0 or i == n - 1):
                self.inv_mass[i] = 0.0
            else:
                mass = self.model.body_mass[bid]
                self.inv_mass[i] = 0.0 if mass <= 0.0 else 1.0 / mass

            quat = self.data.xquat[bid]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, quat)
            rot = rot.reshape(3, 3)
            inv_i_local = np.diag(1.0 / np.maximum(self.model.body_inertia[bid], 1e-12))
            inv_i_w = rot @ inv_i_local @ rot.T
            self.inv_inertia_w_flat[i * 9 : (i + 1) * 9] = inv_i_w.flatten()

    def _limit_forces(self):
        force_mag = np.linalg.norm(self.force_node)
        if force_mag > self.f_limit:
            self.force_node *= self.f_limit / force_mag

    def _calc_wrenches(self):
        self.x_flat = self.data.xpos[self.vec_bodyid[:]].flatten()
        self.quat_flat = self.data.xquat[self.vec_bodyid[:]].flatten()
        self._fill_inv_mass_inertia()

        self.cosserat2_math.computeWrenches(
            self.x_flat,
            self.quat_flat,
            self.inv_mass,
            self.inv_inertia_w_flat,
            self.model.opt.timestep,
            self.force_node_flat,
            self.tau_direct_flat,
        )
        self.force_node = self.force_node_flat.reshape((self.nv + 2, 3))
        self.tau_direct = self.tau_direct_flat.reshape((self.nv + 2, 3))
        self._limit_forces()

    def _forces_to_torques(self):
        x_sites = self.data.site_xpos[self.vec_siteid[:]]
        adj_dist = x_sites[1:] - x_sites[:-1]
        excl_joints = 0
        ids_i = range(excl_joints, self.nv + 2 - excl_joints)
        self.torq_node = force2torq(self.force_node, adj_dist, ids_i)

        body_invquat = T.quat_inverse_many(self.data.xquat[self.vec_bodyid[:]])
        for i in range(1, len(self.torq_node)):
            mujoco.mju_rotVecQuat(self.torq_node[i], self.torq_node[i], body_invquat[i])
            mujoco.mju_rotVecQuat(self.tau_direct[i], self.tau_direct[i], body_invquat[i])
            self.torq_node[i] += self.tau_direct[i]

        self.torq_node = np.flip(self.torq_node, 0)

    def update_torque(self):
        self._calc_wrenches()
        self._forces_to_torques()
        if self.bothweld:
            self.data.qfrc_passive[self.qvel0_addr - 3 : self.qvellast_addr + 1] += self.torq_node[
                :-1
            ].flatten()
        else:
            self.data.qfrc_passive[self.qvel0_addr : self.qvellast_addr + 1] += self.torq_node[
                1:-1
            ].flatten()
