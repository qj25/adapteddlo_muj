import numpy as np
import mujoco

import adapteddlo_muj.utils.transform_utils as T
import adapteddlo_muj.utils.mjc2_utils as mjc2
import adapteddlo_muj.controllers.geds_cpp.RodGeds as RodGeds
from adapteddlo_muj.utils.dlo_utils import force2torq


class DLORopeGeds:
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
        samples_per_span=4,
        k_p_endpoint=1.0e4,
        k_d_endpoint=1.0e2,
    ):
        self.model = model
        self.data = data
        self.bothweld = bothweld
        self.f_limit = f_limit
        self.segment_length = segment_length
        self.radius = radius
        self.samples_per_span = samples_per_span
        self.k_p_endpoint = k_p_endpoint
        self.k_d_endpoint = k_d_endpoint

        self.nv = n_link - 1
        self.vec_siteid = np.zeros(self.nv + 2, dtype=int)
        self.vec_bodyid = np.zeros(self.nv + 2, dtype=int)
        self._init_sitebody()

        self.force_node = np.zeros((self.nv + 2, 3))
        self.torq_node = np.zeros((self.nv + 2, 3))
        self.tau_roll = np.zeros((self.nv + 2, 3))
        self.force_node_flat = self.force_node.flatten()
        self.torq_node_flat = self.torq_node.flatten()
        self.tau_roll_flat = self.tau_roll.flatten()

        self.x_flat = np.zeros((self.nv + 2) * 3)
        self.quat_flat = np.zeros((self.nv + 2) * 4)

        self.reset_rot = overall_rot
        self.overall_rot = overall_rot
        self.p_thetan = 0.0

        self.alpha_bar = alpha_bar
        self.beta_bar = beta_bar
        self._material_from_alpha_beta()

        self._endpoint_start_pos = None
        self._endpoint_end_pos = None

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
        self._init_geds_cpp()

    def _material_from_alpha_beta(self):
        r = self.radius
        j1 = np.pi * r**4 / 2.0
        ix = np.pi * r**4 / 4.0
        dt = float(self.model.opt.timestep)
        stiff_scale = dt * dt * 1.0e2
        self.youngs_modulus = (self.alpha_bar / ix) * stiff_scale
        self.torsion_modulus = (self.beta_bar / j1) * stiff_scale

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
        self.x_flat = self.data.site_xpos[self.vec_siteid[:]].flatten().copy()
        self.quat_flat = self.data.xquat[self.vec_bodyid[:]].flatten().copy()

    def _init_geds_cpp(self):
        self._capture_rest_pose()
        self.geds_math = RodGeds.RodGeds(
            self.nv + 2,
            self.segment_length,
            2.0 * self.radius,
            self.youngs_modulus,
            self.torsion_modulus,
        )
        self.geds_math.setNumSamples(self.samples_per_span)
        self.geds_math.reinitRest(self.x_flat, self.quat_flat)

    def reset_neutral(self):
        self._capture_rest_pose()
        self.geds_math.reinitRest(self.x_flat, self.quat_flat)

    def set_endpoint_targets(self, start_pos=None, end_pos=None):
        """Persistent endpoint targets on first/last DLO chain bodies (world frame). None = free."""
        self._endpoint_start_pos = None if start_pos is None else np.asarray(start_pos, dtype=float).copy()
        self._endpoint_end_pos = None if end_pos is None else np.asarray(end_pos, dtype=float).copy()

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
        self.geds_math.setMaterial(self.youngs_modulus, self.torsion_modulus)

    def _update_xvecs(self):
        return None

    def _reset_vel(self):
        self.data.qvel[self.dlo_joint_qveladdr_full] = np.zeros(self.rxqva_len * 3)

    def reset_qvel_rotx(self):
        self.data.qvel[self.rotx_qveladdr] = np.zeros(self.rxqva_len)

    def _limit_forces(self):
        force_mag = np.linalg.norm(self.force_node)
        if force_mag > self.f_limit:
            self.force_node *= self.f_limit / force_mag

    def _calc_elastic_wrenches(self, skip_start=False, skip_end=False):
        self.x_flat = self.data.site_xpos[self.vec_siteid[:]].flatten()
        self.quat_flat = self.data.xquat[self.vec_bodyid[:]].flatten()

        self.geds_math.computeElasticWrenches(
            self.x_flat,
            self.quat_flat,
            self.force_node_flat,
            self.tau_roll_flat,
        )
        self.force_node = self.force_node_flat.reshape((self.nv + 2, 3))
        self.tau_roll = self.tau_roll_flat.reshape((self.nv + 2, 3))

        if skip_start:
            self.force_node[0] = 0.0
            self.tau_roll[0] = 0.0
        if skip_end:
            self.force_node[-1] = 0.0
            self.tau_roll[-1] = 0.0

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
            mujoco.mju_rotVecQuat(self.tau_roll[i], self.tau_roll[i], body_invquat[i])
            self.torq_node[i] += self.tau_roll[i]

        self.torq_node = np.flip(self.torq_node, 0)

    def _apply_endpoint_pd(self, start_pos, end_pos):
        self.data.xfrc_applied[self.vec_bodyid[:], :3] = 0.0

        if start_pos is not None:
            bid = self.vec_bodyid[0]
            x_cur = self.data.xpos[bid].copy()
            v_cur = self.data.cvel[bid, 3:6].copy()
            f_hold = self.k_p_endpoint * (start_pos - x_cur) - self.k_d_endpoint * v_cur
            self.data.xfrc_applied[bid, :3] = f_hold

        if end_pos is not None:
            bid = self.vec_bodyid[-1]
            x_cur = self.data.xpos[bid].copy()
            v_cur = self.data.cvel[bid, 3:6].copy()
            f_hold = self.k_p_endpoint * (end_pos - x_cur) - self.k_d_endpoint * v_cur
            self.data.xfrc_applied[bid, :3] = f_hold

    def update_torque(self, start_pos=None, end_pos=None):
        if start_pos is None:
            start_pos = self._endpoint_start_pos
        if end_pos is None:
            end_pos = self._endpoint_end_pos

        skip_start = start_pos is not None
        skip_end = end_pos is not None

        self._calc_elastic_wrenches(skip_start=skip_start, skip_end=skip_end)
        self._forces_to_torques()
        self._apply_endpoint_pd(start_pos, end_pos)

        if self.bothweld:
            self.data.qfrc_passive[self.qvel0_addr - 3 : self.qvellast_addr + 1] += self.torq_node[
                :-1
            ].flatten()
        else:
            self.data.qfrc_passive[self.qvel0_addr : self.qvellast_addr + 1] += self.torq_node[
                1:-1
            ].flatten()
