import numpy as np
import mujoco

import adapteddlo_muj.utils.mjc2_utils as mjc2
import adapteddlo_muj.controllers.cosserat_cpp.RodCosserat as RodCosserat

# Bisect instability: flip one key to False per test round (order matters).
# 1. angular_damping      — body-frame -k_d*omega on child torques
# 2. stretch_forces       — distance springs (world-frame mj_applyFT -> qfrc_passive)
# 3. material_scale       — scales k_bend/k_twist (1.0 = full alpha_bar/beta_bar)
# 4. shadow_num_iters_4   — shadow solve iterations (False -> 2 iters)
# 5. position_spring_torque — False re-enables inv_dt velocity drive in C++
# 6. joint_child_torques  — False uses per-node torques (old output path)
# 7. qfrc_passive_torques — False applies bend torques via mj_applyFT
# 8. body_xpos            — False uses site_xpos for tangents / rest capture
COSSERAT_FIXES = {
    "angular_damping": False,
    "stretch_forces": True,
    "material_scale": True,
    "shadow_num_iters_4": True,
    "position_spring_torque": True,
    "joint_child_torques": True,
    "qfrc_passive_torques": True,
    "body_xpos": True,
}


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
        num_iters=4 if COSSERAT_FIXES["shadow_num_iters_4"] else 2,
        k_torque=0.01,
        k_stretch=None,
        k_angular_damp=0.2,
        joint_damping=0.01,
        material_scale=0.2 if COSSERAT_FIXES["material_scale"] else 1.0,
    ):
        self.model = model
        self.data = data
        self.bothweld = bothweld
        self.f_limit = f_limit
        self.segment_length = segment_length
        self.radius = radius
        self.num_iters = num_iters
        self.k_torque = k_torque
        self.k_angular_damp = k_angular_damp
        self.joint_damping = joint_damping
        self.material_scale = material_scale

        self.nv = n_link - 1
        self.vec_siteid = np.zeros(self.nv + 2, dtype=int)
        self.vec_bodyid = np.zeros(self.nv + 2, dtype=int)
        self._init_sitebody()

        self.force_node = np.zeros((self.nv + 2, 3))
        self.torq_node = np.zeros((self.nv + 2, 3))
        self.force_node_flat = self.force_node.flatten()
        self.torq_node_flat = self.torq_node.flatten()
        self.x_flat = np.zeros((self.nv + 2) * 3)
        self.quat_flat = np.zeros((self.nv + 2) * 4)

        self.reset_rot = overall_rot
        self.overall_rot = overall_rot
        self.p_thetan = 0.0

        self.alpha_bar = alpha_bar * 500
        self.beta_bar = beta_bar
        self._material_from_alpha_beta(k_stretch)

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
        self._init_cosserat_cpp()

    def _apply_joint_damping(self):
        """Override MuJoCo ball-joint damping for rope links (XML default is 0.01)."""
        for dof in self.dlo_joint_qveladdr_full:
            self.model.dof_damping[dof] = self.joint_damping

    def _material_from_alpha_beta(self, k_stretch=None):
        s = self.material_scale
        dt = float(self.model.opt.timestep)
        stiff_scale = dt * dt * 1.0e2
        self.k_bend = s * self.alpha_bar
        self.k_twist = s * self.beta_bar
        if k_stretch is None:
            self.k_stretch = s * 50.0 * self.alpha_bar * stiff_scale
        else:
            self.k_stretch = k_stretch

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

    def _node_positions(self):
        if COSSERAT_FIXES["body_xpos"]:
            return self.data.xpos[self.vec_bodyid[:]]
        return self.data.site_xpos[self.vec_siteid[:]]

    def _capture_neutral_state(self):
        self.x_flat = self._node_positions().flatten().copy()
        self.quat_flat = self.data.xquat[self.vec_bodyid[:]].flatten().copy()

    def _init_cosserat_cpp(self):
        self._capture_neutral_state()
        self.cosserat_math = RodCosserat.RodCosserat(
            self.nv + 2,
            self.segment_length,
            self.k_bend,
            self.k_twist,
        )
        stretch_k = self.k_stretch if COSSERAT_FIXES["stretch_forces"] else 0.0
        self.cosserat_math.setStretchStiffness(stretch_k)
        self.cosserat_math.setNumIterations(self.num_iters)
        self.cosserat_math.setTorqueGain(self.k_torque)
        self.cosserat_math.setVelocityDrive(not COSSERAT_FIXES["position_spring_torque"])
        self.cosserat_math.setJointChildTorques(COSSERAT_FIXES["joint_child_torques"])
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
        stretch_k = self.k_stretch if COSSERAT_FIXES["stretch_forces"] else 0.0
        self.cosserat_math.setStretchStiffness(stretch_k)

    def _pinned_node_ids(self):
        n = self.nv + 2
        if self.bothweld:
            return {0, n - 1}
        return set()

    def _limit_wrenches(self):
        force_mag = np.linalg.norm(self.force_node)
        if force_mag > self.f_limit:
            self.force_node *= self.f_limit / force_mag

    def _limit_torques(self):
        torq_mag = np.linalg.norm(self.torq_node)
        if torq_mag > self.f_limit:
            self.torq_node *= self.f_limit / torq_mag

    def _apply_angular_damping(self):
        if not COSSERAT_FIXES["angular_damping"] or self.k_angular_damp <= 0.0:
            return
        pinned = self._pinned_node_ids()
        inv_quat = np.zeros(4, dtype=np.float64)
        omega_b = np.zeros(3, dtype=np.float64)
        for i in range(1, self.nv + 2):
            if i in pinned:
                continue
            bid = int(self.vec_bodyid[i])
            omega_w = self.data.cvel[bid, :3]
            inv_quat[:] = self.data.xquat[bid]
            inv_quat[1:] *= -1.0
            mujoco.mju_rotVecQuat(omega_b, omega_w, inv_quat)
            self.torq_node[i] -= self.k_angular_damp * omega_b

    def _calc_centerline_wrenches(self):
        self.x_flat = self._node_positions().flatten()
        self.quat_flat = self.data.xquat[self.vec_bodyid[:]].flatten()
        dt = float(self.model.opt.timestep)
        self.cosserat_math.computeWrenches(
            self.x_flat,
            self.quat_flat,
            dt,
            self.force_node_flat,
            self.torq_node_flat,
        )
        self.force_node = self.force_node_flat.reshape((self.nv + 2, 3))
        self.torq_node = self.torq_node_flat.reshape((self.nv + 2, 3))
        self._apply_angular_damping()

    def _apply_stretch_forces(self):
        if not COSSERAT_FIXES["stretch_forces"]:
            return
        pinned = self._pinned_node_ids()
        zero_torque = np.zeros(3, dtype=np.float64)
        for i in range(self.nv + 2):
            if i in pinned:
                continue
            force = self.force_node[i]
            if np.linalg.norm(force) < 1e-12:
                continue
            bid = int(self.vec_bodyid[i])
            mujoco.mj_applyFT(
                self.model,
                self.data,
                force,
                zero_torque,
                self.data.xpos[bid],
                bid,
                self.data.qfrc_passive,
            )

    def _update_xvecs(self):
        return None

    def _reset_vel(self):
        self.data.qvel[self.dlo_joint_qveladdr_full] = np.zeros(self.rxqva_len * 3)

    def reset_qvel_rotx(self):
        self.data.qvel[self.rotx_qveladdr] = np.zeros(self.rxqva_len)

    def _apply_bend_torques(self):
        pinned = self._pinned_node_ids()
        zero_force = np.zeros(3, dtype=np.float64)
        if COSSERAT_FIXES["qfrc_passive_torques"]:
            if self.bothweld:
                self.data.qfrc_passive[self.qvel0_addr - 3 : self.qvellast_addr + 1] += (
                    self.torq_node[:-1].flatten()
                )
            else:
                self.data.qfrc_passive[self.qvel0_addr : self.qvellast_addr + 1] += (
                    self.torq_node[1:-1].flatten()
                )
            return
        xfrc = np.zeros(3, dtype=np.float64)
        for i in range(self.nv + 2):
            if i in pinned:
                continue
            lfrc = self.torq_node[i]
            if np.linalg.norm(lfrc) < 1e-12:
                continue
            bid = int(self.vec_bodyid[i])
            mujoco.mju_rotVecQuat(xfrc, lfrc, self.data.xquat[bid])
            mujoco.mj_applyFT(
                self.model,
                self.data,
                zero_force,
                xfrc,
                self.data.xpos[bid],
                bid,
                self.data.qfrc_passive,
            )

    def update_torque(self):
        self._calc_centerline_wrenches()
        self._limit_wrenches()
        self._apply_stretch_forces()
        self._limit_torques()
        self._apply_bend_torques()
