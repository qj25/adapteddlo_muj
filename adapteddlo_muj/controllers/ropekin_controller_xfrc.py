"""
Discrete Elastic Rope controller on a no-damping, no-spring
finite element rod.

Assumptions:
- one end is fixed (xml id 1 if one end, xml id -1 if both ends)
- equality constraint of endpoint stable

Drawbacks:
- tuning of force limits (f_lim) required

Possible solutions:
- raise limit (remove is best)
- adjust mass
- change vel_reset freq (5-10 --> requires more testing)
- adjust connect constraint stiffness

To-do:

stopped:
- test slack force (use grav (temp) to slack it)
- apply force
"""

from time import time
# import math
import numpy as np
import mujoco

import adapteddlo_muj.utils.transform_utils as T
import adapteddlo_muj.controllers.dlo_cpp.Dlo_iso as Dlo_iso
import adapteddlo_muj.utils.mjc2_utils as mjc2
# from adapteddlo_muj.utils.filters import ButterLowPass

class DLORopeXfrc:
    # Base rope controller is for both ends fixed
    def __init__(
        self,
        model,
        data,
        n_link,
        overall_rot=0.,
        alpha_bar=1.345/10,
        beta_bar=0.789/10,
        f_limit=1000.,  # used only for LHB test
        incl_prevf=False
    ):
        self.model = model
        self.data = data

        # init variable
        self.d_vec = 0
        self.nv = n_link - 1 - self.d_vec * 2
        self.vec_siteid = np.zeros(self.nv+2, dtype=int)
        self.vec_bodyid = np.zeros(self.nv+2, dtype=int)
        self.link_bodyid = np.zeros(self.nv+1, dtype=int)
        
        self._init_sitebody()

        self.x = np.zeros((self.nv+2,3))
        self.x_vel = np.zeros((self.nv+2,3))
        self.e = np.zeros((self.nv+1,3))

        self.force_node = np.zeros((self.nv+2,3))
        self.incl_prevf = incl_prevf
        if self.incl_prevf:
            self.force_node_prev = np.zeros((self.nv+2,3))
            self.prevf_ratio = 0.1
        self.force_node_flat = self.force_node.flatten()

        self.reset_rot = overall_rot 
        self.overall_rot = self.reset_rot
        self.p_thetan = self.reset_rot % (2. * np.pi)
        if self.p_thetan > np.pi:
            self.p_thetan -= 2 * np.pi
        self.theta_displace = self.p_thetan

        # Calc bars
        self.e_bar = np.zeros(self.nv+1)

        # define variable constants
        self.alpha_bar = alpha_bar
        self.beta_bar = beta_bar
        self.f_limit = f_limit
        
        # define base bishop frame
        self.bf0_bar = np.zeros((3,3))
        self.bf_end = np.zeros((3,3))
        self.bf_end_flat = self.bf_end.flatten()

        self.dlo_joint_ids = []
        self.dlo_joint_qveladdr = []
        for i in range(1, n_link):    # from freejoint2 to freejoint(n_links)
            fj_str = f'J_{i}'
            if i == (n_link-1):
                fj_str = 'J_last'
            self.dlo_joint_ids.append(mjc2.obj_name2id(
                self.model,"joint",fj_str
            ))
            # if self.model.jnt_type(
                # self.dlo_joint_ids[-1]
            # ) == self.model.mjtJoint.mjJNT_FREE:
                # print('it is free joint')
                # print(fj_str)
                # print(self.model.jnt_dofadr[self.dlo_joint_ids[-1]])
            self.dlo_joint_qveladdr.append(
                self.model.jnt_dofadr[self.dlo_joint_ids[-1]]
            )
        self.dlo_joint_qveladdr = np.array(self.dlo_joint_qveladdr)
        self.dlo_joint_qveladdr_full = [
            n for n in range(
                self.dlo_joint_qveladdr[0],
                self.dlo_joint_qveladdr[-1] + 3
            )
        ]
        self.rotx_qveladdr = self.dlo_joint_qveladdr[:] + 2
        self.rxqva_len = len(self.rotx_qveladdr)

        self._init_dlo_cpp()
        # self.dlo_math.changestepgain(0.)
        # init filter
        # fs = 1.0 / self.model.opt.timestep
        # cutoff = 30
        # self.lowpass_filter = ButterLowPass(cutoff, fs, order=5)

    def _init_sitebody(self):
        for i in range(self.d_vec, self.nv+2 + self.d_vec):
            ii = (self.nv+1 + 2*self.d_vec) - i
            ii_s = ii
            ii_b = ii
            if ii == (self.nv+1):
                ii_s = 'last'
                ii_b = 'last2'
            if ii == (self.nv):
                ii_b = 'last'
            if ii == 0:
                ii_b = 'first'
            # id starts from 'last' section
            self.vec_siteid[i - self.d_vec] = mjc2.obj_name2id(
                self.model,"site",'S_{}'.format(ii_s)
            )

            self.vec_bodyid[i - self.d_vec] = mjc2.obj_name2id(
                self.model,"body",'B_{}'.format(ii_b)
            )
        self.ropeend_bodyid = mjc2.obj_name2id(
            self.model,"body",'stiffrope'
        )
        self.startsec_site = mjc2.obj_name2id(
            self.model,"site",'S_last'.format(self.nv+1 + self.d_vec)
        )
        self.endsec_site = mjc2.obj_name2id(
            self.model,"site",'S_first'.format(1 + self.d_vec)
        )

    def _update_xvecs(self):
        # start_t = time()

        self.x = self.data.site_xpos[
            self.vec_siteid[:]
        ].copy()
        # end_t1 = time()

        # for i in range(self.nv+2):
        #     # self.x_prev[i] = self.x[i]
        #     self.x[i] = self.data.site_xpos[self.vec_siteid[i]].copy()
        #     # self.x_vel[i] = self.x[i] - self.x_prev[i]
        #     # input(self.x[i])

        # end_t2 = time()
        # print(f"t_np = {end_t1 - start_t}")
        # print(f"t_loop = {end_t2 - end_t1}")

    def _init_resetbody_vars(self):
        self.xpos_reset = self.model.body_pos[
            self.vec_bodyid[:]
        ].copy()
        self.xquat_reset = self.model.body_quat[
            self.vec_bodyid[:]
        ].copy()
        # print(self.vec_bodyid[:])

    def set_resetbody_vars(self):
        self._init_resetbody_vars()

    def _init_dlo_cpp(self):
        self._update_xvecs()
        self._init_resetbody_vars()
        self._x2e()
        self._init_bf()
        # self._update_bishf()
        self.dlo_math = Dlo_iso.DLO_iso(
            self.x.flatten(),
            self.bf0_bar.flatten(),
            self.p_thetan,
            self.overall_rot,
            self.alpha_bar,
            self.beta_bar
        )
        self._init_o2m()

    def _update_dlo_cpp(self):
        self._update_xvecs()
        self._update_bishf_S()
        bf_align = self.dlo_math.updateVars(
            self.x.flatten(),
            self.bf0_bar.flatten(),
            self.bf_end_flat
        )
        self.bf_end = self.bf_end_flat.reshape((3,3))
        self.p_thetan = self._get_thetan()
        self.overall_rot = self.dlo_math.updateTheta(self.p_thetan)
        return bf_align

    def get_dlosim(self):
        ropeend_pos = self.model.body_pos[
            self.ropeend_bodyid,:
        ].copy()
        ropeend_quat = self.model.body_quat[
            self.ropeend_bodyid,:
        ].copy()
        return (
            ropeend_pos,
            ropeend_quat,
            self.overall_rot,
            self.p_thetan
        )

    def set_dlosim(
        self,
        ropeend_pos,
        ropeend_quat,
        overall_rot,
        p_thetan
    ):
        self.model.body_pos[
            self.ropeend_bodyid,:
        ] = ropeend_pos
        self.model.body_quat[
            self.ropeend_bodyid,:
        ] = ropeend_quat
        self.overall_rot = overall_rot
        # self.overall_rot = 27.*(2.*np.pi)
        self.p_thetan = p_thetan
        self.dlo_math.resetTheta(self.p_thetan, self.overall_rot)
        # self.dlo_math.updateTheta(self.p_thetan)

    def reset_body(self):
        self.model.body_pos[
            self.vec_bodyid[:],:
        ] = self.xpos_reset.copy()
        self.model.body_quat[
            self.vec_bodyid[:],:
        ] = self.xquat_reset.copy()
        self._reset_vel()
        mujoco.mj_forward(self.model, self.data)

    def reset_sim(self):
        self.overall_rot = self.reset_rot
        self.p_thetan = self.reset_rot % (2. * np.pi)
        if self.p_thetan > np.pi:
            self.p_thetan -= 2 * np.pi
        self.dlo_math.resetTheta(self.p_thetan, self.overall_rot)

    def change_ropestiffness(self, alpha_bar, beta_bar):
        self.alpha_bar = alpha_bar  # bending modulus
        self.beta_bar = beta_bar    # twisting modulus
        self.dlo_math.changeAlphaBeta(self.alpha_bar, self.beta_bar)

    # ~~~~~~~~~~~~~~~~~~~~~~|formula functions|~~~~~~~~~~~~~~~~~~~~~~


    def _x2e(self):
        # includes calculations for e_bar
        self.e[0] = self.x[1] - self.x[0]
        self.e_bar[0] = np.linalg.norm(self.e[0])
        for i in range(1,self.nv+1):
            self.e[i] = self.x[i+1] - self.x[i]
            self.e_bar[i] = np.linalg.norm(self.e[i])

    def _init_bf(self):
        parll_tol = 1e-6
        self.bf0_bar[0,:] = self.e[0] / self.e_bar[0]
        self.bf0_bar[1,:] = np.cross(self.bf0_bar[0,:], np.array([0, 0, 1.]))
        if np.linalg.norm(self.bf0_bar[1,:]) < parll_tol:
            self.bf0_bar[1,:] = np.cross(self.bf0_bar[0,:], np.array([0, 1., 0]))
        self.bf0_bar[1,:] /= np.linalg.norm(self.bf0_bar[1,:])
        self.bf0_bar[2,:] = np.cross(self.bf0_bar[0,:], self.bf0_bar[1,:])

    def _update_bishf_S(self):
        # updates start of bishop frame
        # t1 = time()
        mat_res = np.zeros(9)
        self.dlo_math.calculateOf2Mf(
            self.data.site_xmat[self.startsec_site],
            mat_res
        )
        mat_res = mat_res.reshape((3,3))
        self.bf0_bar = np.transpose(mat_res)
        # self.bf0_bar = np.transpose(self._of2mf(
        #     self.data.site_xmat[self.startsec_site].reshape((3, 3))
        # ))
        # t3 = time()
        # print(f"*total = {(t3 - t1) * 51.}")
    def _of2mf(self, mat_o):
        # converts object frame to material frame
        return T.quat2mat(
            T.quat_multiply(
                T.mat2quat(mat_o),
                self.qe_o2m_loc
            )
        )

    def _mf2of(self, mat_m):
        # converts material frame to object frame
        return T.quat2mat(
            T.quat_multiply(
                T.mat2quat(mat_m),
                self.qe_m2o_loc
            )
        )

    def _init_loc_rotframe(self, q1, q2):
        qe = T.axisangle2quat(T.quat_error(q1, q2))
        q1_inv = T.quat_inverse(q1)
        qe_loc = T.quat_multiply(T.quat_multiply(q1_inv,qe),q1)
        return qe_loc
    
    def _init_o2m(self):
        # init for mf_adapt and p_thetan
        q_o0 = T.mat2quat(
            self.data.site_xmat[self.startsec_site].reshape((3, 3))
        )
        q_b0 = T.mat2quat(
            np.transpose(self.bf0_bar)
        )
        self.qe_o2m_loc = self._init_loc_rotframe(q_o0, q_b0)
        self.qe_m2o_loc = self._init_loc_rotframe(q_b0, q_o0)

        self.dlo_math.initQe_o2m_loc(self.qe_o2m_loc)

    def _get_thetan(self):
        mat_on = self.data.site_xmat[self.endsec_site]
        mat_bn = np.transpose(self.bf_end)
        mat_mn = np.zeros(9)
        self.dlo_math.calculateOf2Mf(
            mat_on,
            mat_mn
        )
        mat_mn = mat_mn.reshape((3,3))

        theta_n = (
            self.dlo_math.angBtwn3(mat_bn[:,1], mat_mn[:,1], mat_bn[:,0])
            + self.theta_displace
        )
        return theta_n

    def _calc_centerlineF(self):
        # self.force_node_flat = np.zeros((self.nv+2)*3)
        self.dlo_math.calculateCenterlineF2(self.force_node_flat)
        # print(self.force_node_flat)
        self.force_node = self.force_node_flat.reshape((self.nv+2,3))
        if self.incl_prevf:
            self._incl_prevforce()
        # print('force = ')
        # print(self.force_node)

    def _limit_force(self, f1):
        # limit the force on each node to self.f_limit
        for i in range(len(f1)):
            f1_mag = np.linalg.norm(f1[i])
            if f1_mag > self.f_limit:
                f1[i] *= self.f_limit / f1_mag

    def _limit_totalforce(self):
        # limit the total force magnitude of sum of all nodes to f_total_limit
        f_total_limit = 100.
        force_mag = np.linalg.norm(self.force_node)
        if force_mag > f_total_limit:
            self.force_node *= f_total_limit / force_mag
            
    def _incl_prevforce(self):
        self.force_node = (
            self.force_node * (1-self.prevf_ratio)
            + self.force_node_prev * self.prevf_ratio
        )
        self.force_node_prev = self.force_node.copy()

    def _reset_vel(self):
        self.data.qvel[self.dlo_joint_qveladdr_full] = np.zeros(self.rxqva_len*3)

    def reset_qvel_rotx(self):
        self.data.qvel[self.rotx_qveladdr] = np.zeros(self.rxqva_len)

    def get_ropevel(self):
        return mjc2.obj_getvel(
            self.model,
            self.data,
            "body",
            self.vec_bodyid[:]
        )

    def update_force(self, f_scale=1.0, limit_f=False, limit_f_indiv=False, damp_f=False):
        # t1 = time()
        bf_align = self._update_dlo_cpp()
        # excl_joints = 2 - 2*self.d_vec # 2 joints excluded from each side (no force)
        # t2 = time()
        self._calc_centerlineF()
        self.avg_force = np.linalg.norm(self.force_node)/len(self.force_node)
        # t3 = time()

        # if limit_f:
        #     self._limit_totalforce()
        # if limit_f_indiv:
        #     self._limit_force(self.force_node)

        excl_joints = 0
        ids_i = range(excl_joints,self.nv+2-excl_joints)

        self.data.xfrc_applied[self.vec_bodyid[ids_i],:3] = (
            f_scale * self.force_node[ids_i]
        )

        # for i in ids_i:
            # mujoco.mj_applyFT(
                # self.model,self.data,
                # self.force_node[i],
                # np.zeros(3),
                # self.x[i],
                # self.vec_bodyid[i],
                # self.data.qfrc_passive
            # )
        # t4 = time()
        # print("hi")
        # print(t2-t1)
        # print(t3-t2)
        # print(t4-t3)

        return self.force_node.copy(), self.x.copy(), bf_align
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Other things ||~~~~~~~~~~~~~~~~~~~~~~~~~~
    def ropeend_rot(self, rot_a=np.pi/180):
        rot_quat = T.axisangle2quat(np.array([rot_a, 0., 0.]))
        new_quat = T.quat_multiply(rot_quat, self.model.body_quat[self.ropeend_bodyid])
        self.model.body_quat[self.ropeend_bodyid] = new_quat

    def ropeend_pos(self, pos_move=np.array([0., -1e-4, 0.])):
        self.model.body_pos[self.ropeend_bodyid][:] += pos_move.copy()


# ~~~~~~~~~~~~~~~~~~~~~~~~|End of Class|~~~~~~~~~~~~~~~~~~~~~~~~