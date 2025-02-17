import sys
import os
import numpy as np
from math import floor

import mujoco
import mujoco_viewer

import gymnasium as gym
from gymnasium import utils
import pickle
from time import time

import adapteddlo_muj.utils.transform_utils as T
from adapteddlo_muj.utils.mjc_utils import MjSimWrapper
from adapteddlo_muj.utils.xml_utils import XMLWrapper
import adapteddlo_muj.utils.mjc2_utils as mjc2

from adapteddlo_muj.assets.genrope.gdv_O import GenKin_O
from adapteddlo_muj.assets.genrope.gdv_O_xfrc import GenKin_O_xfrc
from adapteddlo_muj.assets.genrope.gdv_N import GenKin_N
from adapteddlo_muj.controllers.ropekin_controller_xfrc import DLORopeXfrc
from adapteddlo_muj.controllers.ropekin_controller_adapt import DLORopeAdapt
from adapteddlo_muj.utils.data_utils import compute_PCA, centralize_devdata


class TestRopeEnv(gym.Env, utils.EzPickle):
    def __init__(
        self,
        do_render=False,
        alpha_bar=1.345/50,   # 1.345/50,
        beta_bar=0.789/50,    # 0.789/50,
        r_pieces=30,    # max. 33
        r_mass=None,
        r_len = 2*np.pi,
        r_thickness=0.03,
        test_type=None,
        overall_rot=None,
        manual_rot=False,
        new_start=False,
        limit_f=False,
        stifftorqtype='adapt',
        grav_on=True,
        rgba_vals=None
    ):
        utils.EzPickle.__init__(self)

        self.do_render = do_render
        self.test_type = test_type
        self.limit_f = limit_f
        self.storqtype = stifftorqtype
        self.rgba_vals = rgba_vals
        # if self.storqtype == 'lop':
        #     self.picklefolder = 'lop'
        # elif self.storqtype == 'adapt':
        #     self.picklefolder = 'adapt'
        # elif self.storqtype == 'hyb':
        #     self.picklefolder = 'hyb'
        if grav_on:
            self.picklefolder = 'real/grav/' + self.storqtype
        else:
            self.picklefolder = 'real/nograv/' + self.storqtype

        # rope init
        self.r_len = r_len
        self.r_pieces = r_pieces
        self.alpha_bar = alpha_bar
        self.beta_bar = beta_bar

        # self.alpha_bar = 0.
        # self.beta_bar = 0.

        self.r_mass = r_mass
        self.r_thickness = r_thickness
        self.rope_initpose = np.array([
            0., 0.0, 0.5,
            1., 0., 0., 0.
        ])
        self.rope_initpose[0] += self.r_len/2
        self.overall_rot = 0.0 # 27 * (2*np.pi) # 57 * (np.pi/180)
        if overall_rot is not None and not manual_rot:
            self.overall_rot = overall_rot
            self.overall_rot_tmp = 0.0
        if manual_rot:
            self.overall_rot_tmp = overall_rot
        self.overall_rot_4mconst = overall_rot

        # init stiffnesses for capsule
        J1 = np.pi * (self.r_thickness/2)**4/2.
        Ix = np.pi * (self.r_thickness/2)**4/4.
        self.stiff_vals = [
            self.beta_bar/J1,
            self.alpha_bar/Ix
        ]

        xml, arm_xml = self._get_xmlstr()

        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.sim = MjSimWrapper(self.model, self.data)

        if self.do_render:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.rend_rate = int(10)
            self.set_viewer_details(
                dist=1.5,
                azi=90.0,
                elev=0.0,
                lookat=np.array([-0.75,0.0,0.10])
            )

            # self.viewer.cam.distance = 5.7628
            # self.viewer.cam.azimuth = -40.478
            # self.viewer.cam.elevation = -12.434
            # self.viewer.cam.lookat = np.array([-1.17009866, -1.37107526,  0.02327594])
            # self.do_render=False
        else:
            self.viewer = None

        # enable joint visualization option:
        self.scene_option = mujoco.MjvOption()
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        # other variables
        self.max_env_steps = 10000000
        self.env_steps = 0
        self.cur_time = 0
        self.dt = self.model.opt.timestep
        self.freq_velreset = 0.2

        # init obs
        self.observations = dict(
            rope_pose=np.zeros((self.r_pieces,3)),
        )

        # init gravity
        if grav_on:
            self.model.opt.gravity[-1] = -9.81
        else:
            self.model.opt.gravity[-1] = 0.0

        self.sim.forward()

        self._init_ids()
        self._get_observations()
        # ropeend_id = mjc2.obj_name2id(self.model, "body", "eef_body")
        # eef_quat1 = np.array(
        #     self.data.xquat[ropeend_id]
        # )
        # print(eef_quat1)
        # print(T.axisangle2quat(T.quat_error(eef_quat1,eef_quat2)))

        # init dlo controller
        self.f_limit = 1000.
        if self.storqtype == 'xfrc':
            self.dlo_sim = DLORopeXfrc(
                model=self.model,
                data=self.data,
                n_link=self.r_pieces,
                alpha_bar=self.alpha_bar,
                beta_bar=self.beta_bar,
                overall_rot=self.overall_rot,
                f_limit=self.f_limit,
            )
            self.joint_qveladdr_full = self.dlo_sim.dlo_joint_qveladdr_full.copy()
        elif self.storqtype == 'adapt':
            self.dlo_sim = DLORopeAdapt(
                model=self.model,
                data=self.data,
                n_link=self.r_pieces,
                alpha_bar=self.alpha_bar,
                beta_bar=self.beta_bar,
                overall_rot=self.overall_rot,
                f_limit=self.f_limit,
                bothweld=self.bothweld
            )
            self.joint_qveladdr_full = self.dlo_sim.dlo_joint_qveladdr_full.copy()
        # if self.do_render:
        #     self.viewer._paused = True

        if (self.test_type == 'lhb'): # or (self.test_type == 'speedtest2'):
            self.start_lhbtest(new_start)
        elif self.test_type == 'mbi':
            self.start_circletest(new_start)
        else:
            self.freq_velreset = 0.2
        # # pickle stuff
        # self._init_pickletool()
        # self._save_initpickle()
        # self._load_initpickle()

    def _init_ids(self):
        if self.storqtype != 'native':
            self.joint_site_idx = np.zeros(self.r_pieces+1, dtype=int)
            for i_sec in range(self.r_pieces):
                self.joint_site_idx[i_sec] = mjc2.obj_name2id(
                    self.model,
                    "site",
                    'S_{}'.format(i_sec)
                )
            self.joint_site_idx[self.r_pieces] = mjc2.obj_name2id(
                self.model,
                "site",
                'S_{}'.format('last')
            )


        # init vec_bodyid for cable
        self.vec_bodyid = np.zeros(self.r_pieces, dtype=int)
        for i in range(self.r_pieces):
            if i == 0:
                i_name = 'first'
            elif i == self.r_pieces-1:
                i_name = 'last'
            else:
                i_name = str(i)
            self.vec_bodyid[i] = mjc2.obj_name2id(
                self.model,"body",'B_' + i_name
            )
        self.eef_site_idx = mjc2.obj_name2id(
            self.model,
            "site",
            'S_last'
        )
        self.ropeend_body_id = mjc2.obj_name2id(self.model,"body","stiffrope")
        if self.storqtype == 'native':
            self.ropeend_getstate_bodyid = mjc2.obj_name2id(self.model,"body","stiffrope")
    
            self.joint_ids = []
            self.joint_qveladdr = []
            for i in range(1, self.r_pieces):    # from freejoint2 to freejoint(n_links)
                fj_str = f'J_{i}'
                if i == (self.r_pieces-1):
                    fj_str = 'J_last'
                self.joint_ids.append(mjc2.obj_name2id(
                    self.model,"joint",fj_str
                ))
                # if self.model.jnt_type(
                    # self.dlo_joint_ids[-1]
                # ) == self.model.mjtJoint.mjJNT_FREE:
                    # print('it is free joint')
                    # print(fj_str)
                    # print(self.model.jnt_dofadr[self.dlo_joint_ids[-1]])
                self.joint_qveladdr.append(
                    self.model.jnt_dofadr[self.joint_ids[-1]]
                )
            self.joint_qveladdr = np.array(self.joint_qveladdr)
            self.joint_qveladdr_full = [
                n for n in range(
                    self.joint_qveladdr[0],
                    self.joint_qveladdr[-1] + 3
                )
            ]

    def _get_xmlstr(self):
        # load model
        # update rope model
        if self.storqtype == 'native':
            rope_xml_file = "nativerope1dkin.xml"
            overall_file = "overall_native.xml"
        else:
            rope_xml_file = "dlorope1dkin.xml"
            overall_file = "overall.xml"
        world_base_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets/world_realexp.xml"
        )
        box_path = os.path.join(
            os.path.dirname(world_base_path),
            "anchorbox.xml"
        )
        weldweight_path = os.path.join(
            os.path.dirname(world_base_path),
            "weldweight.xml"
        )
        rope_path = os.path.join(
            os.path.dirname(world_base_path),
            rope_xml_file
        )

        self.bothweld = False
        if self.storqtype=='native':
            GenKin_N(
                r_len=self.r_len,
                r_thickness=self.r_thickness,
                r_pieces=self.r_pieces,
                r_mass=self.r_mass,
                stiff_vals=self.stiff_vals,
                j_damp=0.01,
                init_pos=self.rope_initpose[:3],
                init_quat=self.rope_initpose[3:],
                coll_on=True,
                rope_type="capsule",
                vis_subcyl=False,
                obj_path=rope_path,
                rgba_vals=self.rgba_vals
            )
        if self.storqtype=='xfrc':
            GenKin_O_xfrc(
                r_len=self.r_len,
                r_thickness=self.r_thickness,
                r_pieces=self.r_pieces,
                r_mass=self.r_mass,
                j_stiff=0.0,
                j_damp=0.01,
                init_pos=self.rope_initpose[:3],
                init_quat=self.rope_initpose[3:],
                coll_on=True,
                d_small=0.,
                rope_type="capsule",
                vis_subcyl=False,
                obj_path=rope_path,
                rgba_vals=self.rgba_vals
            )
        if self.storqtype=='adapt':
            GenKin_O(
                r_len=self.r_len,
                r_thickness=self.r_thickness,
                r_pieces=self.r_pieces,
                r_mass=self.r_mass,
                j_stiff=0.0,
                j_damp=0.01,
                init_pos=self.rope_initpose[:3],
                init_quat=self.rope_initpose[3:],
                coll_on=True,
                d_small=0.,
                rope_type="capsule",
                vis_subcyl=False,
                obj_path=rope_path,
                rgba_vals=self.rgba_vals
            )

        self.xml = XMLWrapper(world_base_path)
        dlorope = XMLWrapper(rope_path)
        anchorbox = XMLWrapper(box_path)
        weldweight = XMLWrapper(weldweight_path)

        if self.test_type == 'mbi':
            dlorope.merge(
                weldweight,
                element_name="body",
                attrib_name="name",
                attrib_value="B_last",
                action="append",
            )

        self.xml.merge_multiple(
            anchorbox, ["worldbody", "equality", "contact"]
        )
        if self.storqtype == "native":
            self.xml.merge_multiple(
                dlorope, ["worldbody","extension"]
            )
        else:
            self.xml.merge_multiple(
                dlorope, ["worldbody"]
            )
        asset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets/" + overall_file
        )

        xml_string = self.xml.get_xml_string()

        model = mujoco.MjModel.from_xml_string(xml_string)
        mujoco.mj_saveLastXML(asset_path,model)

        return xml_string, None

    def step(self, action=np.zeros(6)):
        # t1 = time()
        # print(f"self.alpha_bar = {self.alpha_bar}")
        # print(f"self.beta_bar = {self.beta_bar}")
        if self.storqtype == 'xfrc':
            self.dlo_sim.update_force()
        elif self.storqtype == 'adapt':
            self.dlo_sim.update_torque()

        # print(self.data.ncon)
        # print(mjc2.obj_id2name(self.model,"geom",self.data.contact[0].geom[0]))
        # print(mjc2.obj_id2name(self.model,"geom",self.data.contact[0].geom[1]))

        # self.dlo_sim.reset_qvel_rotx()
        # t2 = time()
        self.sim.step()
        self.sim.forward()
        # t3 = time()
        if self.do_render:
            if self.env_steps%self.rend_rate==0:
                self.viewer.render()
                # self.print_viewer_details()
        # if self.env_steps%10000==0:
        #     self.viewer.render()

        self.env_steps += 1
        self.cur_time += self.dt

        done = self.env_steps > self.max_env_steps

        if self.instability_check():
            print(f'unstable {self.env_steps}')
            self.viewer._paused = True
            # input()
            # if self.do_render:
                # self.viewer.render()
                # self.viewer._paused = True

        # # print times
        # print(t2-t1)
        # print(t3-t2)

        return self._get_observations(), 0, done, False, 0

    def _get_observations(self):
        self.observations['rope_pose'] = np.concatenate((
            self.data.xpos[
                self.vec_bodyid[:self.r_pieces]
            ].copy(),
            [self.data.site_xpos[
                self.eef_site_idx
            ].copy()]
        ))
        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        if self.viewer == None:
            if self.do_render:
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        else:
            if not self.do_render:
                self.viewer.close()
                self.viewer = None

        if self.storqtype == 'native':
            self.sim.forward()
        else:
            self.dlo_sim.reset_body()
            self.sim.forward()
            self.dlo_sim.reset_sim()

        # reset obs
        self.observations = dict(
            rope_pose=np.zeros((self.r_pieces,3)),
        )

        # reset time
        self.cur_time = 0   #clock time of episode
        self.env_steps = 0

        # pickle
        self._load_initpickle()

        return None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| External Funcs ||~~~~~~~~~~~~~~~~~~~~~~~~~~

    def hold_pos(self, hold_time=2., rend=False):
        init_time = self.cur_time
        # self.print_collisions()
        # if rend and self.do_render:
        #     self.viewer.render()
        while (self.cur_time-init_time) <= hold_time:
            # print(f"self.cur_time = {self.cur_time}")
            # body_id3 = mjc2.obj_name2id(self.model,"body","stiffrope")
            # print(self.model.body_pos[body_id3][:])
            # body_id3 = mjc2.obj_name2id(self.model,"body","eef_body")
            # print(self.model.body_pos[body_id3][:])
            # start_t = time()
            self.step()
            # print(time()-start_t)
            # print(self.cur_time-init_time)
            # print(hold_time)
            if self.env_steps%self.freq_velreset==0:
                self.reset_vel()

    def instability_check(self):
        if np.linalg.norm(self.data.qvel[self.joint_qveladdr_full]) > 300.:
            return True
        return False

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Pickle Stuff ||~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _init_pickletool(self):
        self.ocvt_picklepath = 'ocvt' # 'rob3.pickle'
        self.ocvt_picklepath = self.ocvt_picklepath + '.pickle'
        self.ocvt_picklepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/" + self.ocvt_picklepath
        )

    def _save_initpickle(self):
        ## initial movements
        self.hold_pos(10.)

        ## create pickle
        if self.storqtype == 'native':
            self.init_pickle = self.get_state2()
        else:
            self.init_pickle = self.get_state()

        ## save pickle
        with open(self.ocvt_picklepath, 'wb') as f:
            pickle.dump(self.init_pickle,f)
        input('Pickle saved!')

    def _load_initpickle(self):
        with open(self.ocvt_picklepath, 'rb') as f:
            self.init_pickle = pickle.load(f)
        if self.storqtype == 'native':
            self.set_state2(self.init_pickle)
        else:
            self.set_state(self.init_pickle)

    def get_state(self):
        (
            ropeend_pos,
            ropeend_quat,
            overall_rot,
            p_thetan
        ) = self.dlo_sim.get_dlosim()

        state = np.empty(
            mujoco.mj_stateSize(
                self.model,
                mujoco.mjtState.mjSTATE_PHYSICS
            )
        )
        mujoco.mj_getState(
            self.model, self.data, state,
            spec=mujoco.mjtState.mjSTATE_PHYSICS
        )

        bodypose_data = [
            self.model.body_pos[:].copy(),
            self.model.body_quat[:].copy()
        ]
        return [
            np.concatenate((
                [0], # [self.cur_time],
                [0], # [self.env_steps],
                ropeend_pos,
                ropeend_quat,
                [overall_rot],
                [p_thetan],
            )),
            state,
            bodypose_data
        ]

    def set_state(self, p_state):
        self.cur_time = 0 # p_state[0][0]
        self.env_steps = 0 # p_state[0][1]
        self.dlo_sim.set_dlosim(
            p_state[0][2:5],
            p_state[0][5:9],
            # 27. * (2.*np.pi),
            p_state[0][9],
            p_state[0][10],
        )
        mujoco.mj_setState(
            self.model, self.data, p_state[1],
            spec=mujoco.mjtState.mjSTATE_PHYSICS
        )
        self.model.body_pos[:] = p_state[2][0]
        self.model.body_quat[:] = p_state[2][1]

        # self.dlo_sim.overall_rot = 27. * (2.*np.pi)
        # self.dlo_sim.dlo_math.resetTheta(
        #     self.dlo_sim.p_thetan, self.dlo_sim.overall_rot
        # )

        # self.sim.forward()
        self.step()
        self._get_observations()
        self.dlo_sim._update_xvecs()

    def get_state2(self):
        ropeend_pos = self.model.body_pos[
            self.ropeend_getstate_bodyid,:
        ].copy()
        ropeend_quat = self.model.body_quat[
            self.ropeend_getstate_bodyid,:
        ].copy()

        state = np.empty(
            mujoco.mj_stateSize(
                self.model,
                mujoco.mjtState.mjSTATE_PHYSICS
            )
        )
        mujoco.mj_getState(
            self.model, self.data, state,
            spec=mujoco.mjtState.mjSTATE_PHYSICS
        )

        bodypose_data = [
            self.model.body_pos[:].copy(),
            self.model.body_quat[:].copy()
        ]
        return [
            np.concatenate((
                [0], # [self.cur_time],
                [0], # [self.env_steps],
                ropeend_pos,
                ropeend_quat,
            )),
            state,
            bodypose_data
        ]
    def set_state2(self, p_state):
        self.cur_time = 0 # p_state[0][0]
        self.env_steps = 0 # p_state[0][1]
        self.model.body_pos[
            self.ropeend_getstate_bodyid,:
        ] = p_state[0][2:5]
        self.model.body_quat[
            self.ropeend_getstate_bodyid,:
        ] = p_state[0][5:9]
        mujoco.mj_setState(
            self.model, self.data, p_state[1],
            spec=mujoco.mjtState.mjSTATE_PHYSICS
        )
        self.model.body_pos[:] = p_state[2][0]
        self.model.body_quat[:] = p_state[2][1]
        # self.dlo_sim.overall_rot = 27. * (2.*np.pi)
        # self.dlo_sim.dlo_math.resetTheta(
        #     self.dlo_sim.p_thetan, self.dlo_sim.overall_rot
        # )

        # self.sim.forward()
        self.step()
        self._get_observations()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| End Pickle Stuff ||~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Validation Stuff ||~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Circle ||~~~~~~~~~~~~~~~~~~~~~~~~~~

    def circle_init(self):
        # assuming that rope axis is parallel to x axis
        # self.hold_pos(10.)
        holder_offset = np.array([0.12,0.0,0.0])
        self.stablestep = False
        y_offset = 0.3
        start_pos = self.data.xpos[
            mjc2.obj_name2id(
                self.model, 'body', 'stiffrope'
            )
        ]
        end_pos = self.data.xpos[
            mjc2.obj_name2id(
                self.model, 'body',
                'B_last'
            )
        ]
        e_pos = end_pos - start_pos
        e_pos -= holder_offset
        e_pos += self.r_len/self.r_pieces
        step_len = 5e-3
        n_steps = e_pos[0] / step_len
        if n_steps < 0:
            step_len = - step_len
            n_steps = int(-n_steps)
        step_ylen = 2 * y_offset / n_steps
        n_steps = int(n_steps/2)
        step_remain = e_pos[0] - 2* n_steps * step_len
        for i in range(2):
            self.ropeend_pos(np.array([step_len, step_ylen, step_ylen]))
        for i in range(2):
            self.ropeend_pos(np.array([step_len, step_ylen, -step_ylen]))

        for i in range(n_steps-4):
            print(f"steps: {i+1} / {n_steps-4}")
            # input('stepped')
            # self.ropeend_pos(np.array([0., 5*step_ylen, 0.]), bodymove_name="anchor_box")
            # self.ropeend_pos_all(np.array([step_len, step_ylen, 0.]))
            self.ropeend_pos(np.array([step_len, step_ylen, 0.]))
        for i in range(360):
            print(f"steps: {i+1} / {360}")
            self.ropeend_rot(rot_axis=2)
        for i in range(n_steps):
            print(f"steps: {i+1} / {n_steps}")
            self.ropeend_pos(np.array([step_len, 0., 0.]))
        self.ropeend_pos(np.array([step_remain, 0., 0.]))
        for i in range(n_steps):
            print(f"steps: {i+1} / {n_steps}")
            self.ropeend_pos(np.array([0., -step_ylen, 0.]))

        self.stablestep = False
        self.hold_pos()
        self.reset_vel()
        self.hold_pos()
        for i in range(50):
            self.hold_pos(0.3)
            self.reset_vel()

    def check_e_PCA_circle(self):
        if self.storqtype == 'native':
            joint_site_pos = self.observations['rope_pose'].copy()
        else:
            joint_site_pos = np.array(
                self.data.site_xpos[self.joint_site_idx[:]]
            ).copy()
        joint_site_pos = joint_site_pos[:self.r_pieces-1]
        # exclude 2 nodes to account for overlap
        return compute_PCA(
            joint_site_pos[:,0],
            joint_site_pos[:,1],
            joint_site_pos[:,2],
        )

    def start_circletest(self, new_start):
        # if self.do_render:
        #     self.set_viewer_details(
        #         4.5076,
        #         45.922,
        #         -19.810,
        #         np.array([-3.46078809, -0.26378238, 0.63761223])
        #     )
        # Test for multiple rope types (vary alpha and beta bar)
        # alpha = 1.
        # r_len = 2*np.pi * self.r_pieces / (self.r_pieces-1)
        self.freq_velreset = 0.2
        self.stable_bool = True
        e_tol = 7.5 # 0.075
        circtest_picklename = 'mbitest1.pickle'
        circtest_picklename = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/mbi/" + self.picklefolder + "/" + circtest_picklename
        )
        if new_start:
            self.circle_oop = False
            self.circle_init()
            if self.storqtype == 'native':
                self.init_pickle = self.get_state2()
            else:
                self.init_pickle = self.get_state()
            with open(circtest_picklename, 'wb') as f:
                pickle.dump(self.init_pickle,f)
            input(f'Circle Pickle saved!')
        else:
            with open(circtest_picklename, 'rb') as f:
                self.init_pickle = pickle.load(f)

            if self.storqtype == 'native':
                self.set_state2(self.init_pickle)
            else:
                # set overall rot
                self.init_pickle[0][9] = self.overall_rot
                self.p_thetan = self.overall_rot % (2. * np.pi)
                if self.p_thetan > np.pi:
                    self.p_thetan -= 2 * np.pi
                self.init_pickle[0][10] = self.p_thetan
                self.set_state(self.init_pickle)

            if self.storqtype == 'native':
                self.rot_x_rads2(x_rads=self.overall_rot)
                self.reset_vel()
            else:
                self.rot_x_rads(x_rads=self.overall_rot_tmp)

            # # get and apply force normal to the circle
            norm_force = self.get_rope_normal()
            # self.apply_force_t(t=0.3,force_dir=np.array([0., 0., 1.]))
            # self.apply_force_t(t=0.8,force_dir=norm_force)

            # self.hold_pos(100.)
            # create a for loop that checks PCA error at each iter
            max_e = 0.
            e_outofplane = 0.
            self.circle_oop = False
            t_step = 0.2
            hold_t = 30
            hold_steps = int(hold_t / t_step)
            print('0')
            for i in range(hold_steps):
                # self.ropeend_rot(rot_axis=0)
                # if not i % int(1.76*360) and i != 0:
                #     self.reset_vel()
                #     self.hold_pos(10.)
                # if not i % 50:
                #     self.reset_vel()
                #     self.hold_pos(0.3)
                # self.reset_vel()
                sys.stdout.write(f"\033[{1}F")
                # print(f"e_outofplane = {e_outofplane}")
                print(f"Holding for {i+1} / {hold_steps}..")
                self.hold_pos(t_step, rend=True)
                if self.instability_check():
                    self.stable_bool = False
                e_outofplane = self.check_e_PCA_circle()
                # print(self.data.contact)
                if e_outofplane > max_e:
                    max_e = e_outofplane
                if max_e > e_tol or self.data.ncon > 0:
                    # print(f"e_tol = {e_tol}")
                    # print(f"e_outofplane = {e_outofplane}")
                    # print(f"max_e = {max_e}")

                    self.circle_oop = True
                    # print(f'b_a = {self.beta_bar/self.alpha_bar} ==================================')
                    # print(f'out of plane theta_crit = {self.overall_rot} ==================================')
                    return e_outofplane
                # print(f'e_outofplane = {e_outofplane}')
            # print(f"e_tol = {e_tol}")
            # print(f"e_outofplane = {e_outofplane}")
            # print(f"max_e = {max_e}")
            # if e_outofplane > e_tol or max_e > 5:
            if max_e > e_tol:
                self.circle_oop = True
                # print(f'b_a = {self.beta_bar/self.alpha_bar} ==================================')
                # print(f'out of plane theta_crit = {self.overall_rot} ==================================')
                # print(f'e_outofplane = {e_outofplane}')
            if not self.stable_bool:
                self.circle_oop = True
                input('Unstable sim: press "Enter" to continue..')
            return e_outofplane

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Utils ||~~~~~~~~~~~~~~~~~~~~~~~~~~
    def print_viewer_details(self):
        print(f"distance = {self.viewer.cam.distance}")
        print(f"azimuth = {self.viewer.cam.azimuth}")
        print(f"elevation = {self.viewer.cam.elevation}")
        print(f"lookat = {self.viewer.cam.lookat}")

    def set_viewer_details(
        self,
        dist, azi, elev, lookat      
    ):
        self.viewer.cam.distance = dist
        self.viewer.cam.azimuth = azi
        self.viewer.cam.elevation = elev
        self.viewer.cam.lookat = lookat

    def reset_vel(self):
        self.data.qvel[:] = np.zeros(len(self.data.qvel[:]))
        self.sim.forward()
        # pass

    def rot_x_rads(self, x_rads):
        n_rotsteps = int(x_rads/(np.pi)*180)
        rad_leftover = (
            x_rads
            - (n_rotsteps / 180 * np.pi)
        )
        if n_rotsteps > 0:
            print('0')
        for i in range(n_rotsteps):
            sys.stdout.write(f"\033[{1}F")
            print(f"init rot stage (degs): {i+1}/{n_rotsteps}")
            self.ropeend_rot(rot_axis=0)
        self.ropeend_rot(rot_a=rad_leftover,rot_axis=0)

    def rot_x_rads2(self, x_rads):
        x_rads /= 2.0   # rotate from both ends
        body_id = mjc2.obj_name2id(self.model,"body","eef_body")

        n_rotsteps = int(x_rads/(np.pi)*180)
        rad_leftover = (
            x_rads
            - (n_rotsteps / 180 * np.pi)
        )
        if n_rotsteps > 0:
            print('0')
        for i in range(n_rotsteps):
            sys.stdout.write(f"\033[{1}F")
            print(f"init rot stage (degs): {2*(i+1)}/{2*n_rotsteps}")
            self.ropeend_rot2(body_id, rot_axis=0)
        self.ropeend_rot2(body_id, rot_a=rad_leftover, rot_axis=0)

    def ropeend_rot(self, rot_a=np.pi/180, rot_axis=0):
        rot_arr = np.zeros(3)
        rot_arr[rot_axis] = rot_a
        rot_quat = T.axisangle2quat(rot_arr)
        new_quat = T.quat_multiply(rot_quat, self.model.body_quat[self.ropeend_body_id])
        self.model.body_quat[self.ropeend_body_id] = new_quat
        self.hold_pos(0.02)

    def ropeend_rot2(self, body_id, rot_a=np.pi/180, rot_axis=0):
        rot_arr = np.zeros(3)
        rot_arr[rot_axis] = rot_a
        rot_quat = T.axisangle2quat(rot_arr)
        rot_quat2 = T.axisangle2quat(-rot_arr)
        new_quat = T.quat_multiply(rot_quat, self.model.body_quat[self.ropeend_body_id])
        new_quat2 = T.quat_multiply(rot_quat2, self.model.body_quat[body_id])
        self.model.body_quat[self.ropeend_body_id] = new_quat
        self.model.body_quat[body_id] = new_quat2
        self.hold_pos(0.1)

    def ropeend_pos(
            self,
            pos_move=np.array([0., -1e-4, 0.]),
            # bodymove_name="stiffrope"
        ):
        self.model.body_pos[self.ropeend_body_id][:] += pos_move.copy()
        self.hold_pos(0.5)

    def ropeend_pos_all(
        self,
        pos_move=np.array([0., -1e-4, 0.]),
    ):
        body_id2 = mjc2.obj_name2id(self.model,"body","eef_body")
        self.model.body_pos[body_id2][:] += pos_move
        body_id3 = self.ropeend_body_id
        self.model.body_pos[body_id3][:] -= pos_move
        # self.hold_pos(0.3)
        self.hold_pos(0.5)

    def apply_force(
        self,
        body_name=None,
        force_dir=np.array([0., 0., 0.01]),
    ):
        if body_name is None:
            body_name = "B_{}".format(int(self.r_pieces/2)+1)
        # print(self.sim.data.xfrc_applied[i])
        body_id = mjc2.obj_name2id(self.model,"body",body_name)
        # print(self.model.body_pos[body_id])
        # print(self.sim.data.xfrc_applied[body_id,:3])
        self.data.xfrc_applied[body_id,:3] = force_dir
        # print("Giving force:")
        # print(self.data.xfrc_applied[body_id,:3])

    def apply_force_t(
        self,
        t,
        body_name=None,
        force_dir=np.array([0., 0., 0.01]),
    ):
        # apply force for a period of t
        start_t = self.cur_time
        while (self.cur_time-start_t) < t:
            self.apply_force(body_name=body_name,force_dir=force_dir)
            self.step()
        self.apply_force(body_name=body_name,force_dir=np.zeros(3))

    def get_rope_normal(self):
        # assuming rope is in a flat with two ends connected
        # takes equally-spaced points along circumference
        # to get normal to circle
        circum_pts = [
            [0, int(self.r_pieces/2)],
            [int(self.r_pieces/4), int(3*self.r_pieces/4)],
        ]
        circ_norm = np.cross(
            (
                self.observations['rope_pose'][circum_pts[0][0]]
                - self.observations['rope_pose'][circum_pts[0][1]]
            ),
            (
                self.observations['rope_pose'][circum_pts[1][0]]
                - self.observations['rope_pose'][circum_pts[1][1]]
            ),
        )
        circ_norm /= np.linalg.norm(circ_norm)
        return circ_norm

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| End Validation Stuff ||~~~~~~~~~~~~~~~~~~~~~~~~~~

# TestEnv()

"""
[[ 1.80432751e-03 -9.99979560e-01 -6.13380804e-03  4.92719293e-01]
 [-9.99998236e-01 -1.80108827e-03 -5.33578529e-04  1.34835376e-01]
 [ 5.22520093e-04  6.13475997e-03 -9.99981046e-01  4.85597924e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
[[ 9.99998212e-01  1.80113071e-03  5.33586426e-04 -1.09985694e-01]
 [ 1.80437008e-03 -9.99979556e-01 -6.13383343e-03  4.87772733e-01]
 [ 5.22527727e-04  6.13478571e-03 -9.99981046e-01  4.29488271e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
[[ 2.30518624e-07 -9.99977864e-01 -6.65727793e-03  6.02096427e-01]
 [ 9.99984255e-01  3.75928764e-05 -5.61214931e-03  3.54359916e-01]
 [ 5.61227553e-03 -6.65717169e-03  9.99962091e-01 -5.36221347e-02]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]

"""