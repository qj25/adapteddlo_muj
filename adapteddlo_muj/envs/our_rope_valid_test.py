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
from adapteddlo_muj.assets.genrope.gdv_O_weld2 import GenKin_O_weld2
from adapteddlo_muj.controllers.ropekin_controller_adapt import DLORopeAdapt
from adapteddlo_muj.utils.data_utils import compute_PCA, centralize_devdata


class TestRopeEnv(gym.Env, utils.EzPickle):
    def __init__(
        self,
        do_render=False,
        alpha_bar=1.345/50,   # 1.345/50,
        beta_bar=0.789/50,    # 0.789/50,
        r_pieces=30,    # max. 33
        r_mass=0.58,
        r_len = 2*np.pi,
        r_thickness=0.03,
        test_type=None,
        overall_rot=None,
        new_start=False,
        limit_f=False,
    ):
        utils.EzPickle.__init__(self)

        self.do_render = do_render
        self.test_type = test_type
        self.limit_f = limit_f
        self.picklefolder = 'adapt'

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
        self.overall_rot = 0. # 27 * (2*np.pi) # 57 * (np.pi/180)
        if overall_rot is not None:
            self.overall_rot = overall_rot

        xml, arm_xml = self._get_xmlstr()

        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.sim = MjSimWrapper(self.model, self.data)

        if self.do_render:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.rend_rate = int(10)

            self.viewer.cam.distance = 5.7628
            self.viewer.cam.azimuth = -40.478
            self.viewer.cam.elevation = -12.434
            self.viewer.cam.lookat = np.array([-1.17009866, -1.37107526,  0.02327594])
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
        if (self.test_type == 'lhb') or (self.test_type == 'speedtest2'):
            self.ropeend_body_id = mjc2.obj_name2id(self.model,"body","eef_body2")
        else:
            self.ropeend_body_id = mjc2.obj_name2id(self.model,"body","stiffrope")


    def _get_xmlstr(self):
        # load model
        # update rope model
        world_base_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets/world_test.xml"
        )
        box_path = os.path.join(
            os.path.dirname(world_base_path),
            "anchorbox.xml"
        )
        rope_path = os.path.join(
            os.path.dirname(world_base_path),
            "dlorope1dkin.xml"
        )

        if (self.test_type == 'lhb') or (self.test_type == 'speedtest2'):
            self.bothweld = True
            GenKin_O_weld2(
                r_len=self.r_len,
                r_thickness=self.r_thickness,
                r_pieces=self.r_pieces,
                # r_mass=self.r_mass,
                j_stiff=0.0,
                j_damp=0.5,
                init_pos=self.rope_initpose[:3],
                init_quat=self.rope_initpose[3:],
                d_small=0.,
                rope_type="capsule",
                vis_subcyl=False,
                obj_path=rope_path,
            )
        elif self.test_type == 'mbi':
            self.bothweld = False
            GenKin_O(
                r_len=self.r_len,
                r_thickness=self.r_thickness,
                r_pieces=self.r_pieces,
                # r_mass=self.r_mass,
                j_stiff=0.0,
                j_damp=0.5,
                init_pos=self.rope_initpose[:3],
                init_quat=self.rope_initpose[3:],
                coll_on=True,
                d_small=0.,
                rope_type="capsule",
                vis_subcyl=False,
                obj_path=rope_path,
            )
        elif self.test_type == 'speedtest1':
            # j_damp = self.r_len / 9.29
            j_damp = 0.5
            GenKin_O(
                r_len=self.r_len,
                r_thickness=self.r_thickness,
                r_pieces=self.r_pieces,
                # r_mass=self.r_mass,
                j_stiff=0.0,
                j_damp=j_damp,
                init_pos=self.rope_initpose[:3],
                init_quat=self.rope_initpose[3:],
                d_small=0.,
                rope_type="capsule",
                vis_subcyl=False,
                obj_path=rope_path,
            )
        else:
            input(f'Invalid test_type: {self.test_type}')

        self.xml = XMLWrapper(world_base_path)
        dlorope = XMLWrapper(rope_path)
        anchorbox = XMLWrapper(box_path)

        self.xml.merge_multiple(
            anchorbox, ["worldbody", "equality", "contact"]
        )
        self.xml.merge_multiple(
            dlorope, ["worldbody"]
        )
        asset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets/overall.xml"
        )

        xml_string = self.xml.get_xml_string()

        model = mujoco.MjModel.from_xml_string(xml_string)
        mujoco.mj_saveLastXML(asset_path,model)

        return xml_string, None

    def step(self, action=np.zeros(6)):
        # t1 = time()
        self.dlo_sim.update_torque()

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
        if np.linalg.norm(self.data.qvel[self.dlo_sim.dlo_joint_qveladdr_full]) > 300.:
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
        self.init_pickle = self.get_state()

        ## save pickle
        with open(self.ocvt_picklepath, 'wb') as f:
            pickle.dump(self.init_pickle,f)
        input('Pickle saved!')

    def _load_initpickle(self):
        with open(self.ocvt_picklepath, 'rb') as f:
            self.init_pickle = pickle.load(f)
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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| End Pickle Stuff ||~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Validation Stuff ||~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Lhb things ||~~~~~~~~~~~~~~~~~~~~~~~~~~

    def lhb_init(self):
        # Twist 27 turns (27 * 360)
        s_ss_center = None
        s_ss_mincenter = 10.
        l_shorten = 0.3
        n_steps = 100
        step_len = l_shorten / n_steps / 2
        # n_steps = int(l_shorten / step_len)
        # print(self.sitename2pos("r_joint0_site"))
        # print(self.sitename2pos("r_joint180_site"))
        # print(self.sitename2pos("r_joint0_site")-self.sitename2pos("r_joint180_site"))
        pos_move = np.array([step_len, -step_len, -step_len])
        for i in range(1):
            self.ropeend_pos_all(pos_move=pos_move.copy())
        # pos_move = np.array([0., -step_len, 0.])
        # self.ropeend_pos_all(pos_move=-pos_move.copy())
        # self.ropeend_pos_all(pos_move=pos_move.copy())
        pos_move = np.array([step_len, step_len, step_len])
        for i in range(1):
            self.ropeend_pos_all(pos_move=pos_move.copy())

        pos_move = np.array([step_len, 0., 0.])
        print('0')
        for i in range(n_steps-2):
            sys.stdout.write(f"\033[{1}F")
            print(f"init stage: {i+1}/{n_steps-2}")
            # self.reset_vel()
            # print(self.sitename2pos("r_joint0_site"))
            # print(self.sitename2pos("r_joint180_site"))
            # print(self.sitename2pos("r_joint0_site")-self.sitename2pos("r_joint180_site"))
            self.ropeend_pos_all(pos_move=pos_move.copy())
            # self.ropeend_pos(pos_move=pos_move.copy())

            # print(i)
            # if i % 1 == 0:
                # self.reset_vel()
        # for i in range(27*360):
        #     self.ropeend_rot(rot_axis=0)
        #     self.reset_vel()
        # step_remain = l_shorten - 2* n_steps * step_len
        # self.ropeend_pos(np.array([step_remain, 0., 0.]))
        # print(self.sitename2pos("r_joint0_site"))
        # print(self.sitename2pos("r_joint0_site"))
        # print(self.sitename2pos("r_joint0_site")-self.sitename2pos("r_joint180_site"))

    def lhb_testing(self):
        print('0')
        n_holdsteps = 50
        for i in range(n_holdsteps):
            sys.stdout.write(f"\033[{1}F")
            print(f"testing stage: {i+1}/{n_holdsteps}")
            self.hold_pos(0.2)
            # if i % 1 == 0:
            #     self.reset_vel()
        # self.reset_vel()
        # hold_time = 30.
        # init_time = self.cur_time
        # while (self.cur_time-init_time) < hold_time:
        #     self.step()

        s_ss, fphi, max_devi = self.lhb_var_compute()
        s_ss_center = s_ss.copy()
        fphi_center = fphi.copy()
        # fphi_min_id = np.where(fphi == fphi.min())[0][0]
        # s_ss_min_id = floor(len(s_ss) / 2)
        # s_ss_center = s_ss - s_ss[s_ss_min_id]
        # fphi_center = fphi.copy()
        # min_id_diff = fphi_min_id - s_ss_min_id
        # if min_id_diff > 0:
        #     fphi_center[:-min_id_diff] = fphi[min_id_diff:]
        #     fphi_center[-min_id_diff:] = np.ones(min_id_diff)
        # elif min_id_diff < 0:
        #     fphi_center[-min_id_diff:] = fphi[:min_id_diff]
        #     fphi_center[:-min_id_diff] = np.ones(-min_id_diff)
        # while (self.cur_time-init_time) < hold_time:
        #     self.step()
        #     s_ss, fphi = self.lhb_var_compute()
        #     maxdevi_id = np.where(fphi == fphi.min())[0][0]
        #     s_ss_maxdevi = abs(s_ss[maxdevi_id])
        #     if s_ss_maxdevi < s_ss_mincenter or s_ss_center is None:
        #         s_ss_mincenter = s_ss_maxdevi
        #         s_ss_center = s_ss.copy()
        #         fphi_center = fphi.copy()
        return s_ss_center, fphi_center, max_devi

    def lhb_var_compute(self):
        joint_site_pos = np.array(
            self.data.site_xpos[self.joint_site_idx[:]]
        ).copy()
        t_vec = joint_site_pos[1:] - joint_site_pos[:self.r_pieces]
        for i in range(len(t_vec)):
            t_vec[i] = t_vec[i] / np.linalg.norm(t_vec[i])
        e_x = np.array([-1., 0., 0.])
        devi_set = np.arccos(np.dot(t_vec, e_x))
        s_step = self.r_len / (self.r_pieces)
        s = np.zeros(len(devi_set))
        for i in range(len(devi_set)):
            s[i] = s_step * i + s_step/2
        s = s - np.mean(s)
        s, devi_set = centralize_devdata(s, devi_set)
        max_devi = np.max(devi_set)
        max_devi2 = max_devi
        # max_devi2 = 0.919
        m_const = self.overall_rot/self.dlo_sim.dlo_math.bigL_bar
        # print(m_const)
        # input(self.overall_rot)
        s_ss = (
            s * (self.beta_bar*m_const/(2*self.alpha_bar))
            * np.sqrt((1-np.cos(max_devi2))/(1+np.cos(max_devi2)))
        )
        # s_ss = s_ss - np.mean(s_ss)
        fphi = (np.cos(devi_set)-np.cos(max_devi))/(1-np.cos(max_devi))
        return s_ss, fphi, max_devi

    def start_lhbtest(self, new_start):
        if self.do_render:
            self.set_viewer_details(
                5.7628,
                -40.478,
                -12.434,
                np.array([-1.17009866, -1.37107526,  0.02327594])
            )
        """
        For LHB:
            - low mass: 0.058 --> lower mass, more centered
            - low damp: 0.3
            - hold step: 3 seconds --> longer hold more centered
            - vel reset: every sim step --> more frequent, more stable
                - (can try to implement reset_vel at fixed time intervals)
        """
        # init vars
        self.freq_velreset = 0.2
        # Test for 1 rope type - length, alpha, beta
        # localized helical buckling test
        lhb_picklename = 'lhbtest{}.pickle'.format(self.r_pieces)
        lhb_picklename = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/lhb/" + self.picklefolder + "/" + lhb_picklename
        )
        if new_start:
            self.lhb_init()
            self.init_pickle = self.get_state()
            with open(lhb_picklename, 'wb') as f:
                pickle.dump(self.init_pickle,f)
            print('Pickle saved!')
            # input('Pickle saved!')
            # s_ss_center, fphi_center =  self.lhb_testing()
        else:
            with open(lhb_picklename, 'rb') as f:
                self.init_pickle = pickle.load(f)

            # set overall rot
            self.set_state(self.init_pickle)
            self.init_pickle[0][9] = self.overall_rot
            self.p_thetan = self.overall_rot % (2. * np.pi)
            if self.p_thetan > np.pi:
                self.p_thetan -= 2 * np.pi
            self.init_pickle[0][10] = self.p_thetan

        s_ss_center, fphi_center, max_devi =  self.lhb_testing()
            # s_ss_center = None
            # s_ss_mincenter = 10.
            # hold_time = 50.
            # init_time = self.cur_time
            # while (self.cur_time-init_time) < hold_time:
            #     self.step()
            #     # shouldn't this line onwards be out of the loop?
            #     # no, it is in the loop to ensure the final graphs are centralized
            #     s_ss, fphi = self.lhb_var_compute()
            #     maxdevi_id = np.where(fphi == fphi.min())[0][0]
            #     s_ss_maxdevi = abs(s_ss[maxdevi_id])
            #     if s_ss_maxdevi < s_ss_mincenter or s_ss_center is None:
            #         s_ss_mincenter = s_ss_maxdevi
            #         s_ss_center = s_ss.copy()
            #         fphi_center = fphi.copy()

        print(f"fphi = {fphi_center}")
        print(f"s_ss = {s_ss_center}")
        print(f"max_devi = {max_devi}")
        pickledata2 = [
            max_devi,
            np.array(
                self.data.site_xpos[self.joint_site_idx[:]]
            ).copy()
        ]
        pickledata_path2 = 'lhb{}_miscdata.pickle'.format(self.r_pieces)
        pickledata_path2 = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/lhb/" + self.picklefolder + "/" + pickledata_path2
        )
        with open(pickledata_path2, 'wb') as f:
            pickle.dump(pickledata2,f)

        pickledata = [fphi_center, s_ss_center]
        pickledata_path = 'lhb{}.pickle'.format(self.r_pieces)
        pickledata_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/lhb/" + self.picklefolder + "/" + pickledata_path
        )
        with open(pickledata_path, 'wb') as f:
            pickle.dump(pickledata,f)
        print('pickled data')

        print(f"r_pieces = {self.r_pieces}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Circle ||~~~~~~~~~~~~~~~~~~~~~~~~~~

    def circle_init(self):
        # assuming that rope axis is parallel to x axis
        # self.hold_pos(10.)
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
        self.ropeend_pos(np.array([step_remain - 1e-3, 0., 0.]))
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
        if self.do_render:
            self.set_viewer_details(
                4.5076,
                45.922,
                -19.810,
                np.array([-3.46078809, -0.26378238, 0.63761223])
            )
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
            self.init_pickle = self.get_state()
            with open(circtest_picklename, 'wb') as f:
                pickle.dump(self.init_pickle,f)
            input(f'Circle Pickle saved!')
        else:
            with open(circtest_picklename, 'rb') as f:
                self.init_pickle = pickle.load(f)

            # set overall rot
            self.init_pickle[0][9] = self.overall_rot
            self.p_thetan = self.overall_rot % (2. * np.pi)
            if self.p_thetan > np.pi:
                self.p_thetan -= 2 * np.pi
            self.init_pickle[0][10] = self.p_thetan

            self.set_state(self.init_pickle)

            # # get and apply force normal to the circle
            norm_force = self.get_rope_normal()
            # self.apply_force_t(t=0.3,force_dir=np.array([0., 0., 1.]))
            self.apply_force_t(t=0.8,force_dir=norm_force)

            # self.hold_pos(100.)
            # create a for loop that checks PCA error at each iter
            max_e = 0.
            e_outofplane = 0.
            self.circle_oop = False
            print('0')
            # print('0')
            for i in range(100):
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
                print(f"Holding for {i+1} / {100}..")
                self.hold_pos(0.2, rend=True)
                if self.instability_check():
                    self.stable_bool = False
                e_outofplane = self.check_e_PCA_circle()
                if e_outofplane > max_e:
                    max_e = e_outofplane
                if max_e > e_tol:
                    print(f"e_tol = {e_tol}")
                    print(f"e_outofplane = {e_outofplane}")
                    print(f"max_e = {max_e}")

                    self.circle_oop = True
                    print(f'b_a = {self.beta_bar/self.alpha_bar} ==================================')
                    print(f'out of plane theta_crit = {self.overall_rot} ==================================')
                    return e_outofplane
                # print(f'e_outofplane = {e_outofplane}')
            print(f"e_tol = {e_tol}")
            print(f"e_outofplane = {e_outofplane}")
            print(f"max_e = {max_e}")
            if max_e > e_tol:
            # if e_outofplane > e_tol or max_e > 5:
                self.circle_oop = True
                print(f'b_a = {self.beta_bar/self.alpha_bar} ==================================')
                print(f'out of plane theta_crit = {self.overall_rot} ==================================')
                # print(f'e_outofplane = {e_outofplane}')
            if not self.stable_bool:
                self.circle_oop = True
                input('Unstable sim: press "Enter" to continue..')
            return e_outofplane

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Speed Test ||~~~~~~~~~~~~~~~~~~~~~~~~~~

    def run_speedtest1(self):
        start_t = time()
        start_step = self.env_steps
        pos_move = np.array([1.5e-3, -1.5e-3, -1.5e-3])
        print('0')
        for i in range(25):
            sys.stdout.write(f"\033[{1}F")
            print(f"Testing for {i+1} / {50} steps..")
            self.ropeend_pos_all(pos_move=pos_move.copy())

        for i in range(360):
            self.ropeend_rot(rot_axis=0)

        pos_move = np.array([1.5e-3, 1.5e-3, 1.5e-3])
        for i in range(25):
            sys.stdout.write(f"\033[{1}F")
            print(f"Testing for {i+1+25} / {50} steps..")
            self.ropeend_pos_all(pos_move=pos_move.copy())
        total_steps = self.env_steps - start_step
        total_time = time() - start_t
        real_speed = total_time / total_steps
        real_v_sim_speed = real_speed / self.dt

        print(f"Time taken = {total_time} s / {total_steps} steps")
        print(f"    = {real_speed} s / steps")
        print(f"    = {total_time} s / {total_steps*self.dt} s")
        print(f"    = {real_v_sim_speed} s / s (real time / sim time)")

        return real_v_sim_speed

    def run_speedtest2(self):
        # Load lhb_test and carry out speed that with that simulation
        self.freq_velreset = 0.2
        lhb_picklename = 'lhbtest{}.pickle'.format(self.r_pieces)
        lhb_picklename = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/lhb/" + self.picklefolder + "/" + lhb_picklename
        )
        with open(lhb_picklename, 'rb') as f:
            self.init_pickle = pickle.load(f)
        self.set_state(self.init_pickle)
        self.init_pickle[0][9] = self.overall_rot
        self.p_thetan = self.overall_rot % (2. * np.pi)
        if self.p_thetan > np.pi:
            self.p_thetan -= 2 * np.pi
        self.init_pickle[0][10] = self.p_thetan

        # run for 10,000 steps
        start_t = time()
        start_step = self.env_steps
        st_steps = 10000
        print('0')
        for i in range(st_steps):
            if (i+1) % 1000 == 0:
                sys.stdout.write(f"\033[{1}F")
                print(f"Testing for {i+1} / {st_steps} steps..")
            self.step()

        total_steps = self.env_steps - start_step
        total_time = time() - start_t
        real_speed = total_time / total_steps
        real_v_sim_speed = real_speed / self.dt

        print(f"Time taken = {total_time} s / {total_steps} steps")
        print(f"    = {real_speed} s / steps")
        print(f"    = {total_time} s / {total_steps*self.dt} s")
        print(f"    = {real_v_sim_speed} s / s (real time / sim time)")

        return real_v_sim_speed

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

    def ropeend_rot(self, rot_a=np.pi/180, rot_axis=0):
        rot_arr = np.zeros(3)
        rot_arr[rot_axis] = rot_a
        rot_quat = T.axisangle2quat(rot_arr)
        new_quat = T.quat_multiply(rot_quat, self.model.body_quat[self.ropeend_body_id])
        self.model.body_quat[self.ropeend_body_id] = new_quat
        self.hold_pos(0.02)

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