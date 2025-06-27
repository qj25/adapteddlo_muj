import os
import numpy as np

import mujoco
import mujoco_viewer

import gymnasium as gym
from gymnasium import utils
import pickle
import sys

import adapteddlo_muj.utils.transform_utils as T
# from adapteddlo_muj.controllers.pose_controller_ur5 import PoseController
from adapteddlo_muj.utils.mjc_utils import MjSimWrapper
from adapteddlo_muj.utils.xml_utils import XMLWrapper
import adapteddlo_muj.utils.mjc2_utils as mjc2
from adapteddlo_muj.utils.ik_utils import ik_denso
from adapteddlo_muj.assets.genrope.gdv_N import GenKin_N
from adapteddlo_muj.assets.genrope.gdv_O_xfrc import GenKin_O_xfrc
from adapteddlo_muj.assets.genrope.gdv_O import GenKin_O

from adapteddlo_muj.controllers.ropekin_controller_adapt import DLORopeAdapt
from adapteddlo_muj.controllers.ropekin_controller_xfrc import DLORopeXfrc
# from adapteddlo_muj.utils.ik_ur5.Ikfast_ur5 import Uik
# from adapteddlo_muj.controllers.joint_controller_ur5 import joint_sum

class ValidRnR3Env(gym.Env, utils.EzPickle):
    def __init__(
        self,
        do_render=True,
        alpha_bar=1.345/50,   # 1.345/50,
        beta_bar=0.789/50,    # 0.789/50,
        r_pieces=30,    # max. 33
        r_mass=0.58,
        r_len = 2*np.pi,
        r_thickness=0.03,
        overall_rot=None,
        manual_rot=False,
        plugin_name="cable",
        rgba_vals=None,
    ):
        utils.EzPickle.__init__(self)

        self.do_render = do_render
        # rope init
        self.r_len = r_len
        self.r_pieces = r_pieces
        self.alpha_bar = alpha_bar
        self.beta_bar = beta_bar
        self.r_mass = r_mass
        self.r_thickness = r_thickness
        self.init_pos=np.array([0.385,0.0,0.3])
        self.init_quat=np.array([0.707,0.,0.707,0.])
        self.rope_initpose = np.array([
            0.0, 0.0, 0.0,
            1., 0., 0., 0.
        ])
        self.rope_initpose[:3] = self.init_pos.copy()
        self.rope_initpose[0] += self.r_len

        # add 2 pieces for front and back
        self.rope_initpose[0] += (self.r_len)/self.r_pieces
        self.r_len *= (self.r_pieces+2)/self.r_pieces
        self.r_pieces += 2

        self.plugin_name = plugin_name
        self.rgba_vals = rgba_vals
        self.velreset = False
        print(f"initial overall_rot = {overall_rot}")
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
            self.set_viewer_details(
                dist=0.5,
                azi=-90,
                elev=0,
                lookat=np.array([0.585, 0.,  0.3])
            )
        else:
            self.viewer = None

        # enable joint visualization option:
        self.scene_option = mujoco.MjvOption()
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        # misc class data
        self.dt = self.model.opt.timestep
        self.max_action = 0.002

        # for i in range(26):
            # print(f"id={i}:  type={mujoco.mju_type2Str(i)}")
            
        # for i in range(6):
        #     print("joint names:")
        #     print(mjc2.obj_id2name(self.model,"joint",i))
        # for i in range(12):
        #     print("actuator names:")
        #     print(mjc2.obj_id2name(self.model,"actuator",i))
        # ref
        self.eef_site_name = "eef_site"
        self.eef_site_idx = mjc2.obj_name2id(
            self.model,
            "site",
            self.eef_site_name
        )
        actuator_names = []
        for i in range(6):
            actuator_names.append(mjc2.obj_id2name(self.model,"actuator",i))
        self._init_ids(actuator_names)
        grav_comp_act = []
        for i in range(6,12):
            grav_comp_act.append(mjc2.obj_id2name(self.model,"actuator",i))
        self.gcact_id = [
            mjc2.obj_name2id(self.model, "actuator", n)
            for n in grav_comp_act
        ]

        # other variables
        self.max_env_steps = 10000000
        self.env_steps = 0
        self.cur_time = 0
        self.dt = self.model.opt.timestep

        # init obs
        self.observations = dict(
            qpos=np.zeros(6),
            eef_pos=np.zeros(3),
            eef_quat=np.zeros(4),
            eef_vel=np.zeros(6),
            ft_world=np.zeros(6),
        )

        # # Get weld pose
        # init_pos=np.array([0.35,0.0,0.3])
        # init_quat=np.array([0.,1.,0.,0.])
        # goal = np.concatenate((init_pos,init_quat))
        # # find self.init_qpos through move_to_pose function
        # self.init_qpos = np.array([
        #     1.23530872e-08,  5.05769873e-01,
        #     1.92988983e+00, -1.04436514e-07,
        #     7.05933183e-01,  3.14159125e+00
        # ])
        # self.ik_arm = ik_denso(init_qpos=self.init_qpos)
        # self.init_qpos = self.ik_arm.calc_ik(
        #     goal, self.init_qpos.copy()
        # ).copy()
        # rob_pos = np.array(
        #     self.data.site_xpos[self.eef_site_idx]
        # )
        # print(self.data.site_xpos)
        # eefmat = np.array(
        #     self.data.site_xmat[self.eef_site_idx].reshape((3, 3))
        # )
        # rob_quat = T.mat2quat(eefmat)
        # self.relpose = np.zeros(7)
        # self.relpose[:3] = init_pos - rob_pos
        # self.relpose[3:] = T.axisangle2quat(T.quat_error(init_quat,rob_quat))
        # print(self.relpose)
        # input(self.model.eq_data)

        ## Controller stuff
        # self.qpos_goal = np.zeros(6)
        # self.base_qpos = np.array([
            # -3.12648459e-06, -1.32630504e-01,
            # -4.27119783e-03, -2.20483337e-04,
            # 2.80776196e-03,  2.92219323e-04
        # ])
        # self.init_qpos = np.array([
            # 1.23530872e-08,  5.05769873e-01,
            # 1.92988983e+00, -1.04436514e-07,
            # 7.05933183e-01,  3.14159125e+00
        # ])

        # self.qpos_goal = np.array([
            # -4.76658521e-07, -9.66914989e-02,
            # -6.05241353e-05, -2.72934914e-06,
            # 2.68649278e-02,  3.67922750e-05
        # ])
        # self.qpos_goal = np.array([
        #     7.40857786e-07,  6.38881793e-01,
        #     2.27269786e+00, -3.14160763e+00,
        #     1.34089981e+00,  3.14161037e+00
        # ])
        self._jd = np.array([
            2.60179266e-08,  5.91557993e-01,
            2.27995352e+00, -3.14159265e+00,
            1.35875935e+00,  3.14159265e+00
        ])
        self.qpos_tol = 1e-4
        
        self.init_qpos = np.array([
            7.40857786e-07,  6.38881793e-01,
            2.27269786e+00, -3.14160763e+00,
            1.34089981e+00,  3.14161037e+00
        ])
        self.init_qvel = np.zeros(6)
        self.data.qpos[self.joint_qposids[:6]] = np.array(self.init_qpos)
        self.data.qvel[self.joint_dofids[:6]] = np.array(self.init_qvel)
        self.current_joint_positions = np.array([self.data.qpos[ji] for ji in self.joint_ids])        

        # weld fix
        self.model.eq_data[0][3:6] = np.zeros(3)
        # self.model.eq_data[0][3] = -self.r_len/self.r_pieces
        self.model.eq_data[0][3] = 0.0
        self.model.eq_data[0][6:10] = T.quat_multiply(
            self.model.eq_data[0][6:10],
            np.array([0.707, 0., 0.707, 0.])
        )
        # self.data.eq_active = False
        # self.model.eq_data[0][6:10] = 

        # gravity compensation
        self.model.opt.gravity[-1] = -9.81
        # self.model.opt.gravity[-1] = 0.0
        self.grav_comp()

        self.sim.forward()

        self._get_observations()

        # # get weld relation
        # ropeend_id = mjc2.obj_name2id(self.model, "body", "B_last2")
        # eef_pos2 = self.data.xpos[ropeend_id]
        # eef_quat2 = np.array(
        #     self.data.xquat[ropeend_id]
        # )
        # eef_id = mjc2.obj_name2id(self.model, "body", "eef_body")
        # eef_pos1 = self.data.xpos[eef_id]
        # eef_quat1 = np.array(
        #     self.data.xquat[eef_id]
        # )
        # print(eef_pos2-eef_pos1)
        # print(eef_pos1)
        # print(eef_pos2)
        # print(T.axisangle2quat(T.quat_error(eef_quat1,eef_quat2)))
        # input() 

        self.ik_arm = ik_denso(init_qpos=self.init_qpos)

        # # pickle stuff
        # self._init_pickletool()
        # self._save_initpickle()
        # self._load_initpickle()

    def _init_ids(self, actuator_names):
        self.actuator_ids = [
            mjc2.obj_name2id(self.model, "actuator", n)
            for n in actuator_names
        ]
        self.actuator_names = actuator_names
        # print(self.actuator_ids)
        # print(actuator_names)
        # input()
        self.joint_ids = self.model.actuator_trnid[self.actuator_ids, 0]
        self.joint_qposids = self.model.jnt_qposadr[self.joint_ids]
        self.joint_dofids = self.model.jnt_dofadr[self.joint_ids]

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
        self.eef_site_idx2 = mjc2.obj_name2id(
            self.model,
            "site",
            'S_last'
        )
        self.ropeend_body_id = mjc2.obj_name2id(self.model,"body","stiffrope")
        self.ropeend_getstate_bodyid = mjc2.obj_name2id(self.model,"body","stiffrope")

        self.joint_ids2 = []
        self.joint_qveladdr = []
        for i in range(1, self.r_pieces):    # from freejoint2 to freejoint(n_links)
            fj_str = f'J_{i}'
            if i == (self.r_pieces-1):
                fj_str = 'J_last'
            self.joint_ids2.append(mjc2.obj_name2id(
                self.model,"joint",fj_str
            ))
            # if self.model.jnt_type(
                # self.dlo_joint_ids2[-1]
            # ) == self.model.mjtJoint.mjJNT_FREE:
                # print('it is free joint')
                # print(fj_str)
                # print(self.model.jnt_dofadr[self.dlo_joint_ids2[-1]])
            self.joint_qveladdr.append(
                self.model.jnt_dofadr[self.joint_ids2[-1]]
            )
        self.joint_qveladdr = np.array(self.joint_qveladdr)
        self.joint_qveladdr_full = [
            n for n in range(
                self.joint_qveladdr[0],
                self.joint_qveladdr[-1] + 3
            )
        ]

    def _get_xmlstr(self):
        if self.plugin_name == 'cable':
            ropexml = "nativerope1dkin.xml"
            overallxml = "overall_native.xml"
        elif self.plugin_name in ['wire','wire_qst']:
            ropexml = "dlorope1dkin.xml"
            overallxml = "overall.xml"

        # load model
        # update rope model
        world_base_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets/world_rnrvalid.xml"
        )
        robot_path = os.path.join(
            os.path.dirname(world_base_path),
            "densovs060/densovs060_wireclamp.xml"
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
            ropexml
        )
        j_damp = 0.01
        self.bothweld = False
        GenKin_N(
            r_len=self.r_len,
            r_thickness=self.r_thickness,
            r_pieces=self.r_pieces,
            r_mass=self.r_mass,
            stiff_vals=self.stiff_vals,
            j_damp=j_damp,
            init_pos=self.rope_initpose[:3],
            init_quat=self.rope_initpose[3:],
            coll_on=True,
            rope_type="capsule",
            vis_subcyl=False,
            obj_path=rope_path,
            plugin_name=self.plugin_name,
            rgba_vals=self.rgba_vals
        )
        if self.plugin_name != 'cable':
            ropexml = "dlorope1dkin.xml"
            overallxml = "overall.xml"
        
        self.xml = XMLWrapper(world_base_path)
        anchorbox = XMLWrapper(box_path)
        robotarm = XMLWrapper(robot_path)
        dlorope = XMLWrapper(rope_path)
        weldweight = XMLWrapper(weldweight_path)
        # pandagripper = XMLWrapper(gripper_path)
        # miscobj = XMLWrapper(miscobj_path)
        dlorope.merge(
            weldweight,
            element_name="body",
            attrib_name="name",
            attrib_value="B_last",
            action="append",
        )

        self.xml.merge_multiple(
            robotarm, ["worldbody", "asset", "actuator", "extension"]
            # robotarm, ["worldbody", "asset", "actuator"]
            # robotarm, ["worldbody", "asset", "actuator", "default", "keyframe", "sensor"]
        )
        self.xml.merge_multiple(
            dlorope, ["worldbody", "extension"]
        )
        self.xml.merge_multiple(
            anchorbox, ["equality", "contact"]
        )

        asset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets/" + overallxml
        )

        xml_string = self.xml.get_xml_string()

        model = mujoco.MjModel.from_xml_string(xml_string)
        mujoco.mj_saveLastXML(asset_path,model)
        
        return xml_string, robotarm
    
    def step(self, action=np.zeros(6)):
        # self.viewer._paused = True
        self.grav_comp()
        # print(self.data.ncon)
        # print(f"contacts = {self._print_contacts()}")
        self.data.ctrl[self.joint_ids] = action
        # print(self.overall_rot)
        # print(self.observations['eef_pos'])
        # if self.env_steps==1000:
        #     self.data.eq_active = True

        self.sim.step()
        self.sim.forward()
        self.cur_time += self.dt

        # if self.velreset:
        #     if self.env_steps%100==0:
        #         self.data.qvel = np.zeros_like(self.data.qvel)

        if self.env_steps%1==0:
            if self.do_render:
                self.viewer.render()
                # self.viewer._paused = True

        self.env_steps += 1
    
        done = self.env_steps > self.max_env_steps
        
        return self._get_observations(), 0, done, False, 0

    def hold_step(self,targ_qpos):
        qpos_diff = 999
        max_action = 0.0707
        control_freq=40
        ctrl_ts = 1 / control_freq
        dyn_ts = self.model.opt.timestep
        interpolate_steps = np.ceil(
            ctrl_ts / dyn_ts
        )
        qpos_diff = self.joint_sum(targ_qpos, - self.observations['qpos'])
        move_dir = qpos_diff
        j0 = self._jd
        action = self.scale_action(
            move_dir, out_max=max_action
        )
        steps = 0
        # print(f"qpos_diff = {qpos_diff}")
        # print(f"qpos_targ = {targ_qpos}")
        # print(f"qpos_curr = {self.observations['qpos']}")
        while steps < ctrl_ts/dyn_ts:
            steps += 1
            # jd = self.joint_sum(
                # j0, (action*steps/interpolate_steps)
            # )
            jd = (
                j0 + (action*steps/interpolate_steps)
            )

            self.step(action=jd)
            # print(f"jd = {jd}")
            # print(f"init_qpos = {self.init_qpos}")
            # input()
            # qpos_diff = self.move_to_qpos(
                # jd,
                # qpos_stepsize=0.1,
            # )
            # self.viewer._paused = True
        self._jd = jd
    
    def _get_observations(self):
        # get eef_vel
        ee_vel = mjc2.obj_getvel(
            self.model,
            self.data,
            "site",
            self.eef_site_idx
        )

        self.observations['eef_vel'] = ee_vel.copy()
        
        # get eef_pos and eef_quat
        eef_pos = np.array(
            self.data.site_xpos[self.eef_site_idx]
        )
        # print(mjc2.obj_id2name(self.model, "site", self.eef_site_idx))
        # input("here")
        self.observations['eef_pos'] = eef_pos
        eefmat = np.array(
            self.data.site_xmat[self.eef_site_idx].reshape((3, 3))
        )
        # print('here')
        # print(self.observations['eef_pos'])
        eef_quat = T.mat2quat(eefmat)
        # if np.linalg.norm(eef_quat+self.prev_quat) < 1e-3:
        #     self.quat_switch *= -1
        self.observations['eef_quat'] = eef_quat
        self.prev_quat = eef_quat
        self.eef_mat = eefmat

        self.observations["qpos"] = self.data.qpos[self.joint_qposids[:6]].copy()
        self.observations["qvel"] = self.data.qvel[self.joint_dofids[:6]].copy()

        self.observations['rope_pose'] = np.concatenate((
            self.data.xpos[
                self.vec_bodyid[:self.r_pieces]
            ].copy(),
            [self.data.site_xpos[
                self.eef_site_idx2
            ].copy()]
        ))
        self.observations['rope_pose'] = self.observations['rope_pose'][1:-1]

        # # Check native weld working fine
        # self.endrope_site_idx = mjc2.obj_name2id(
            # self.model,
            # "site",
            # "S_last"
        # )
        # endropemat = np.array(
            # self.data.site_xmat[self.endrope_site_idx].reshape((3, 3))
        # )
        # print('endropemat:')
        # print(endropemat)

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
        
        self.data.qpos[self.joint_qposids[:6]] = np.array(self.init_qpos)
        self.data.qvel[self.joint_dofids[:6]] = np.array(self.init_qvel)

        self.sim.forward()

        # # reset controller
        # self.controller.reset()

        # reset obs
        self.observations = dict(
            qpos=np.zeros(6),
            eef_pos=np.zeros(3),
            eef_quat=np.zeros(4),
            eef_vel=np.zeros(6),
            ft_world=np.zeros(6),
        )

        # reset time
        self.cur_time = 0   #clock time of episode
        self.env_steps = 0

        # pickle
        # self._load_initpickle()

        return None
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| External Funcs ||~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # def move_to_qpos(self, targ_pos):
    #     # move to joint pose
    #     # intialize variables
    #     done = False
    #     step_counter = 0
        
    #     while not done:
    #         step_counter += 1
    #         j_pos = self.observations['qpos']
    #         move_dir = targ_pos[:6] - j_pos
    #         move_dir = move_dir * 10
    #         move_step = move_dir.copy()
    #         self.step(move_step)
    #         done = T.check_proximity(
    #             targ_pos, self.observations['qpos'], d_tol=5e-3
    #         )
    #     print(step_counter)

    def hold_pos(self, hold_time=2.):
        init_time = self.cur_time
        # self.print_collisions()
        fixed_qpos = self.observations['qpos'].copy()
        while (self.cur_time-init_time) < hold_time:
            self.hold_step(fixed_qpos)
        return fixed_qpos

    def move_then_hold(self):
        for i in range(10000):
            joint_add = np.zeros(6)
            joint_add[0] = 0.00002*self.env_steps
            joint_add[2] = -0.00003*self.env_steps
            j_desired = np.array(self.current_joint_positions+joint_add)
            self.step(action=j_desired)
        
        for i in range(10000):
            self.step(action=j_desired)
            
        final_qpos = np.array([self.data.qpos[ji] for ji in self.joint_ids])
        print(final_qpos)
        print(j_desired)
        print(j_desired-final_qpos)
        eef_pos = np.array(
            self.data.site_xpos[self.eef_site_idx]
        )
        eefmat = np.array(
            self.data.site_xmat[self.eef_site_idx].reshape((3, 3))
        )
        # print('here')
        # print(self.observations['eef_pos'])
        eef_quat = T.mat2quat(eefmat)
        print(eef_pos)
        print(T.quat2axisangle(eef_quat))

        self.viewer._paused = True
        self.step(action=j_desired)


    def grav_comp(self):
        self.base_torq = self.sim.data.qfrc_bias[self.joint_ids]
        self.sim.data.ctrl[self.gcact_id] = self.base_torq       

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| IK ||~~~~~~~~~~~~~~~~~~~~~~~~~~

    def joint_sum(self, a, b):
        return ((a+b)+np.pi)%(2.*np.pi)-np.pi

    def scale_action(self, action, out_max = 0.015):
        """
        Clips @action to be within self.input_min and self.input_max, and then re-scale the values to be within
        the range self.output_min and self.output_max

        Args:
            action (Iterable): Actions to scale

        Returns:
            np.array: Re-scaled action
        """
        # print(f"out_max = {out_max}")
        len_input = len(action)
        input_max = np.array([1] * len_input)
        input_min = np.array([-1] * len_input)
        output_max = np.array([out_max] * len_input)
        output_min = np.array([-out_max] * len_input)

        action_scale = abs(output_max - output_min) / abs(
            input_max - input_min
        )
        action_output_transform = (output_max + output_min) / 2.0
        action_input_transform = (input_max + input_min) / 2.0
        action = np.clip(action, input_min, input_max)
        transformed_action = (
            action - action_input_transform
        ) * action_scale + action_output_transform

        return transformed_action

    def move_to_qpos_good(
        self,
        targ_qpos,
        # qpos_stepsize=0.005
    ):
        qpos_diff = 999
        control_freq=40
        ctrl_ts = 1 / control_freq
        dyn_ts = self.model.opt.timestep
        interpolate_steps = np.ceil(
            ctrl_ts / dyn_ts
        )
        # while True:
            # self.step(action=self.init_qpos)
        print('0')
        # print('0')
        # print('0')
        # print('0')
        # print('0')
        # print('0')
        # print('0')
        while np.linalg.norm(qpos_diff) > self.qpos_tol:
            qpos_diff = self.joint_sum(targ_qpos, - self.observations['qpos'])
            move_dir = qpos_diff
            j0 = self._jd
            action = self.scale_action(
                move_dir, out_max=self.max_action
            )
            steps = 0
            sys.stdout.write(f"\033[{1}F")
            print(f"error = {np.linalg.norm(qpos_diff)}")
            # print(f"qpos_diff = {qpos_diff[:3]}")
            # print(f"{qpos_diff[3:]}")
            # print(f"qpos_targ = {targ_qpos[:3]}")
            # print(f"{targ_qpos[3:]}")
            # print(f"qpos_curr = {self.observations['qpos'][:3]}")
            # print(f"{self.observations['qpos'][3:]}")
            while steps < ctrl_ts/dyn_ts:
                steps += 1
                # jd = self.joint_sum(
                    # j0, (action*steps/interpolate_steps)
                # )
                jd = (
                    j0 + (action*steps/interpolate_steps)
                )

                self.step(action=jd)
                # print(f"jd = {jd}")
                # print(f"init_qpos = {self.init_qpos}")
                # input()
                # qpos_diff = self.move_to_qpos(
                    # jd,
                    # qpos_stepsize=0.1,
                # )
                # self.viewer._paused = True
            self._jd = jd
            # print(qpos_diff)
            # print(self.observations['qpos'])
            # print(np.linalg.norm(qpos_diff))
            # print(f"jd = {self._jd}")
        # print('hi')
        # print(repr(self.base_qpos))

    def move_to_qpos(
        self,
        targ_qpos,
        qpos_stepsize=0.005,
    ):
        rob_qpos = self.observations['qpos'].copy()
        qpos_diff = targ_qpos-rob_qpos
        # print(qpos_diff)
        total_dqpos = np.linalg.norm(qpos_diff)
        total_steps = int(total_dqpos/qpos_stepsize+0.5)
        if total_steps > 0:
            qpos_step = qpos_diff / total_steps
            for i_step in range(total_steps+1):
                qpos_next = rob_qpos+qpos_step*i_step
                # print(self.observations['qpos'])
                # print(qpos_next)
                self.step(action=qpos_next)
        else:
            qpos_next = targ_qpos
        for i in range(1):
            # print(qpos_next)
            self.step(action=qpos_next)
        self.viewer._paused = True

        # # mujoco.mj_forward(self.model, self.data)
        # print(repr(targ_qpos))
        # print(repr(qpos_next))
        # input()
        # self.step(action=qpos_next)
        return targ_qpos-rob_qpos

    def move_to_pose(
        self,
        targ_pos=None,
        targ_quat=None
    ):
        rob_pos = self.observations['eef_pos'].copy()
        rob_quat = self.observations['eef_quat'].copy()
        if targ_pos is None:
            targ_pos = rob_pos
        else:
            targ_pos = targ_pos
        if targ_quat is None:
            targ_quat = rob_quat # np.array([0, 1., 0, 0])
        else:
            targ_quat = targ_quat
        goal = np.concatenate((targ_pos,targ_quat))
        targ_qpos = self.ik_arm.calc_ik(goal, self.observations['qpos']).copy()
        self.move_to_qpos_good(targ_qpos=targ_qpos)

    def rot_arm_endx(self, x_rad):
        n_halfs = int(x_rad/np.pi)
        rot_leftover = x_rad - n_halfs*np.pi
        for i in range(n_halfs):
            rob_qpos = self.observations['qpos'].copy()
            targ_qpos = rob_qpos.copy()
            targ_qpos[5] += np.pi
            self.move_to_qpos_good(targ_qpos=targ_qpos)
        rob_qpos = self.observations['qpos'].copy()
        targ_qpos = rob_qpos.copy()
        targ_qpos[5] += rot_leftover
        self.move_to_qpos_good(targ_qpos=targ_qpos)

    def rot_x_rads(self, x_rads):
        rot_step = np.pi/180
        if x_rads < 0:
            rot_step *= -1.0
        n_rotsteps = int(x_rads/rot_step)
        rad_leftover = (
            x_rads
            - (n_rotsteps * rot_step)
        )
        print('0')
        # fixed_qpos = self.observations['qpos'].copy()
        for i in range(n_rotsteps):
            sys.stdout.write(f"\033[{1}F")
            print(f"init rot stage (degs): {i+1}/{n_rotsteps}")
            self.ropeend_rot(rot_a=rot_step,rot_axis=0)
        self.ropeend_rot(rot_a=rad_leftover,rot_axis=2)

    def ropeend_rot(self, rot_a=np.pi/180, rot_axis=0):
        rot_arr = np.zeros(3)
        rot_arr[rot_axis] = rot_a
        rot_quat = T.axisangle2quat(rot_arr)
        new_quat = T.quat_multiply(rot_quat, self.model.body_quat[self.ropeend_body_id])
        self.model.body_quat[self.ropeend_body_id] = new_quat
        self.hold_pos(0.05)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| End IK ||~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Pickle Stuff ||~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def _init_pickletool(self):
        # self.rl2_picklepath = 'rl2_' + str(self.r_pieces) # 'rob3.pickle'
        # self.rl2_picklepath = self.rl2_picklepath + self.stiff_str + '.pickle'
        self.tm3_picklepath = 'tm3' # 'rob3.pickle'
        self.tm3_picklepath = self.tm3_picklepath + '.pickle'
        self.tm3_picklepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/" + self.tm3_picklepath
        )

    def _save_initpickle(self):
        ## initial movements
        start_qpos = np.array([-np.pi/2,-np.pi/2,np.pi/2,-np.pi/2,-np.pi/2,0])
        self.move_to_qpos(start_qpos)
        self.hold_pos(10.)
        # cur_pos = self.observations['eef_pos'].copy()
        # new_pos = cur_pos + np.array([0., 0., -0.5])
        # self.move_to_pos(targ_pos=new_pos)

        ## create pickle
        self.init_pickle = self.get_state()

        ## save pickle
        with open(self.tm3_picklepath, 'wb') as f:
            pickle.dump(self.init_pickle,f)
        input('Pickle saved!')

    def _load_initpickle(self):
        with open(self.tm3_picklepath, 'rb') as f:
            self.init_pickle = pickle.load(f)
        self.set_state(self.init_pickle)

    # def get_state(self):
    #     prev_ctrlstate = self.controller.get_ctrlstate()
    #     # (
    #     #     ropeend_pos,
    #     #     ropeend_quat,
    #     #     overall_rot,
    #     #     p_thetan
    #     # ) = self.dlo_sim.get_dlosim()
    #     state = np.empty(
    #         mujoco.mj_stateSize(
    #             self.model,
    #             mujoco.mjtState.mjSTATE_PHYSICS
    #         )
    #     )
    #     mujoco.mj_getState(
    #         self.model, self.data, state,
    #         spec=mujoco.mjtState.mjSTATE_PHYSICS
    #     )
    #     return [
    #         np.concatenate((
    #             [0], # [self.cur_time],
    #             [0], # [self.env_steps],
    #             # ropeend_pos,
    #             # ropeend_quat,
    #             # [overall_rot],
    #             # [p_thetan],
    #         )),
    #         prev_ctrlstate,
    #         state
    #     ]
    
    # def set_state(self, p_state):
    #     self.cur_time = 0 # p_state[0][0]
    #     self.env_steps = 0 # p_state[0][1]
    #     # self.dlo_sim.set_dlosim(
    #     #     p_state[0][2:5],
    #     #     p_state[0][5:9],
    #     #     p_state[0][9],
    #     #     p_state[0][10],
    #     # )
    #     mujoco.mj_setState(
    #         self.model, self.data, p_state[2],
    #         spec=mujoco.mjtState.mjSTATE_PHYSICS
    #     )
    #     self.controller.set_ctrlstate(p_state[1])
    #     self.sim.forward()
    #     # self.controller.reset()
    #     self.controller.update_state()
    #     self._get_observations()
    #     # self.dlo_sim._update_xvecs()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| End Pickle Stuff ||~~~~~~~~~~~~~~~~~~~~~~~~~~

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

    def _print_contacts(self):
        # Get the number of contacts
        n_contacts = self.data.ncon
        print(f"Number of contacts: {n_contacts}")
    
        # Iterate through each contact and print its details
        for i in range(n_contacts):
            contact = self.data.contact[i]
            
            # Print basic contact information
            print(f"Contact {i+1}:")
            print(f"  - Body 1: {mjc2.obj_id2name(self.model,'body',contact.geom1)} (index of the first body)")
            print(f"  - Body 2: {mjc2.obj_id2name(self.model,'body',contact.geom2)} (index of the second body)")
            
            # Contact position (in world coordinates)
            print(f"  - Contact point (in world coordinates): {contact.pos}")
            
            # # Normal force at the contact point
            # print(f"  - Normal force: {contact.force}")
            
            # # Contact normal vector
            # print(f"  - Normal vector: {contact.normal}")
            
            # # The relative velocity at the contact point (if needed)
            # print(f"  - Relative velocity: {contact.vel}")
            
            print("")

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