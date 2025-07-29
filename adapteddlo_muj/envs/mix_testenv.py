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

from adapteddlo_muj.assets.genrope.gdv_N_weld2 import GenKin_N_weld2
from adapteddlo_muj.controllers.pseudo_ropekin_controller_both_ro import DERRopeBase
from adapteddlo_muj.utils.data_utils import compute_PCA
from adapteddlo_muj.utils.filters import ButterLowPass
from adapteddlo_muj.utils.mjc_utils import get_sensor_id, get_sensor_force2, fix_ftsensor_weld


class TestCableEnv(gym.Env, utils.EzPickle):
    def __init__(
        self,
        do_render=False,
        alpha_bar=1.345/50,   # 1.345/50,
        beta_bar=0.789/50,    # 0.789/50,
        r_pieces=30,    # max. 33
        r_mass=0.58,
        r_len = 2*np.pi,
        r_thickness=0.03,
        j_damp=0.5,
        overall_rot=None,
        rope_initpose=None,
        new_start=True,
        limit_f=False,
        plugin_name='wire',
        stiff_type='native',
    ):
        utils.EzPickle.__init__(self)

        self.do_render = do_render

        stiff_choices = dict(
            native=0,
            lop=1,
            bal=2,
        )
        self.stiff_type = stiff_choices[stiff_type] 
        self.plugin_name = plugin_name

        # rope init
        self.r_len = r_len
        self.r_pieces = r_pieces
        self.alpha_bar = alpha_bar
        self.beta_bar = beta_bar
        self.r_mass = r_mass
        self.r_thickness = r_thickness
        self.rope_initpose = np.array([
            0., 0.0, 0.5,
            1., 0., 0., 0.
        ])
        if rope_initpose is not None:
            self.rope_initpose = rope_initpose.copy()
        self.rope_initpose[0] += self.r_len/2
        self.overall_rot = 0. # 27 * (2*np.pi) # 57 * (np.pi/180)
        if overall_rot is not None:
            self.overall_rot = overall_rot # / (2.*np.pi) * 360.
        # init stiffnesses for capsule
        J1 = np.pi * (self.r_thickness/2)**4/2.
        Ix = np.pi * (self.r_thickness/2)**4/4.
        self.stiff_vals = [
            self.beta_bar/J1,
            self.alpha_bar/Ix
        ]
        if self.stiff_type != 0: self.stiff_vals = [0.,0.]
        self.j_damp = j_damp
        # if self.stiff_type == 2:
        # self.alpha_bar /=2
        # self.beta_bar /=2

        mjc2.init_plugins()
        xml, arm_xml = self._get_xmlstr()

        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.sim = MjSimWrapper(self.model, self.data)
        
        if self.do_render:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.rend_rate = int(10)
            self.set_viewer_details(
                dist=11.5,
                azi=52.5,
                elev=-31.0,
                lookat=np.array([2.02583349844652, 0.283788769044593, -0.7689941924964794])
            )
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
        self.freq_velreset = 1000
        self.ropemass = self.get_bodymass_total()

        # filter
        fs = 1.0 / self.dt
        cutoff = 30
        self.lowpass_filter = ButterLowPass(cutoff, fs, order=5)

        # init obs
        self.observations = dict(
            rope_pose=np.zeros((self.r_pieces,3)),
            sensor0_pose=np.zeros(7),
            sensor1_pose=np.zeros(7),
            ft_world_0=np.zeros(6),
            ft_world_1=np.zeros(6),
        )

        # init gravity
        self.model.opt.gravity[-1] = 0.0

        # if self.do_render:
        #     self.viewer._paused = True
        self.sim.forward()

        self._init_ids()
        self._init_boxfollow()

        self._get_observations()

        self.f_limit = 1000.
        self.der_sim = DERRopeBase(
            model=self.model,
            data=self.data,
            n_link=self.r_pieces,
            alpha_bar=self.alpha_bar,
            beta_bar=self.beta_bar,
            overall_rot=self.overall_rot,
            f_limit=self.f_limit,
        )

        self.freq_velreset = 0.2

        # init curvature test quats
        bodystart_id = mjc2.obj_name2id(self.model,"body","B_first")
        bodyanchor_id = mjc2.obj_name2id(self.model,"body","eef_body2")
        self.curve_test_pos = [
            self.model.body_quat[bodystart_id][:].copy(),
            self.model.body_quat[bodyanchor_id][:].copy()
        ]

        # # pickle stuff
        self._init_pickletool()
        # self._save_initpickle()
        if not new_start:
            self._load_initpickle()

    def _init_ids(self):
        # # body_names of attached sensors A and B, respectively.
        self.eefname_list = ['', '']
        self.sensorsitename_list = ['sensor_site2', 'sensor_site1']
        for i in range(len(self.sensorsitename_list)):
            self.eefname_list[i] = mjc2.obj_id2name(
                self.model, "body",
                self.model.site_bodyid[
                    mjc2.obj_name2id(self.model, "site",
                        self.sensorsitename_list[i]
                    )
                ]
            )
        self.sensorids_list = [
            get_sensor_id(self.model, self.sensorsitename_list[0]),
            get_sensor_id(self.model, self.sensorsitename_list[1])
        ]
        # self.eefname_list = ['B_last', 'B_first']

        # box body ids
        self.box_bodyids = [
            mjc2.obj_name2id(self.model,"body",self.eefname_list[0]),
            mjc2.obj_name2id(self.model,"body",self.eefname_list[1])
        ]

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
        self.ropeend_body_id = mjc2.obj_name2id(self.model,"body","eef_body2")
        self.ropeend_getstate_bodyid = mjc2.obj_name2id(self.model,"body","stiffrope")

        self.joint_ids = []
        self.joint_qposaddr = []
        self.joint_qveladdr = []
        for i in range(1, self.r_pieces):    # from freejoint2 to freejoint(n_links)
            fj_str = f'J_{i}'
            if i == (self.r_pieces-1):
                fj_str = 'J_last'
            self.joint_ids.append(mjc2.obj_name2id(
                self.model,"joint",fj_str
            ))
            # if self.model.jnt_type(
                # self.der_joint_ids[-1]
            # ) == self.model.mjtJoint.mjJNT_FREE:
                # print('it is free joint')
                # print(fj_str)
                # print(self.model.jnt_dofadr[self.der_joint_ids[-1]])
            self.joint_qposaddr.append(
                self.model.jnt_qposadr[self.joint_ids[-1]]
            )
            self.joint_qveladdr.append(
                self.model.jnt_dofadr[self.joint_ids[-1]]
            )
        self.joint_qposaddr = np.array(self.joint_qposaddr)
        self.joint_qveladdr = np.array(self.joint_qveladdr)
        self.joint_qveladdr_full = [
            n for n in range(
                self.joint_qveladdr[0],
                self.joint_qveladdr[-1] + 3
            )
        ]
        # plugin instance id
        known_body_name = "B_first"
        self.plgn_instance = self.model.body_plugin[
            mjc2.obj_name2id(self.model, "body", known_body_name)
        ]
    
    def _change_boxweld(self, relpose):
        for i in range(self.model.neq):
            if (
                self.model.eq_obj1id[i] 
                == mjc2.obj_name2id(self.model,"body",'B_last')
            ):
                self.model.eq_obj1id[i] = mjc2.obj_name2id(
                    self.model,"body",'B_last2'
                )
                self.model.eq_data[i][3:10] = relpose

    def _init_boxfollow(self):
        mujoco.mj_fwdPosition(self.model, self.data)
        self.box_bodyid = mjc2.obj_name2id(self.model,"body","eef_body")
        self.end_bodyid = mjc2.obj_name2id(self.model,"body","B_last2")
        self.box_end_relpose = np.zeros(7)
        self.box_end_relpose[3:] = T.quat_multiply(
            T.quat_inverse(self.data.xquat[self.end_bodyid]),
            self.data.xquat[self.box_bodyid]
        )
        mujoco.mju_rotVecQuat(
            self.box_end_relpose[:3],
            (
                self.data.xpos[self.box_bodyid]
                - self.data.xpos[self.end_bodyid]
            ),
            T.quat_inverse(self.data.xquat[self.end_bodyid])
        )
        self._change_boxweld(self.box_end_relpose.copy())

    def _boxposquat_follow(self):
        mujoco.mj_fwdPosition(self.model, self.data)
        self.box_worldpose = np.zeros(7)
        self.box_worldpose[3:] = T.quat_multiply(
            self.box_end_relpose[3:],
            self.data.xquat[self.end_bodyid]
        )
        mujoco.mju_rotVecQuat(
            self.box_worldpose[:3],
            (
                self.box_end_relpose[:3]
            ),
            self.data.xquat[self.end_bodyid]
        )
        self.box_worldpose[:3] += self.data.xpos[self.end_bodyid]
        self.model.body_pos[self.box_bodyid] = self.box_worldpose[:3].copy()
        self.model.body_quat[self.box_bodyid] = self.box_worldpose[3:].copy()
        # print(self.data.xpos[self.box_bodyid])
        # print(self.data.xquat[self.box_bodyid])
        # mujoco.mj_fwdPosition(self.model, self.data)
        # self.viewer.render()
        # input()

    def _boxpos_follow(self):
        mujoco.mj_fwdPosition(self.model, self.data)
        self.box_worldpose = np.zeros(7)
        self.box_worldpose[3:] = T.quat_multiply(
            self.box_end_relpose[3:],
            self.data.xquat[self.end_bodyid]
        )
        mujoco.mju_rotVecQuat(
            self.box_worldpose[:3],
            (
                self.box_end_relpose[:3]
            ),
            self.data.xquat[self.end_bodyid]
        )
        self.box_worldpose[:3] += self.data.xpos[self.end_bodyid]
        self.model.body_pos[self.box_bodyid] = self.box_worldpose[:3].copy()
        # self.model.body_quat[self.box_bodyid] = self.box_worldpose[3:].copy()

    def _get_xmlstr(self):
        # load model
        # update rope model
        world_base_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets/world.xml"
        )
        box_path = os.path.join(
            os.path.dirname(world_base_path),
            "anchorbox.xml"
        )
        rope_path = os.path.join(
            os.path.dirname(world_base_path),
            "nativerope1dkin.xml"
        )
        weldweight_path = os.path.join(
            os.path.dirname(world_base_path),
            "weldweight.xml"
        )
        GenKin_N_weld2(
            r_len=self.r_len,
            r_thickness=self.r_thickness,
            r_pieces=self.r_pieces,
            r_mass=self.r_mass,
            # stiff_vals=[0.,0.],
            stiff_vals=self.stiff_vals,
            j_damp=self.j_damp,
            coll_on=False,
            init_pos=self.rope_initpose[:3],
            init_quat=self.rope_initpose[3:],
            rope_type="capsule",
            vis_subcyl=True,
            obj_path=rope_path,
            plugin_name=self.plugin_name,
            twist_displace=self.overall_rot
        )

        self.xml = XMLWrapper(world_base_path)
        derrope = XMLWrapper(rope_path)
        anchorbox = XMLWrapper(box_path)
        weldweight = XMLWrapper(weldweight_path)

        weld_bool = False
        if weld_bool:
            derrope.merge(
                weldweight,
                element_name="body",
                attrib_name="name",
                attrib_value="B_first",
                action="append",
            )
            derrope.merge(
                weldweight,
                element_name="body",
                attrib_name="name",
                attrib_value="B_last2",
                action="append",
            )

        self.xml.merge_multiple(
            anchorbox, ["worldbody", "equality", "contact", "sensor"]
        )
        self.xml.merge_multiple(
            derrope, ["worldbody", "extension"]
        )
        asset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets/overall_native.xml"
        )
        xml_string = self.xml.get_xml_string()
        # with open(asset_path, "w+") as f:
            # f.write(xml_string)
        model = mujoco.MjModel.from_xml_string(xml_string)
        mujoco.mj_saveLastXML(asset_path,model)
        # model = mujoco.MjModel.to_xml_string(model)
        # xml_string = 
        # with open(asset_path, "w+") as f:
        #     f.write(xml_string)

        return xml_string, None
    
    def step(self, action=np.zeros(6)):
        # print('===========================================================')
        # input(self.env_steps)
        # t1 = time()
        # t2 = time()
        # self.model.eq_data[0][3] = 0.0
        # print(f"eq_data = {self.model.eq_data[0]}")
        # print(f"body_pos = {self.data.xpos[8]}")
        # print(f"body_quat = {self.data.xquat[8]}")
        body_id = mjc2.obj_name2id(self.model, "body", "B_2")
        jnt_id = mjc2.obj_name2id(
            self.model,"joint","J_2"
        )
        dof_adr = self.model.jnt_dofadr[jnt_id]
        dof_count = 3
        dof_ids = slice(dof_adr,dof_adr + dof_count)
        self.der_sim.update_force()
        if self.stiff_type == 1:
            self.data.qfrc_passive += self.der_sim.qfrc_our.copy()
            input('xfrc not working for this controller.. use another env.')
        elif self.stiff_type == 2:
            # print('hi')
            # print(self.der_sim.qfrc_our2)
            self.data.qfrc_passive += self.der_sim.qfrc_our2.copy()
            # input()
        if self.env_steps%100==0:
            # b8_ft = self._get_abi_ftsensor("S_8")
            # print(f"B_8_ft = {b8_ft}")
            # b10_ft = self._get_abi_ftsensor("S_10")
            # print(f"B_10_ft = {b10_ft}")

            pass
            # print("External force applied:", self.data.xfrc_applied[body_id])
            # print("External force computed:", self.data.cfrc_ext[body_id])
            # print("Internal constraint/joint force:", self.data.cfrc_int[body_id])
            # print("efc_force:", self.data.efc_force[body_id])
            # print("efc_force:", self.data.efc_force)
            # print("qfrc_bias:", self.data.qfrc_bias[dof_ids])
            # print(self.data.qfrc_bias.reshape((6,3)))
            # print(self.data.qfrc_passive.reshape((6,3)))
            # print(self.data.qfrc_applied.reshape((6,3)))
            # print("qfrc_passive:", self.data.qfrc_passive[dof_ids])
            # print("qfrc_actuator:", self.data.qfrc_actuator[dof_ids])
            # pos_diff = (
            #     self.data.xpos[mjc2.obj_name2id(self.model,"body","eef_body2_sensor")]
            #     - self.data.xpos[mjc2.obj_name2id(self.model,"body","B_first")]
            # )
            # pos_diff2 = (
            #     self.data.xpos[mjc2.obj_name2id(self.model,"body","eef_body_sensor")]
            #     - self.data.xpos[mjc2.obj_name2id(self.model,"body","B_last2")]
            # )
            # print(f'pos_diff = {pos_diff}')
            # print(f'pos_diff2 = {pos_diff2}')

            # print(f"joint_torques = {self.data.qfrc_passive.reshape((self.r_pieces+1,3))}")
            # print(f"joint_torques_our = {self.der_sim.qfrc_our.reshape((self.r_pieces+1,3))}")
            # print(f"joint_torques_our2 = {self.der_sim.qfrc_our2.reshape((self.r_pieces+1,3))}")
            # print(f"pos_nodes = {self.observations['rope_pose']}")
            # cal_misc = (
            #     self.data.qfrc_passive.reshape((self.r_pieces+1,3))[2,1]
            #     -self.data.qfrc_passive.reshape((self.r_pieces+1,3))[3,1]
            # )
            # a1 = T.angle_btwn(
            #     self.observations['rope_pose'][0]-self.observations['rope_pose'][1],
            #     self.observations['rope_pose'][1]-self.observations['rope_pose'][2],
            # )
            # print(f"cal_misc = {cal_misc}")
            # print(f"actual_ratio = {cal_misc/self.observations['ft_world_0'][0]}")
            # print(f"expect_ratio = {np.cos(a1)}")

            # # for torque eval
            # cw_tau = -self.data.qfrc_passive.reshape((self.r_pieces+1,3))[2,1]
            # ccw_tau = self.observations['ft_world_1'][4]
            # d_norm = self.observations['rope_pose'][0][2] - self.observations['rope_pose'][1][2]
            # print(f"tau_diff = {(cw_tau + ccw_tau)}")
            # print(f"tau_forc = {(self.observations['ft_world_1'][0])*d_norm}")
            # input()
            # print(f"theoretical ratio = {np.cos()}")
            # print(self.observations['ft_world_eef_body'][1])
            # cal_misc2 = (
                # self.data.qfrc_passive.reshape((self.r_pieces+1,3))[3,1]
                # -self.data.qfrc_passive.reshape((self.r_pieces+1,3))[4,1]
            # )
            # print(f"cal_misc2 = {cal_misc2}")
            # print(f"ratio2 = {cal_misc2/self.observations['ft_world_eef_body'][0]}")

        # self.data.eq_active[0] = 0
        # print('eq_data:')
        # print(self.model.eq_data[0])
        # input()
        if self.env_steps%100==0:
            print(f"env_steps = {self.env_steps} |=========================")
            print(f"qfrc_bal = {self.data.qfrc_passive.reshape((self.r_pieces+1,3))}")
        # print(f"qfrc_applied = {self.data.qfrc_applied.reshape((self.r_pieces+1,3))}")
        # print(f"pos = {self.data.xpos[self.vec_bodyid]}")
        self.sim.step()
        self.sim.forward()
        # print(f"qfrc_applied = {self.data.qfrc_applied.reshape((self.r_pieces+1,3))}")
        if self.env_steps%100==0:
            print(f"qfrc_native = {self.data.qfrc_passive.reshape((self.r_pieces+1,3))}")
        # print(f"pos2 = {self.data.xpos[self.vec_bodyid]}")
        # input()
        # t3 = time()
        if self.do_render:
            if self.env_steps%self.rend_rate==0:
                self.viewer.render()
        # self.viewer.render()

        self.env_steps += 1
        self.cur_time += self.dt

        done = self.env_steps > self.max_env_steps

        if self.instability_check():
            print(f'unstable {self.env_steps}')
            input()
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
        # id_0 ft is of ropeend. id_1 is top parent body.
        self._get_ftworld(0)
        self._get_ftworld(1)
        # Get the stored torques
        # Get plugin state with bounds checking
        start = self.model.plugin_stateadr[self.plgn_instance]
        if self.plgn_instance == self.model.nplugin - 1:
            # Last plugin - use remaining state
            self.stored_torques = self.data.plugin_state[start:start+self.model.nv]
            E_total = self.data.plugin_state[-1]

        else:
            # Not last plugin - use next plugin's start as end
            self.stored_torques = self.data.plugin_state[start:start+self.model.nv]
            E_total = self.data.plugin_state[start+self.model.nv]

    def _get_abi_ftsensor(self, sensor_sitename):
        sensorsite_id = mjc2.obj_name2id(self.model, "site", sensor_sitename)
        sensor_pose = np.concatenate((
            self.data.site_xpos[sensorsite_id].copy(),
            T.mat2quat(self.data.site_xmat[sensorsite_id].copy())
        ))
        sensor_id = get_sensor_id(self.model, sensor_sitename)
        sensor_ft = get_sensor_force2(
            self.model,
            self.data,
            sensor_id,
            sensor_sitename,
            # self.eef_name,
            # sensor_pose[:3],
            # sensor_pose[3:],
        )
        return sensor_ft

    def _get_ftworld(self, eef_id):
        site_name = self.sensorsitename_list[eef_id]
        # eef_id either 0 or 1
        self.observations['sensor'+str(eef_id)+'_pose'][:3] = self.data.xpos[self.box_bodyids[eef_id]].copy()
        self.observations['sensor'+str(eef_id)+'_pose'][3:] = self.data.xquat[self.box_bodyids[eef_id]].copy()

        eef_ft = get_sensor_force2(
            self.model,
            self.data,
            self.sensorids_list[eef_id],
            site_name,
            # self.eef_name,
            # self.observations['sensor'+str(eef_id)+'_pose'][:3],
            # self.observations['sensor'+str(eef_id)+'_pose'][3:],
        )
        eef_ft_filtered = self.lowpass_filter(eef_ft.reshape((-1, 6)))[0, :]
        eef_rotmat = T.quat2mat(self.observations['sensor'+str(eef_id)+'_pose'][3:])
        f_world = eef_rotmat.dot(eef_ft_filtered[:3])
        t_world = eef_rotmat.dot(eef_ft_filtered[3:])
        f_world = eef_ft_filtered[:3]
        t_world = eef_ft_filtered[3:]
        f_world = eef_ft[:3]
        t_world = eef_ft[3:] / 2.0
        # t_world = fix_ftsensor_weld(
        #     f_world,
        #     eef_ft[3:],
        #     (
        #         self.observations['rope_pose'][eef_id-1]
        #         - self.observations['sensor'+str(eef_id)+'_pose'][:3]
        #     )
        # )
        self.observations['ft_world_'+str(eef_id)] = np.concatenate((f_world,t_world))
        # print(self.observations['ft_world_'+eef_name])
        # if self.env_steps%100==0:
        #     print("==========================================================")
        #     # print(f"force_{eef_id} = {self.data.sensordata.reshape((4,3))}")
        #     print(f"force_{eef_id} = {self.observations['ft_world_'+str(eef_id)].reshape((2,3))}")
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
        
        self.der_sim.reset_body()
        self.sim.forward()
        self.der_sim.reset_sim()

        # reset obs
        self.observations = dict(
            rope_pose=np.zeros((self.r_pieces,3)),
            sensor0_pose=np.zeros(7),
            sensor1_pose=np.zeros(7),
            ft_world_0=np.zeros(6),
            ft_world_1=np.zeros(6),
        )

        # reset time
        self.cur_time = 0   #clock time of episode
        self.env_steps = 0

        # pickle
        self._load_initpickle()

        return None
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| External Funcs ||~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def hold_pos(self, hold_time=2.):
        init_time = self.cur_time
        # self.print_collisions()
        while (self.cur_time-init_time) < hold_time:
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
            # if self.env_steps % 1000 == 0:
                # print(self.env_steps)
                # print(self.data.qfrc_passive)
                # print(self.stiff_type)
            if self.env_steps%self.freq_velreset==0:
                self.reset_vel()

    def instability_check(self):
        if np.linalg.norm(self.data.qvel[self.joint_qveladdr_full]) > 200.:
            return True
        return False

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Pickle Stuff ||~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def _init_pickletool(self):
        self.ocvt_picklepath = 'nativetest' # 'rob3.pickle'
        self.ocvt_picklepath = self.ocvt_picklepath + '.pickle'
        self.ocvt_picklepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/test/" + self.ocvt_picklepath
        )

    def _save_initpickle(self):
        ## initial movements
        self.hold_pos(10.)

        ## create pickle
        self.init_pickle = self.get_state()

        ## save pickle
        with open(self.ocvt_picklepath, 'wb') as f:
            pickle.dump(self.init_pickle,f)
        print('Pickle saved!')

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
        ) = self.der_sim.get_dersim()

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
                [overall_rot],
                [p_thetan],
            )),
            state,
            bodypose_data
        ]
    
    def set_state(self, p_state):
        self.cur_time = 0 # p_state[0][0]
        self.env_steps = 0 # p_state[0][1]
        self.model.body_pos[
            self.ropeend_getstate_bodyid,:
        ] = p_state[0][2:5]
        self.model.body_quat[
            self.ropeend_getstate_bodyid,:
        ] = p_state[0][5:9]
        self.der_sim.set_dersim(
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
        # self.der_sim.overall_rot = 27. * (2.*np.pi)
        # self.der_sim.der_math.resetTheta(
        #     self.der_sim.p_thetan, self.der_sim.overall_rot
        # )

        # self.sim.forward()
        self.step()
        self._get_observations()
        self.der_sim._update_xvecs()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| End Pickle Stuff ||~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Validation Stuff ||~~~~~~~~~~~~~~~~~~~~~~~~~~
    def print_nativeinfo(self):
        id_print = self.vec_bodyid[int(len(self.vec_bodyid)/2)]
        qadr = (
            self.model.jnt_qposadr[self.model.body_jntadr[id_print]]
            + self.model.body_dofnum[id_print]-3
        )
        quat_diff = np.zeros(4)
        omega_curve = np.zeros(3)
        # for i in range(len(self.model.body_quat)):
        #     print(mjc2.obj_id2name(self.model, "body", i))
        #     print(self.model.jnt_qposadr[self.model.body_jntadr[i]])
        #     print(self.model.body_jntadr[i])
        # input()
        # print(self.vec_bodyid)
        dofadr = self.model.jnt_dofadr[self.model.body_jntadr[id_print]]
        # print(id_print)
        # print(dofadr)
        # self.der_sim.update_force()
        qfrc_jnt = self.data.qfrc_passive[dofadr:dofadr+self.model.body_dofnum[id_print]]
        qfrc_jnt2 = self.der_sim.qfrc_our2[dofadr:dofadr+self.model.body_dofnum[id_print]]
        # print(self.der_sim.qfrc_our)
        # print(self.der_sim.qfrc_our2)
        # print(qfrc_jnt)
        mujoco.mju_mulQuat(quat_diff, self.model.body_quat[id_print], self.data.qpos[qadr:qadr+4])
        mujoco.mju_quat2Vel(omega_curve, quat_diff, 1.0)
        print(f"env_steps = {self.env_steps} |=========================")
        # print(self.model.body_quat)
        # print(f"qpos = {self.data.qpos[qadr:qadr+4]}")
        # print(f"body_quat = {self.model.body_quat[id_print]}")
        # print(f"quat_diff = {quat_diff}")
        # print(self.data.qfrc_passive)
        # print(self.der_sim.force_node)
        # print(self.der_sim.x)
        print(f"omega_curve = {omega_curve}")
        print(f"qfrc_passive = {qfrc_jnt}")
        print(f"qfrc_our2 = {qfrc_jnt2}")
        print(f"ratio1 = {qfrc_jnt[1]/omega_curve[1]}")
        print(f"ratio2 = {qfrc_jnt2[1]/omega_curve[1]}")
        print()
        # input()

    def set_curve(self, om):
        # make sure collision is turned off for both eef_boxes and rope
        if any(
            ((val > np.pi) or (val < -np.pi))
            for val in om
        ):
            print(f"Error: om value ({om}) too large (more than one round).")
            input()
        quat_jnt = T.axisangle2quat(om)
        for i in self.joint_qposaddr[:2]:
            self.data.qpos[i:i+4] = quat_jnt
        for i in self.joint_qposaddr[2:]:
            self.data.qpos[i:i+4] = quat_jnt
        #     # self.data.qpos[i:i+4] = T.quat_inverse(quat_jnt)

        self._boxpos_follow()
        mjc2.update_weld_relpose(self.model, self.data, "weld_end")
        self.step()
        self._get_observations()
        self.der_sim._update_xvecs()

    def test_force_curvature(self,l_shorten=0.3,n_steps=100):
        step_len = l_shorten / n_steps / 2
        pos_move = np.array([step_len, -step_len, -step_len])
        for i in range(1):
            self.ropeend_pos_all(pos_move=pos_move.copy())
            self.print_nativeinfo()
        pos_move = np.array([step_len, step_len, step_len])
        for i in range(1):
            self.ropeend_pos_all(pos_move=pos_move.copy())
        pos_move = np.array([step_len, 0., 0.])
        print('0')
        for i in range(n_steps-2):
            # sys.stdout.write(f"\033[{1}F")
            print(f"init stage: {i+1}/{n_steps-2}")
            # self.reset_vel()
            # print(self.sitename2pos("r_joint0_site"))
            # print(self.sitename2pos("r_joint180_site"))
            # print(self.sitename2pos("r_joint0_site")-self.sitename2pos("r_joint180_site"))
            self.ropeend_pos_all(pos_move=pos_move.copy())
            self.print_nativeinfo()

    def test_force_curvature2(self,om_val):
        self.model.opt.gravity[-1] = 0.0
        self.model.opt.gravity[-1] = -9.81
        print(f"om_val = {om_val}")
        quat_move = T.axisangle2quat(np.array([0.,-om_val/2,0.]))
        # bodystart_id = mjc2.obj_name2id(self.model,"body","stiffrope")
        jntfreeqposaddr = self.model.jnt_qposadr[mjc2.obj_name2id(
            self.model,"joint","freejoint_A"
        )]
        # self.data.qpos[jntfreeqposaddr+3:jntfreeqposaddr+7] = T.quat_multiply(
        self.model.body_quat[mjc2.obj_name2id(self.model,"body","B_first")] = T.quat_multiply(
            # self.data.qpos[jntfreeqposaddr+3:jntfreeqposaddr+7],
            self.curve_test_pos[0],
            # np.array([0.707,-0.707,0.,0.])
            quat_move
        )
        # dont move anchor, move weld condition instead
        # bodyanchor_id = mjc2.obj_name2id(self.model,"body","eef_body2")
        # self.model.body_quat[bodyanchor_id][:] = T.quat_multiply(
        #     self.curve_test_pos[1],
        #     quat_move
        # )
        # print(self.model.eq_data[1][3:7])  # Update quaternion
        mjc2.update_weld_relpose(self.model, self.data, "weld_start")
        self.set_curve(np.array([0.,om_val/(self.r_pieces-1), 0.]))
        # self.set_curve(np.array([0.,om_val, 0.]))
        # self.hold_pos(1.1)
        # self.data.eq_active[0] = 0
        # print("Released eq_weld.")
        # self.hold_pos(10.)
        self.print_nativeinfo()

    def test_restoringbend(self):
        # remove one connect eq
        self.data.eq_active[1] = 0
        # choose midpoint of rope
        mid_id = self.vec_bodyid[int(len(self.vec_bodyid)/2)]
        # set new weld obj to midpoint
        self.model.eq_obj1id[0] = mid_id
        # move weld box to midpoint
        body_id2 = mjc2.obj_name2id(self.model,"body","eef_body")
        self.model.eq_data[0][3:6] /= 2.
        self.model.body_pos[body_id2][:] = (
            self.data.xpos[mid_id][:]
            - self.model.eq_data[0][3:6]
        )
        body_id1 = mjc2.obj_name2id(self.model,"body","eef_body2")
        self.model.body_pos[body_id1][2] += 1

        self.hold_pos(10.)


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
        pos_move = np.array([step_len, step_len, step_len])
        for i in range(1):
            self.ropeend_pos_all(pos_move=pos_move.copy())
        
        self.rot_x_rads2(x_rads=self.overall_rot)

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
            #     self.reset_vel()
        # step_remain = l_shorten - 2* n_steps * step_len
        # self.ropeend_pos(np.array([step_remain, 0., 0.]))
        # print(self.sitename2pos("r_joint0_site"))
        # print(self.sitename2pos("r_joint0_site"))
        # print(self.sitename2pos("r_joint0_site")-self.sitename2pos("r_joint180_site"))
    
    def lhb_testing(self):
        print('0')
        for i in range(100):
            sys.stdout.write(f"\033[{1}F")
            print(f"testing stage: {i+1}/{100}")
            self.hold_pos(0.2)
            # if i % 1 == 0:
            #     self.reset_vel()
        # self.reset_vel()
        # hold_time = 30.
        # init_time = self.cur_time
        # while (self.cur_time-init_time) < hold_time:
        #     self.step()

        s_ss, fphi = self.lhb_var_compute()
        fphi_min_id = np.where(fphi == fphi.min())[0][0]
        s_ss_min_id = floor(len(s_ss) / 2)
        s_ss_center = s_ss - s_ss[s_ss_min_id]
        fphi_center = fphi.copy()
        min_id_diff = fphi_min_id - s_ss_min_id
        if min_id_diff > 0:
            fphi_center[:-min_id_diff] = fphi[min_id_diff:]
            fphi_center[-min_id_diff:] = np.ones(min_id_diff)
        elif min_id_diff < 0:
            fphi_center[-min_id_diff:] = fphi[:min_id_diff]
            fphi_center[:-min_id_diff] = np.ones(-min_id_diff)

        # while (self.cur_time-init_time) < hold_time:
        #     self.step()
        #     s_ss, fphi = self.lhb_var_compute()
        #     maxdevi_id = np.where(fphi == fphi.min())[0][0]
        #     s_ss_maxdevi = abs(s_ss[maxdevi_id])
        #     if s_ss_maxdevi < s_ss_mincenter or s_ss_center is None:
        #         s_ss_mincenter = s_ss_maxdevi
        #         s_ss_center = s_ss.copy()
        #         fphi_center = fphi.copy()
        return s_ss_center, fphi_center
    
    def lhb_var_compute(self):
        joint_site_pos = self.observations['rope_pose'].copy()
        t_vec = joint_site_pos[1:] - joint_site_pos[:self.r_pieces]
        for i in range(len(t_vec)):
            t_vec[i] = t_vec[i] / np.linalg.norm(t_vec[i])
        e_x = np.array([-1., 0., 0.])
        devi_set = np.arccos(np.dot(t_vec, e_x))
        max_devi = np.max(devi_set)
        s_step = self.r_len / (self.r_pieces)
        s = np.zeros(len(devi_set))
        for i in range(len(devi_set)):
            s[i] = s_step * i + s_step/2
        s = s - np.mean(s)
        max_devi2 = max_devi
        # max_devi2 = 0.919
        s_ss = (
            s * (self.beta_bar*12/(2*self.alpha_bar))
            * np.sqrt((1-np.cos(max_devi2))/(1+np.cos(max_devi2)))
        )
        # s_ss = s_ss - np.mean(s_ss)
        fphi = (np.cos(devi_set)-np.cos(max_devi))/(1-np.cos(max_devi))
        return s_ss, fphi

    def start_lhbtest(self, new_start):
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
        lhb_picklename = 'lhbtest{}_native.pickle'.format(self.r_pieces)
        lhb_picklename = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/lhb/native_test025/" + lhb_picklename
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

        s_ss_center, fphi_center =  self.lhb_testing()
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
        # input()
        # print(f"max_devi = {max_devi}")
        pickledata = [fphi_center, s_ss_center]
        pickledata_path = 'lhb{}_native.pickle'.format(self.r_pieces)
        pickledata_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/lhb/native_test025/" + pickledata_path
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
        joint_site_pos = self.observations['rope_pose'].copy()
        joint_site_pos = joint_site_pos[:self.r_pieces-1]
        # exclude 2 nodes to account for overlap
        return compute_PCA(
            joint_site_pos[:,0],
            joint_site_pos[:,1],
            joint_site_pos[:,2],
        )

    def start_circletest(self, new_start):
        # Test for multiple rope types (vary alpha and beta bar)
        # alpha = 1.
        # r_len = 2*np.pi * self.r_pieces / (self.r_pieces-1)
        self.freq_velreset = 0.2
        e_tol = 1000. # 0.075
        circtest_picklename = 'mbitest1_native.pickle'
        circtest_picklename = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/mbi/native/" + circtest_picklename
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

            self.set_state(self.init_pickle)
    
            self.rot_x_rads(x_rads=self.overall_rot)

            # # get and apply force normal to the circle
            norm_force = self.get_rope_normal()
            # norm_force /= 5.0
            # self.apply_force_t(t=0.3,force_dir=np.array([0., 0., 1.]))
            self.apply_force_t(t=0.8,force_dir=norm_force)

            # self.hold_pos(100.)
            # create a for loop that checks PCA error at each iter
            max_e = 0.
            print('0')
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
                print(f"Holding for {i+1} / {100}..")
                self.hold_pos(0.2)
                e_outofplane = self.check_e_PCA_circle()
                if e_outofplane > max_e:
                    max_e = e_outofplane
                # print(f'e_outofplane = {e_outofplane}')
            self.circle_oop = False
            print(f"e_tol = {e_tol}")
            print(f"e_outofplane = {e_outofplane}")
            print(f"max_e = {max_e}")
            if e_outofplane > e_tol or max_e > 5:
            # if e_outofplane > e_tol:
                self.circle_oop = True
                print(f'b_a = {self.beta_bar/self.alpha_bar} ==================================')
                print(f'out of plane theta_crit = {self.overall_rot} ==================================')
                # print(f'e_outofplane = {e_outofplane}')
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
        self.freq_velreset = 2000
        lhb_picklename = 'lhbtest{}_native.pickle'.format(self.r_pieces)
        lhb_picklename = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/lhb/our/" + lhb_picklename
        )
        with open(lhb_picklename, 'rb') as f:
            self.init_pickle = pickle.load(f)
        self.set_state(self.init_pickle)

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
        dist=None, azi=None, elev=None, lookat=None      
    ):
        if dist is not None:
            self.viewer.cam.distance = dist
        if azi is not None:
            self.viewer.cam.azimuth = azi
        if elev is not None:
            self.viewer.cam.elevation = elev
        if lookat is not None:
            self.viewer.cam.lookat = lookat

    def get_bodymass_total(self):
        return np.sum(self.model.body_mass[
            mjc2.obj_name2id(self.model,"body","B_first")
            :mjc2.obj_name2id(self.model,"body","B_last")+1
        ])

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
        self.hold_pos(0.05)

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

    def body_rot(self, body_id, rot_a=np.pi/180, rot_axis=0):
        rot_arr = np.zeros(3)
        rot_arr[rot_axis] = rot_a
        rot_quat = T.axisangle2quat(rot_arr)
        new_quat = T.quat_multiply(rot_quat, self.model.body_quat[body_id])
        self.model.body_quat[body_id] = new_quat
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