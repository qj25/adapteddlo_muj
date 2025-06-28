"""
To-do:
"""
import mujoco
import adapteddlo_muj.utils.mjc2_utils as mjc2

import glfw
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time

from adapteddlo_muj.envs.stiff_singenv import TestStiffSingEnv

            # 
"""
todo:
    - fix position error when selecting (clicking) points in viewer

note: 
    - the reason why derinmuj and der_cpp values differ slightly is because
        der_cpp applies forces at step1 (env_step 0 to 1) 
        while derinmuj only calculates qfrc_passive for step2 (and only applies in step2)
        therefore, please use derinmuj to compare with subsequent plugins accuracy
    - the reason why the torq_node works in s2f for derinmuj and not der_cpp 
        is because it uses the misc qfrc_passive forces incl. damping and friction.
"""

#======================| Settings |======================
test_part = 0

loadfrompickle = False

stiff_type = 'wire'

use_specified_s2f = False

if use_specified_s2f:
    import adapteddlo_muj.utils.dlo_s2f_specified.Dlo_s2f as Dlo_s2f
else:
    import adapteddlo_muj.utils.dlo_s2f.Dlo_s2f as Dlo_s2f

"""
set damping and stiffness when changing length and number of pieces
"""

r_len = 0.707
r_thickness = 0.0141
r_mass = 1
alpha_val = 1.345/10
beta_val = 0.789/10
r_pieces = 30
overall_rot = 0.0

# r_len = r_pieces*1.0

do_render = True

# om_list = np.array([
#     # 0.*np.pi/10,
#     1.*np.pi/10,
#     2.*np.pi/10,
#     3.*np.pi/10,
#     4.*np.pi/10,
#     5.*np.pi/10,
#     6.*np.pi/10,
#     7.*np.pi/10,
#     8.*np.pi/10,
#     9.*np.pi/10,
#     10.*np.pi/10,
# ])
n_data = 20
om_list = np.linspace(0.0, 2.0*np.pi, n_data-1)
f_data = np.zeros((n_data-1, 3))

#======================| End Settings |======================

#======================| Move calc |======================

#======================| End Move calc |======================

#======================| Main |======================
miscdata_picklename = 'jointtorq_data.pickle'
miscdata_picklename = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dlo_check/data/misc/" + miscdata_picklename
)
img_path = 'jointtorq_data.pickle'
img_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dlo_check/data/img/"
)
# # exclude first free joint (6 indexes) 
# # from double welded-ends rope/cable.
if not loadfrompickle:
    print("Native:")
    env_native = TestStiffSingEnv(
        overall_rot=overall_rot,
        do_render=do_render,
        r_pieces=r_pieces,
        r_len=r_len,
        r_thickness=r_thickness,
        r_mass=r_mass,
        # rope_initpose=np.array([0., 0., 0.5, 0.707, 0, -0.707, 0]),
        alpha_bar=alpha_val,
        beta_bar=beta_val,
        stiff_type=stiff_type
    )
    # env_native.model.body_quat[env_native.ropeend_body_id][:] = np.array(
    #     [0.70710678, 0., -0.70710678, 0.]
    # )
    body_id = mjc2.obj_name2id(
        env_native.model, "body", "stiffrope"
    )

    # make vertical
    env_native.data.qpos[3:7] = np.array([0.7071, 0, -0.7071, 0])

    if use_specified_s2f:
        ds2f = Dlo_s2f.DLO_s2f(
            env_native.r_len,
            env_native.r_pieces,
            env_native.ropemass*(-env_native.model.opt.gravity[-1]),
            # env_native.ropemass*(9.81)
        )
    else:
        ds2f = Dlo_s2f.DLO_s2f(
            env_native.r_len,
            env_native.r_pieces,
            env_native.ropemass*(-env_native.model.opt.gravity[-1]),
            boolErrs=False,
            boolSolveTorq=True
            # env_native.ropemass*(9.81)
        )
    env_native.set_viewer_details(
        dist=1.5,
        azi=52.5,
        elev=-31.0,
        lookat=np.array([0.5, 0.0, 0.707])
    )

    env_native.test_force_curvature2(om_val=om_list[0])
    # input(env_native.data.xpos)
    # input(env_native.data.xquat)
    if test_part == 0:
        # turn off geom vis 2 and turn on geom frames
        # turning an offcentered object will move its weld by a radius offset to welded object
        env_native.viewer.vopt.geomgroup[2] ^= 1
        env_native.viewer.vopt.frame = 2
        # rotate bottom end by pi
        for i in range(90):
            env_native.ropeend_rot(rot_axis=2)
        """ WHY CANT I ROTATE 180 FOR WIRE_QST"""
        # when ready
        input("Press 'Enter' to start experiment part 1.. ..")
        # remove weld on box2
        env_native.data.eq_active[1] = 0
        pass
    else:
        # make part 2
        pass

    for i in range(10000000):
        add_fusr = False
        if add_fusr:
            # have to apply at every step
            # as each render call zeros current xfrc_applied
            objid = mjc2.obj_name2id(
                env_native.model,
                "body", "B_6"
            )
            objid2 = mjc2.obj_name2id(
                env_native.model,
                "body", "B_7"
            )
            # for i in range(env_native.model.nbody):
                # print(mjc2.obj_id2name(
                    # env_native.model,
                    # "body", i
                # ))
            # input()
            f_usr = np.array([0., 0., 0., 0.0, 0.0, 0.])
            ## fails for the below f_usr 
            ## due to quasi-static treatment of twist
            # f_usr = np.array([0., 0.1, 0., 0.3, 0., 0.])
            # new_vec = np.zeros(3)
            # mujoco.mju_rotVecQuat(
            #     new_vec,f_usr[3:],
            #     env_native.data.xquat[objid]
            # )
            # f_usr[3:] = new_vec.copy()
            env_native.data.xfrc_applied[objid] = f_usr.copy()
            fpos_usr = (
                env_native.data.xpos[objid]
                + env_native.data.xpos[objid+1]
            ) / 2.0
        # do force estimation
        if (env_native.env_steps-2)%100==0:
            efp = np.concatenate((
                # env_native.observations['rope_pose'][0].copy(),
                # env_native.observations['rope_pos'][2],
                # env_native.observations['rope_pose'][-1].copy(),
                env_native.observations['sensor0_pose'][:3].copy(),
                env_native.observations['sensor1_pose'][:3].copy(),
            ))
            ntorq = np.zeros_like(env_native.data.qfrc_passive.flatten())
            ntorq[:-3] = env_native.data.qfrc_passive.flatten()[3:]
            # ntorq[:-3] = env_native.stored_torques.flatten()[3:]
            npos = env_native.observations['rope_pose'].flatten()
            nquat = env_native.data.xquat[
                env_native.vec_bodyid[:env_native.r_pieces]
            ].flatten()
            nquat = np.concatenate((
                nquat,
                env_native.data.xquat[
                    mjc2.obj_name2id(env_native.model, "body", "B_last2")
                ]
            ))

            # efp = np.concatenate((
            #     npos.reshape((r_pieces+1,3))[-1],
            #     npos.reshape((r_pieces+1,3))[0]
            # ))
            print(f"||+++++++||=======||*******||+++++++||=======||*******||+++++++||=======||*******|")
            ## force added manually here
            if add_fusr:
                efp = np.concatenate((  # external force positions
                    efp, fpos_usr
                ))
            ## force added from viewer
            f_active = False
            if do_render:
                fvpos_usr, f_viewer, f_active = env_native.viewer.getuserFT()
            if f_active:
                efp = np.concatenate((
                    efp, fvpos_usr
                ))

            n_force = int(len(efp)/3)
            print(f"efp = {efp}")
            print(f"ntorq = {ntorq.reshape((r_pieces+1,3))}")
            print(f"npos = {npos}")
            print(f"nquat = {nquat}")
            if use_specified_s2f:
                ef = np.zeros((n_force,3)).flatten()
                et = np.zeros((n_force,3)).flatten()
                ds2f.calculateExternalForces(
                    efp, ef, et,
                    ntorq, npos, nquat
                )
                n_force_detected = n_force
                ef = ef.reshape((n_force_detected,3))
                et = et.reshape((n_force_detected,3))
            else:
                solvable_check = ds2f.calculateExternalForces(
                    ntorq, npos, nquat
                )
                if solvable_check:
                    n_force_detected = len(ds2f.force_sections)
                    ef = np.zeros((n_force_detected,3))
                    et = np.zeros((n_force_detected,3))
                    for ii in range(n_force_detected):
                        ef[ii] = ds2f.force_sections[ii].get_force()
                        et[ii] = ds2f.force_sections[ii].get_torque()
                else:
                    print("S2F not solvable, returning zeros.")
                    ef = np.zeros((n_force,3)).flatten()
                    et = np.zeros((n_force,3)).flatten()
            np.set_printoptions(precision=17, suppress=False)
            f_sensed = np.concatenate((
                [env_native.observations['ft_world_'+str(0)].reshape((2,3))[0]],
                [env_native.observations['ft_world_'+str(1)].reshape((2,3))[0]]
            ))
            t_sensed = np.concatenate((
                [env_native.observations['ft_world_'+str(0)].reshape((2,3))[1]],
                [env_native.observations['ft_world_'+str(1)].reshape((2,3))[1]]
            ))
            print(f"external_force = {ef}")
            print(f"force_sensed = {f_sensed}")
            if f_active:
                print(f"f_viewer = {f_viewer[:3]}")
            if add_fusr:
                print(f"f_usr = {f_usr[:3]}")
            print(f"external_torque = {et}")
            print(f"torque_sensed = {t_sensed}")
            if f_active:
                print(f"t_viewer = {f_viewer[3:]}")
            if add_fusr:
                print(f"t_usr = {f_usr[3:]}")
            # print(f"predict_err_f = {ef.reshape((n_force,3))-f_sensed}")
            # print(f"predict_err_t = {et.reshape((n_force,3))-t_sensed}")
            # piece_id = 8
            # print(f"force_piece{piece_id} = {env_native.data.xfrc_applied[piece_id]}")
            if f_active:
                print(f"f_viewer = {f_viewer}")
                print(f"fvpos_usr = {fvpos_usr}")

            # # center of mass calculations
            # com_val = env_native.data.subtree_com[mjc2.obj_name2id(
            #     env_native.model,
            #     "body", "B_last"
            # )]
            # print(f'com = {com_val}')

            # input()
            # env_native.viewer._paused = True
        env_native.step()


    env_native.viewer._paused = True
    env_native.step()
    env_native.viewer.render()

    # env_xfrc.viewer._paused = True
    # env_xfrc.step()
    # env_xfrc.viewer.render()

    
    # env_native.hold_pos(11.3)
    # native_pos_data = env_native.observations['rope_pose'].copy()
    # our_f_data = np.zeros(3)
    # our2_f_data = np.zeros(3)