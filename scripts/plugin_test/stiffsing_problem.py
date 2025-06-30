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
test_part = 1

loadfrompickle = False

stiff_type = 'wire_qst'

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

#======================| Extra Func |======================
def rotate_and_record_torque_multi(env_list, env_names, total_steps=360, record_interval=10):
    """
    Rotate the rope end by 1 degree per step for total_steps for multiple environments,
    record torque data every record_interval steps, and plot all results on the same graph.
    
    Args:
        env_list: List of environment objects
        env_names: List of environment names for legend
        total_steps: Total number of rotation steps (default: 360)
        record_interval: Interval between torque recordings (default: 10)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    # Set style and font type as in plot_mbicombined_demo.py
    plt.rcParams.update({'pdf.fonttype': 42})
    plt.style.use('seaborn-v0_8')
    
    # Colors for different environments
    colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k']
    
    plt.figure(figsize=(6, 4))
    
    all_results = []
    
    for i, (env, name) in enumerate(zip(env_list, env_names)):
        print(f"\n=== Testing {name} ===")
        
        # Initialize data storage for this environment
        torque_data = []
        step_numbers = []
        
        print(f"Starting rotation: {total_steps} steps, recording every {record_interval} steps")
        torque_data.append(0.0)
        step_numbers.append(0)
        for step in range(total_steps):
            # Rotate by 1 degree (π/180 radians)
            env.ropeend_rot(rot_a=np.pi/180, rot_axis=2)
            
            # Record torque data every record_interval steps
            if (step + 1) % record_interval == 0:
                # Hold for 1 second
                env.hold_pos(1.0)
                # Get torque data from observations
                # torque_magnitude = np.linalg.norm(env.observations['ft_world_1'][3:])
                torque_magnitude = -env.observations['ft_world_1'][5]
                torque_data.append(torque_magnitude)
                step_numbers.append(step + 1)
                
                print(f"Step {step + 1}: Torque magnitude = {torque_magnitude:.4f}")
        
        # Plot this environment's data
        color = colors[i % len(colors)]
        print(torque_data)
        plt.plot(step_numbers, torque_data, linewidth=2, 
                label=name, alpha=0.8)
        
        # Store results
        all_results.append({
            'name': name,
            'step_numbers': step_numbers,
            'torque_data': torque_data
        })
        
        # Print summary statistics for this environment
        print(f"\n{name} Summary:")
        print(f"Total data points: {len(torque_data)}")
        print(f"Average torque: {np.mean(torque_data):.4f} N⋅m")
        print(f"Max torque: {np.max(torque_data):.4f} N⋅m")
        print(f"Min torque: {np.min(torque_data):.4f} N⋅m")
    
    # Finalize plot
    plt.xlabel('Rotation of Bottom End (°)', fontsize=12)
    plt.ylabel('Torque Magnitude (Nm)', fontsize=12)
    # plt.title('Torque vs Rotation Steps - Multiple Environments', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xticks(np.arange(0, 361, 90))
    plt.tight_layout()
    # Save the figure in the same folder as plot_mbicombined_demo.py
    fig_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "adapteddlo_muj/data/figs/plgn/"
    )
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "stiffsing2_plot.pdf")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {fig_path}")
    
    return all_results

# Example usage:
# all_results = rotate_and_record_torque_multi([env_wireqst, env_wire], ['wire_qst', 'wire'])

#======================| End Extra Func |======================

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

def make_env(plugin_name):
# from double welded-ends rope/cable.
    print("Native:")
    env = TestStiffSingEnv(
        overall_rot=overall_rot,
        do_render=do_render,
        r_pieces=r_pieces,
        r_len=r_len,
        r_thickness=r_thickness,
        r_mass=r_mass,
        # rope_initpose=np.array([0., 0., 0.5, 0.707, 0, -0.707, 0]),
        alpha_bar=alpha_val,
        beta_bar=beta_val,
        stiff_type=plugin_name
    )
    body_id = mjc2.obj_name2id(
        env.model, "body", "stiffrope"
    )
    # make vertical
    env.data.qpos[3:7] = np.array([0.7071, 0, -0.7071, 0])
    if do_render:
        env.set_viewer_details(
            dist=1.5,
            azi=52.5,
            elev=-31.0,
            lookat=np.array([0.3535, 0.0, 0.85])
        )
        env.viewer.vopt.frame = 2
    env.test_force_curvature2(om_val=om_list[0])
    return env

if test_part == 0:
    # turn off geom vis 2 and turn on geom frames
    # turning an offcentered object will move its weld by a radius offset to welded object
    env_test = make_env(plugin_name=stiff_type)
    env_test.viewer.vopt.geomgroup[2] ^= 1
    env_test.viewer.vopt.frame = 2
    # rotate bottom end by pi
    for i in range(360):
        if (i + 1) % 10 == 0:
            # Hold for 1 second
            env_test.hold_pos(1.0)

        env_test.ropeend_rot(rot_axis=2)
    # when ready
    env_test.viewer._paused = True
    print("Press 'Space' to start experiment part 1.. ..")
    # remove weld on box2
    env_test.data.eq_active[1] = 0
    env_test.hold_pos(5.0)
    env_test.viewer._paused = True
    while True:
        env_test.step()

else:
    do_render = False
    # Test both wire_qst and wire models
    print("=== Testing Multiple Environments ===")
    
    # Create environments
    env_wireqst = make_env('wire_qst')
    env_wire = make_env('wire')
    
    # Test both environments and plot on same graph
    all_results = rotate_and_record_torque_multi(
        # [env_wireqst], ['wire_qst']
        [env_wireqst, env_wire], 
        ['adapted', 'j-DER']
    )


    # if use_specified_s2f:
        # ds2f = Dlo_s2f.DLO_s2f(
            # env_native.r_len,
            # env_native.r_pieces,
            # env_native.ropemass*(-env_native.model.opt.gravity[-1]),
            # # env_native.ropemass*(9.81)
        # )
    # else:
        # ds2f = Dlo_s2f.DLO_s2f(
            # env_native.r_len,
            # env_native.r_pieces,
            # env_native.ropemass*(-env_native.model.opt.gravity[-1]),
            # boolErrs=False,
            # boolSolveTorq=True
            # # env_native.ropemass*(9.81)
        # )
    # input(env_native.data.xpos)
    # input(env_native.data.xquat)

    # for i in range(10000000):
    #     add_fusr = False
    #     if add_fusr:
    #         # have to apply at every step
    #         # as each render call zeros current xfrc_applied
    #         objid = mjc2.obj_name2id(
    #             env_native.model,
    #             "body", "B_6"
    #         )
    #         objid2 = mjc2.obj_name2id(
    #             env_native.model,
    #             "body", "B_7"
    #         )
    #         f_usr = np.array([0., 0., 0., 0.0, 0.0, 0.])
    #         env_native.data.xfrc_applied[objid] = f_usr.copy()
    #         fpos_usr = (
    #             env_native.data.xpos[objid]
    #             + env_native.data.xpos[objid+1]
    #         ) / 2.0
    #     # do force estimation
    #     if (env_native.env_steps-2)%100==0:
    #         efp = np.concatenate((
    #             # env_native.observations['rope_pose'][0].copy(),
    #             # env_native.observations['rope_pos'][2],
    #             # env_native.observations['rope_pose'][-1].copy(),
    #             env_native.observations['sensor0_pose'][:3].copy(),
    #             env_native.observations['sensor1_pose'][:3].copy(),
    #         ))
    #         ntorq = np.zeros_like(env_native.data.qfrc_passive.flatten())
    #         ntorq[:-3] = env_native.data.qfrc_passive.flatten()[3:]
    #         # ntorq[:-3] = env_native.stored_torques.flatten()[3:]
    #         npos = env_native.observations['rope_pose'].flatten()
    #         nquat = env_native.data.xquat[
    #             env_native.vec_bodyid[:env_native.r_pieces]
    #         ].flatten()
    #         nquat = np.concatenate((
    #             nquat,
    #             env_native.data.xquat[
    #                 mjc2.obj_name2id(env_native.model, "body", "B_last2")
    #             ]
    #         ))
    #         print(f"||+++++++||=======||*******||+++++++||=======||*******||+++++++||=======||*******|")
    #         ## force added manually here
    #         if add_fusr:
    #             efp = np.concatenate((  # external force positions
    #                 efp, fpos_usr
    #             ))
    #         ## force added from viewer
    #         f_active = False
    #         if do_render:
    #             fvpos_usr, f_viewer, f_active = env_native.viewer.getuserFT()
    #         if f_active:
    #             efp = np.concatenate((
    #                 efp, fvpos_usr
    #             ))

    #         n_force = int(len(efp)/3)
    #         print(f"efp = {efp}")
    #         print(f"ntorq = {ntorq.reshape((r_pieces+1,3))}")
    #         print(f"npos = {npos}")
    #         print(f"nquat = {nquat}")
    #         if use_specified_s2f:
    #             ef = np.zeros((n_force,3)).flatten()
    #             et = np.zeros((n_force,3)).flatten()
    #             ds2f.calculateExternalForces(
    #                 efp, ef, et,
    #                 ntorq, npos, nquat
    #             )
    #             n_force_detected = n_force
    #             ef = ef.reshape((n_force_detected,3))
    #             et = et.reshape((n_force_detected,3))
    #         else:
    #             solvable_check = ds2f.calculateExternalForces(
    #                 ntorq, npos, nquat
    #             )
    #             if solvable_check:
    #                 n_force_detected = len(ds2f.force_sections)
    #                 ef = np.zeros((n_force_detected,3))
    #                 et = np.zeros((n_force_detected,3))
    #                 for ii in range(n_force_detected):
    #                     ef[ii] = ds2f.force_sections[ii].get_force()
    #                     et[ii] = ds2f.force_sections[ii].get_torque()
    #             else:
    #                 print("S2F not solvable, returning zeros.")
    #                 ef = np.zeros((n_force,3)).flatten()
    #                 et = np.zeros((n_force,3)).flatten()
    #         np.set_printoptions(precision=17, suppress=False)
    #         f_sensed = np.concatenate((
    #             [env_native.observations['ft_world_'+str(0)].reshape((2,3))[0]],
    #             [env_native.observations['ft_world_'+str(1)].reshape((2,3))[0]]
    #         ))
    #         t_sensed = np.concatenate((
    #             [env_native.observations['ft_world_'+str(0)].reshape((2,3))[1]],
    #             [env_native.observations['ft_world_'+str(1)].reshape((2,3))[1]]
    #         ))
    #         print(f"external_force = {ef}")
    #         print(f"force_sensed = {f_sensed}")
    #         if f_active:
    #             print(f"f_viewer = {f_viewer[:3]}")
    #         if add_fusr:
    #             print(f"f_usr = {f_usr[:3]}")
    #         print(f"external_torque = {et}")
    #         print(f"torque_sensed = {t_sensed}")
    #         if f_active:
    #             print(f"t_viewer = {f_viewer[3:]}")
    #         if add_fusr:
    #             print(f"t_usr = {f_usr[3:]}")
    #         # print(f"predict_err_f = {ef.reshape((n_force,3))-f_sensed}")
    #         # print(f"predict_err_t = {et.reshape((n_force,3))-t_sensed}")
    #         # piece_id = 8
    #         # print(f"force_piece{piece_id} = {env_native.data.xfrc_applied[piece_id]}")
    #         if f_active:
    #             print(f"f_viewer = {f_viewer}")
    #             print(f"fvpos_usr = {fvpos_usr}")

    #         # # center of mass calculations
    #         # com_val = env_native.data.subtree_com[mjc2.obj_name2id(
    #         #     env_native.model,
    #         #     "body", "B_last"
    #         # )]
    #         # print(f'com = {com_val}')

    #         # input()
    #         # env_native.viewer._paused = True
    #     env_native.step()


    # env_native.viewer._paused = True
    # env_native.step()
    # env_native.viewer.render()

    # env_xfrc.viewer._paused = True
    # env_xfrc.step()
    # env_xfrc.viewer.render()

    
    # env_native.hold_pos(11.3)
    # native_pos_data = env_native.observations['rope_pose'].copy()
    # our_f_data = np.zeros(3)
    # our2_f_data = np.zeros(3)

