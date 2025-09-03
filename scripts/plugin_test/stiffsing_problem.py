"""
To-do:
"""
import mujoco
import adapteddlo_muj.utils.mjc2_utils as mjc2

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
"""
IMPT: Change step size to 0.0001 in world_ssing.xml for lighter/thinner wires
"""

loadfrompickle = True

stiff_type = 'wire'  # 'wire' or 'wire_qst'

"""
set damping and stiffness when changing length and number of pieces
"""

r_len = 0.50
r_pieces = 30
alpha_val = 0.001196450659614982    # Obtained from simple PI
beta_val = 0.001749108044378543
mass_per_length = 0.079/2.98
r_mass = mass_per_length * r_len
r_thickness = 0.012
j_damp = 0.002
overall_rot = 0.0

# r_len = 0.707
# r_thickness = 0.0141
# r_mass = 1
# alpha_val = 1.345/10
# beta_val = 0.789/10
# r_pieces = 30
# overall_rot = 0.0

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
    import pandas as pd
    from scipy import signal

    # Save the figure in the same folder as plot_mbicombined_demo.py
    fig_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "adapteddlo_muj/data/figs/plgn/"
    )
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "adapteddlo_muj/data/plugindata/"
    )
    csvsave_file = os.path.join(data_dir, "stiffsingcont_ft_data.csv")

    # Set style and font type as in plot_mbicombined_demo.py
    plt.rcParams.update({'pdf.fonttype': 42})
    plt.style.use('seaborn-v0_8')
    
    # Colors for different environments
    colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k']
    colors = [
        (0.7,0.2,0.2,0.8),
        (0.2,0.7,0.2,0.8),
        (0.2,0.2,0.7,0.8),
    ]
    
    plt.figure(figsize=(6, 4))
    
    all_results = []
    
    if not loadfrompickle:
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
            plt.plot(step_numbers, torque_data, linewidth=2, 
                    label=name, color=colors[i])
            
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
        t_data = np.column_stack((
            all_results[0]['step_numbers'],
            all_results[0]['torque_data'],
            all_results[1]['torque_data']
        ))
        np.savetxt(data_dir+"t_data.csv", t_data, delimiter=",", header="t1,t2,t3", comments="")
    else:
        # sim_data
        data = pd.read_csv(data_dir+"t_data.csv", header=None).to_numpy()
        step_numbers = data[:,0]
        for i in range(len(env_names)):
            torque_data = data[:,i+1]
            plt.plot(step_numbers, torque_data, linewidth=2, 
                    label=env_names[i], color=colors[i])

    # real data
    data = pd.read_csv(csvsave_file, header=None).to_numpy()
    step_numbers = data[:,0]
    step_numbers = step_numbers * 360 / step_numbers[-1]
    torque_data = data[:,1]
    plt.plot(step_numbers, torque_data, linewidth=2, 
            label="real", color=(0.7,0.7,0.7,0.5))
    # --- Filtering AFTER exclusion ---
    order = 3
    fc = 3.0  # cutoff Hz
    sample_rate = 200.0   # Hz
    sos = signal.butter(order, fc, btype='lowpass', fs=sample_rate, output='sos')
    torque_data_filtered = signal.sosfiltfilt(sos, torque_data)  # zero-phase
    plt.plot(step_numbers, torque_data_filtered, linewidth=2, 
            label="real_filtered", color=colors[2])

    # Finalize plot
    plt.xlabel('Rotation of Bottom End (°)', fontsize=12)
    plt.ylabel('Torque Magnitude (Nm)', fontsize=12)
    # plt.title('Torque vs Rotation Steps - Multiple Environments', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xticks(np.arange(0, 361, 90))
    plt.tight_layout()
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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "adapteddlo_muj/data/misc/" + miscdata_picklename
)
img_path = 'jointtorq_data.pickle'
img_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "adapteddlo_muj/data/img/"
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