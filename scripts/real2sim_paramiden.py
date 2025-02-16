"""
To-do:
"""

import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

from adapteddlo_muj.envs.realgrav_valid_test import TestRopeEnv
from adapteddlo_muj.utils.argparse_utils import r2spi_parse

"""
1. do real twist mbi w grav 
    - real experiment
2. use 0 twist shape to find bending stiffness 
    - in camstuff - run get_dermuj_shape - select wire from right holder to left holder, autosaved pickle
    - from pickle, in dlo_check - run real_testdata with test_type_g = 'mbi_teststiff'.
    - adjust stiff_scale_list to find stiff_scale and alpha with lowest diff_pos
    - add alpha to alpha_glob for the corresponding wire
3. use critical twist to find beta/alpha ratio (test_type_g = 'mbi_teststiff')
    - with new alpha_glob, input ord_glob(overallrotdeg = realcriticaltwist) for corresponding wire
    - adjust bar_glob to get b/a (b_a) where buckling occurs (to 1e-2 accuracy)
    - get corresponding beta
4. use stiffness to compare shapes of real with sims for a few scenarios (depth cam vs real wire)
    - run jointtorq_test for different wire end movements to compare with real wire held by robot arm

Qn: should you do steps 1-3 (parameter identification) with own model or common model (adapted)?
NOTE: b_a tends to be higher for wires due to their small bending stiffness

DATA:   (A-adapted, X-direct, N-native)
white (last number is alpha row is the diff_pos)
-A- alpha = 5.644018525809911e-05 (stiff_scale = 0.013) - 0.007771766311904944
-A- beta = 3.837932597550739e-05 (b_a = 0.68)
-X- alpha = 5.644018525809911e-05 (stiff_scale = 0.013) - 0.007679568747663055
-X- beta = 3.837932597550739e-05 (b_a = 0.68)
-N- alpha = 7.256595247469885e-05 (stiff_scale = 0.018) - 0.0068583062598774405
-N- beta = 5.152182625703618e-05 (b_a = 0.71)

black 
-A- alpha = 0.0009997975674291843 (stiff_scale = 0.248) - 0.01577049623151956
-A- beta = 0.001099777324172103 (b_a = 1.10)
-X- alpha = 0.0009997975674291843 (stiff_scale = 0.248) - 0.015712488291255093
-X- beta = 0.001099777324172103 (b_a = 1.10)
-N- alpha = 0.0010643006362955833 (stiff_scale = 0.264) - 0.015673797398902685
-N- beta = 0.0011175156681103625 (b_a = 1.05)

red 
-A- alpha = 0.00063696780505569 (stiff_scale = 0.158) - 0.013824481117311245
-A- beta = 0.001203869151555254 (b_a = 1.89)
-X- alpha = 0.00063696780505569 (stiff_scale = 0.158) - 0.013749439033478634
-X- beta = 0.001203869151555254 (b_a = 1.89)
-N- alpha = 0.0006732507812930395 (stiff_scale = 0.167) - 0.013383626116227867
-N- beta = 0.0012118514063274711 (b_a = 1.80)

"""

#======================| Settings |======================
parser = r2spi_parse()
args = parser.parse_args()
stest_type = args.stiff
wire_color = args.wirecolor
test_type_g = args.testtype
do_render_g = args.render
new_start_g = bool(args.newstart)
lfp_g = bool(args.loadresults)

grav_on = True

# adjust these
stiff_scale_list = np.linspace(0.3e-2,1.3e-2,11)
# stiff_scale_list = np.linspace(1.85e-2,1.95e-2,11)
# stiff_scale_list = np.linspace(0.247,0.257,11)
# bar_glob = np.array([1.0,1.1]) # b_a_ratio_global adjust beta_bar here to get appropriate ratio
bar_glob = np.array([0.80,0.90]) # b_a_ratio_global adjust beta_bar here to get appropriate ratio
# NOTE: larger beta --> smaller critical twist

# if test_type_g == 'mbi':
#     new_start_g = False


#======================| End Settings |======================
model_type2id = dict(
    adapt=0,
    xfrc=1,
    native=2,
)
# alpha_glob_arr 
# -- rows=model_type(adapt,xfrc,native)
# -- columns=wire_type(white,black,red)
alpha_glob_arr = np.array([
    [5.644018525809911e-05, 0.0009997975674291843, 0.00063696780505569],
    [5.644018525809911e-05, 0.0009997975674291843, 0.00063696780505569],
    [7.256595247469885e-05, 0.0010643006362955833, 0.0006732507812930395],
])
b_a_glob_arr = np.array([
    [1.05,1.10,1.89],
    [1.05,1.10,1.89],
    [0.85,1.05,1.80],
])

rope_len = 1.5
deg2rad = np.pi/180.0
if wire_color == 'white':
    rgba_vals = np.concatenate((np.array([300,300,300])/300,[1]))
    massperlen = 0.087/5.0
    alpha_glob = alpha_glob_arr[model_type2id[stest_type],0]
    ord_glob = 1160.0    # input critical twist here from real experiments
elif wire_color == 'black':
    rgba_vals = np.concatenate((np.array([0,0,0])/300,[1]))
    massperlen = 0.081/2.98
    alpha_glob = alpha_glob_arr[model_type2id[stest_type],1]
    ord_glob = 640.0    # input critical twist here from real experiments
elif wire_color == 'red':
    rgba_vals = np.concatenate((np.array([300,0,0])/300,[1]))
    massperlen = 0.043/2.0
    alpha_glob = alpha_glob_arr[model_type2id[stest_type],2]
    ord_glob = 450.0    # input critical twist here from real experiments

if grav_on:
    picklefolder = 'real/grav'
else:
    picklefolder = 'real/nograv'

def mbi_data(
    beta_bar,
    theta_crit,
    alpha_bar=1.,
):
    b_a = beta_bar / alpha_bar
    return b_a, theta_crit

def mbi_plot(b_a, theta_crit):
    b_a_base = b_a.copy()
    theta_crit_base = 2*np.pi*np.sqrt(3)/(b_a_base)

    max_devi_theta_crit = np.max(np.abs(theta_crit_base - theta_crit))
    avg_deviation = np.linalg.norm(theta_crit_base - theta_crit) / len(theta_crit)
    print(f"max_devi_theta_crit = {max_devi_theta_crit}")
    print(f"avg_deviation = {avg_deviation}")

    plt.figure(f"Michell's Buckling Instability for Adapted with gravity")
    plt.xlabel(r"$\beta/\alpha$")
    plt.ylabel(r'$\theta^n$')
    plt.plot(b_a_base, theta_crit_base)
    plt.plot(b_a, theta_crit)
    plt.legend(['Analytical','Simulation'])
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def mbi_indivtest(
    overall_rot=0.,
    r_len=1.0,
    alpha_val=1.,
    beta_val=1.,
    do_render=False,
    new_start=False,
):
    r_pieces = 52
    r_len = r_len * r_pieces / (r_pieces-2)
    r_thickness = 0.01

    # r_mass = r_len * 2.
    # r_mass = r_len
    r_mass = massperlen * r_len
    
    env = TestRopeEnv(
        overall_rot=overall_rot,
        do_render=do_render,
        r_pieces=r_pieces,
        r_len=r_len,
        r_thickness=r_thickness,
        test_type='mbi',
        alpha_bar=alpha_val,
        beta_bar=beta_val,
        r_mass=r_mass,
        new_start=new_start,
        stifftorqtype=stest_type,
        grav_on=grav_on,
        rgba_vals=rgba_vals
    )

    if not env.circle_oop:
        theta_crit = 0.
    else:
        theta_crit = overall_rot
    
    # env.do_render = False
    if do_render:
        env.viewer.close()
    env.close()
    return theta_crit

def mbi_indivtest2(
    overall_rot=0.,
    r_len=1.0,
    alpha_val=1.,
    beta_val=1.,
    do_render=False,
    new_start=False,
):
    # r_pieces = 50
    # r_len = r_len * r_pieces / (r_pieces-1)
    r_pieces = 52
    r_len = r_len * r_pieces / (r_pieces-2)
    r_thickness = 0.01
    # r_mass = r_len * 2.
    # r_mass = r_len
    r_mass = massperlen * r_len

    env = TestRopeEnv(
        overall_rot=overall_rot,
        do_render=do_render,
        r_pieces=r_pieces,
        r_len=r_len,
        r_thickness=r_thickness,
        test_type='mbi',
        alpha_bar=alpha_val,
        beta_bar=beta_val,
        r_mass=r_mass,
        new_start=new_start,
        stifftorqtype=stest_type,
        grav_on=grav_on,
        rgba_vals=rgba_vals
    )
    if do_render:
        env.set_viewer_details(
            dist=1.5,
            azi=90.0,
            elev=0.0,
            lookat=np.array([-0.75,0.0,0.10])
        )
    # st_steps = 1000
    # print(f"Testing for {0} / {st_steps} steps..")
    # for i in range(1000):
    #     if (i+1) % 100 == 0:
    #         sys.stdout.write(f"\033[{1}F")
    #         print(f"Testing for {i+1} / {st_steps} steps..")
    #     env.step()
    return env.observations['rope_pose'].copy()

def mbi_test(new_start=False, load_from_pickle=False, do_render=False):
    mbi_picklename = picklefolder + '/mbi1.pickle'
    mbi_picklename = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dlo_check/data/mbi/" + mbi_picklename
    )
    if load_from_pickle:
        print('Loading MBI test...')
        with open(mbi_picklename, 'rb') as f:
            pickle_mbidata = pickle.load(f)
        idhalf_pickle = round(len(pickle_mbidata)/2)
        b_a = pickle_mbidata[:idhalf_pickle]
        theta_crit = pickle_mbidata[idhalf_pickle:]
    else:
        print('Starting MBI test...')
        n_data_mbi = 11
        beta_bar_lim = np.array([0.5, 1.25])
        if bar_glob is not None:
            beta_bar_lim = bar_glob.copy()
        beta_bar_step = (beta_bar_lim[1] - beta_bar_lim[0]) / (n_data_mbi - 1)
        beta_bar = np.zeros(n_data_mbi)
        alpha_bar = np.zeros(n_data_mbi)
        for i in range(n_data_mbi):
            beta_bar[i] = beta_bar_step * i + beta_bar_lim[0]
            alpha_bar[i] = 1.
        alpha_bar *= alpha_glob
        beta_bar *= alpha_glob
        b_a = beta_bar / alpha_bar

        theta_crit = np.zeros(n_data_mbi)
        if new_start:
            theta_crit[i] = mbi_indivtest(
                r_len=rope_len,
                alpha_val=alpha_bar[i],
                beta_val=beta_bar[i],
                new_start=new_start,
                do_render=do_render
            )

        overall_rot_deg = 360.0
        if ord_glob is not None:
            overall_rot_deg = ord_glob
        # overall_rot = 3.0
        # overall_rot = 5.5
        # overall_rot = 21.
        for i in range(n_data_mbi):
            # overall_rot = 7.5
            print(f'b_a = {b_a[i]}')
            print(f'overall_rot = {overall_rot_deg}')
            theta_crit[i] = mbi_indivtest(
                r_len=rope_len,
                overall_rot=overall_rot_deg*deg2rad,
                alpha_val=alpha_bar[i],
                beta_val=beta_bar[i],
                # alpha_val=1.0,
                # beta_val=0.5,
                do_render=do_render
            )
            if theta_crit[i] > 1e-7:
                print("b_a_r found!")
                print(f"b_a = {b_a[i]}")
                print(f"alpha = {alpha_bar[i]}")
                print(f"beta = {beta_bar[i]}")
                break
                # overall_rot += 0.1 # * (np.pi/180)
        pickle_mbidata = np.concatenate((b_a,theta_crit))
        with open(mbi_picklename, 'wb') as f:
            pickle.dump(pickle_mbidata,f)
            print('mbi test data saved!')
    
    mbi_plot(b_a=b_a, theta_crit=theta_crit)

def alter_simropepos(pos_arr):
    # altering so that the first and last capsules are halved (defunct)
    pos_arr[0] = (pos_arr[0] + pos_arr[1])/2.0  # measure from the exit of holder 
    # (half rope piece on each end intersects holder on purpose)
    pos_arr[-1] = (pos_arr[-1] + pos_arr[-2])/2.0   # measure from the exit of holder
    pos_arr -= pos_arr[0]   # reference position from the right of holder
    return - pos_arr    # change in reference to real image

def alter_simropepos2(pos_arr):
    return pos_arr[1:-1].copy()

if test_type_g == 'twisting':    
    mbi_test(new_start=new_start_g, load_from_pickle=lfp_g, do_render=do_render_g)
if test_type_g == 'bending':
    # load pickle of pos
    realdata_picklename = wire_color + '0_data.pickle'
    realdata_picklename = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data/der_muj_real/" + realdata_picklename
    )
    with open(realdata_picklename, 'rb') as f:
        real_pos, _ = pickle.load(f)


    min_diff = 999.
    for i in range(len(stiff_scale_list)):
        sim_pos = mbi_indivtest2(
            r_len=rope_len,
            alpha_val=1.0*stiff_scale_list[i]/(2*np.pi)**3,
            beta_val=1.0*stiff_scale_list[i]/(2*np.pi)**3,    # beta has stiffness to discourage twisting
            new_start=new_start_g,
            do_render=do_render_g
        )
        sim_pos = sim_pos[:,[0,2]]
        sim_pos = alter_simropepos(sim_pos)
        
        diff_pos = np.sum(np.linalg.norm(sim_pos-real_pos,axis=1))/len(sim_pos)
        print("|===|NEW DATA |================================================")
        print(f"realpos = {real_pos}")
        print(f"simpos = {sim_pos}")
        print(f"stiff_scale = {stiff_scale_list[i]}")
        print(f"diff_pos = {diff_pos}")
        if diff_pos < min_diff:
            min_diff = diff_pos
        else:
            print("Minimum Found!")
            print(f"final alpha = {1.0*stiff_scale_list[i-1]/(2*np.pi)**3}")
            print(f"stiff_scale = {stiff_scale_list[i-1]}")
            print(f"id = {i-1}")
            break