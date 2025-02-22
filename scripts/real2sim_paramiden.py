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
from adapteddlo_muj.utils.optimize_utils import golden_section_search, midpoint_rootfind, mbi_stiff

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
test_id = args.testid
wire_color = args.wirecolor
test_type_g = args.testtype
do_render_g = bool(args.render)
new_start_g = bool(args.newstart)
lfp_g = bool(args.loadresults)

grav_on = True

# adjust these
stiff_lim = np.array([0.0,1.0])
# stiff_lim = np.array([0.025,0.040])
b_a_lim = np.array([0.0,3.0])
# b_a_lim = np.array([0.7,0.9])
# NOTE: larger beta --> smaller critical twist

# if test_type_g == 'mbi':
#     new_start_g = False


#======================| End Settings |======================
bendstiff_picklename = wire_color + '_' + stest_type + '_' + test_id + '_bendstiff.pickle'
bendstiff_picklename = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "adapteddlo_muj/data/dlo_muj_real/stiff_vals/" + bendstiff_picklename
)
stiff_picklename = wire_color + '_' + stest_type + '_stiff.pickle'
stiff_picklename = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "adapteddlo_muj/data/dlo_muj_real/stiff_vals/" + stiff_picklename
)

model_type2id = dict(
    adapt=0,
    xfrc=1,
    native=2,
)
# alpha_glob_arr 
# -- rows=model_type(adapt,xfrc,native)
# -- columns=wire_type(white,black,red)
# alpha_glob_arr = np.array([
#     [5.644018525809911e-05, 0.0009997975674291843, 0.00063696780505569],
#     [5.644018525809911e-05, 0.0009997975674291843, 0.00063696780505569],
#     [7.256595247469885e-05, 0.0010643006362955833, 0.0006732507812930395],
# ])
# b_a_glob_arr = np.array([
    # [1.05,1.10,1.89],
    # [1.05,1.10,1.89],
    # [0.85,1.05,1.80],
# ])

if test_type_g == 'twisting':
    n_test = 5
    alpha_all = []
    for i in range(n_test):
        bendstiff_picklename = wire_color + '_' + stest_type + '_' + str(i) + '_bendstiff.pickle'
        bendstiff_picklename = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "adapteddlo_muj/data/dlo_muj_real/stiff_vals/" + bendstiff_picklename
        )
        with open(bendstiff_picklename, 'rb') as f:
            alpha_all.append(pickle.load(f))
    alpha_all = np.array(alpha_all)
    alpha_glob = np.mean(alpha_all)
    alpha_cv = np.sum(np.linalg.norm(alpha_all-alpha_glob))/len(alpha_all)/alpha_glob
    print(f"alpha_all = {alpha_all}")
    print(f"alpha_cv = {alpha_cv*100.0}%")
    print(f"alpha_mean = {alpha_glob}")
    input()
    if lfp_g:
        with open(stiff_picklename, 'rb') as f:
            stiff_pickle = pickle.load(f)
        alpha_glob, b_a_glob = stiff_pickle
        print(f"alpha = {alpha_glob}")
        print(f"beta = {alpha_glob*b_a_glob}")
        print(f"b/a = {b_a_glob}")
        input()
rope_len = 1.5
deg2rad = np.pi/180.0
if wire_color == 'white':
    rgba_vals = np.concatenate((np.array([300,300,300])/300,[1]))
    massperlen = 0.087/5.0
    # alpha_glob = alpha_glob_arr[model_type2id[stest_type],0]
    crittwist_all = np.array([970.0, 970.0, 980.0, 975.0, 980.0])
    ord_glob = np.mean(crittwist_all)    # input critical twist here from real experiments
elif wire_color == 'black':
    rgba_vals = np.concatenate((np.array([0,0,0])/300,[1]))
    massperlen = 0.081/2.98
    # alpha_glob = alpha_glob_arr[model_type2id[stest_type],1]
    crittwist_all = np.array([470.0, 500.0, 480.0, 470.0, 490.0])
    ord_glob = np.mean(crittwist_all)    # input critical twist here from real experiments
elif wire_color == 'red':
    rgba_vals = np.concatenate((np.array([300,0,0])/300,[1]))
    massperlen = 0.043/2.0
    # alpha_glob = alpha_glob_arr[model_type2id[stest_type],2]
    crittwist_all = np.array([430.0, 470.0, 445.0, 445.0, 460.0])
    ord_glob = np.mean(crittwist_all)    # input critical twist here from real experiments

if grav_on:
    picklefolder = 'real/grav'
else:
    picklefolder = 'real/nograv'

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
    mbi_stiff2 = mbi_stiff(
        stest_type=stest_type,
        rgba_vals=rgba_vals,
        massperlen=massperlen,
        overall_rot=0,
        r_len=rope_len,
        do_render=do_render_g,
        new_start=new_start_g
    )
    mbi_stiff2.alpha_bar = alpha_glob
    mbi_stiff2.overall_rot = ord_glob * deg2rad
    best_b_a_val = midpoint_rootfind(
        mbi_stiff2.opt_func2,
        b_a_lim[0],
        b_a_lim[1],
        tol=1e-2
    )
    print("Beta/Alpha Found!")
    print(f"b/a = {best_b_a_val}")
    stiff_pickle = [alpha_glob,best_b_a_val]
    with open(stiff_picklename, 'wb') as f:
        pickle.dump(stiff_pickle,f)
if test_type_g == 'bending':
    # load pickle of pos
    realdata_picklename = wire_color + test_id + '_data.pickle'
    realdata_picklename = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "adapteddlo_muj/data/dlo_muj_real/" + realdata_picklename
    )
    
    with open(realdata_picklename, 'rb') as f:
        real_pos, _ = pickle.load(f)

    mbi_stiff1 = mbi_stiff(
        stest_type=stest_type,
        rgba_vals=rgba_vals,
        real_pos=real_pos,
        massperlen=massperlen,
        overall_rot=0,
        r_len=rope_len,
        do_render=do_render_g,
        new_start=new_start_g
    )
    best_stiffval = golden_section_search(
        mbi_stiff1.opt_func,
        stiff_lim[0],
        stiff_lim[1],
        tol=1e-3
    )
    min_diff = mbi_stiff1.opt_func(best_stiffval)
    alpha_stiff = best_stiffval/(2*np.pi)**3
    print("Minimum Found!")
    print(f"final alpha = {alpha_stiff}")
    print(f"stiff_scale = {best_stiffval}")
    print(f"min_diff = {min_diff}")
    
    bendstiff_pickle = alpha_stiff
    with open(bendstiff_picklename, 'wb') as f:
        pickle.dump(bendstiff_pickle,f)