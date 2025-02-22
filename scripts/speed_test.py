"""
To-do:
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pickle

from adapteddlo_muj.envs.plain_rope_valid_test import TestPlainRopeEnv
from adapteddlo_muj.envs.our_xfrc_rope_valid_test import TestRopeXfrcEnv
from adapteddlo_muj.envs.our_rope_valid_test import TestRopeEnv
from adapteddlo_muj.envs.native_cable_valid_test import TestCableEnv
from adapteddlo_muj.utils.argparse_utils import spdt_parse
from adapteddlo_muj.utils.plotter import plot_computetime

#======================| Settings |======================
parser = spdt_parse()
args = parser.parse_args()

new_start = bool(args.newstart)

test_type = 'speedtest2'

r_len = 9.29
r_thickness = 0.05
alpha_val = 1.345
beta_val = 0.789
if test_type == 'speedtest1':
    r_pieces_list = [20,30,40,50,60]
else:
    r_pieces_list = [40, 60, 80, 110, 140, 180]
    # r_pieces_list = [80]
# r_pieces_list = [80]
#======================| End Settings |======================

speedtest_picklename = f'{test_type}_res.pickle'
speedtest_picklename = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "adapteddlo_muj/data/speed_test/" + speedtest_picklename
)

if new_start:
    t_list = np.zeros((len(r_pieces_list),4))

    for i in range(len(r_pieces_list)):
        print(f"Testing speed for {r_pieces_list[i]} pieces.. ..")
        # Plain
        print("Plain:")
        env_plain = TestPlainRopeEnv(
            overall_rot=0.0,
            do_render=False,
            r_pieces=r_pieces_list[i],
            r_len=r_len,
            r_thickness=r_thickness,
            test_type=test_type,
            alpha_bar=alpha_val,
            beta_bar=beta_val,
        )
        if test_type == 'speedtest1':
            t_list[i,0] = env_plain.run_speedtest1()
        else:
            t_list[i,0] = env_plain.run_speedtest2()

        # Native
        print("Native:")
        env_native = TestCableEnv(
            overall_rot=0.0,
            do_render=False,
            r_pieces=r_pieces_list[i],
            r_len=r_len,
            r_thickness=r_thickness,
            test_type=test_type,
            alpha_bar=alpha_val,
            beta_bar=beta_val,
        )
        if test_type == 'speedtest1':
            t_list[i,1] = env_native.run_speedtest1()
        else:
            t_list[i,1] = env_native.run_speedtest2()

        # Our
        print("Xfrc:")
        env_xfrc = TestRopeXfrcEnv(
            overall_rot=0.0,
            do_render=False,
            r_pieces=r_pieces_list[i],
            r_len=r_len,
            r_thickness=r_thickness,
            test_type=test_type,
            alpha_bar=alpha_val,
            beta_bar=beta_val,
        )
        if test_type == 'speedtest1':
            t_list[i,2] = env_xfrc.run_speedtest1()
        else:
            t_list[i,2] = env_xfrc.run_speedtest2()

        # Our2
        print("Bal:")
        env_our = TestRopeEnv(
            overall_rot=0.0,
            do_render=False,
            r_pieces=r_pieces_list[i],
            r_len=r_len,
            r_thickness=r_thickness,
            test_type=test_type,
            alpha_bar=alpha_val,
            beta_bar=beta_val,
            stifftorqtype='bal'
        )
        if test_type == 'speedtest1':
            t_list[i,3] = env_our.run_speedtest1()
        else:
            t_list[i,3] = env_our.run_speedtest2()


    speedtest_data = [r_pieces_list, t_list]
    input("Saving pickle. Press 'Enter' to confirm.. ..")
    with open(speedtest_picklename, 'wb') as f:
        pickle.dump(speedtest_data,f)
    print("Pickle saved!")

else:
    with open(speedtest_picklename, 'rb') as f:
        r_pieces_list, t_list = pickle.load(f)
    print("Pickle loaded!")

    plot_computetime(
        r_pieces_list, 
        [
            t_list[:,0],
            t_list[:,1],
            t_list[:,2],
            t_list[:,3],
        ]
    )

    # plt.figure("Speed Tests")
    # plt.xlabel("r_pieces")
    # plt.ylabel('time ratio (real/sim)')
    # input(len(t_list[0]))
    # for i in range(len(t_list[0])):
    #     plt.plot(r_pieces_list, t_list[:,i])
    # plt.legend(['plain','native','direct','adapt'])
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()