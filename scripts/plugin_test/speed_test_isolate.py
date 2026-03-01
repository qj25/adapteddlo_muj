"""
Speed test that records stiffness/plugin computation time split into total, applyFT, and rest.
Modes: native (Cable), xfrc (Python convert_and_update_force_timed), adapted (wire_qst), jpq-DER (wire).
No plain mode. Plot: one color per mode, 3 lines per mode (total, applyFT, rest); no percentage.
"""

import os
import numpy as np
import pickle

from adapteddlo_muj.envs.native_cable_valid_test import TestCableEnv
from adapteddlo_muj.envs.our_xfrc_rope_valid_test import TestRopeXfrcEnv
from adapteddlo_muj.envs.validitytest_env import TestPluginEnv

from adapteddlo_muj.utils.argparse_utils import spdt_parse
from adapteddlo_muj.utils.plotter import plot_isolate_timing_split

# ======================| Settings |======================
parser = spdt_parse()
args = parser.parse_args()

new_start = bool(args.newstart)

test_type = 'speedtest2'

r_len = 9.29
r_thickness = 0.03
alpha_val = 1.345
beta_val = 0.789
if test_type == 'speedtest1':
    r_pieces_list = [20, 30, 40, 50, 60]
else:
    r_pieces_list = [40, 60, 80, 110, 140, 180]

# Pickle path: (r_pieces_list, t_total, t_applyFT, t_rest) each shape (len(r_pieces_list), 4)
speedtest_isolate_all_picklename = f'{test_type}_isolate_res_all.pickle'
speedtest_isolate_all_picklename = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "adapteddlo_muj/data/speed_test/" + speedtest_isolate_all_picklename
)
# ======================| End Settings |======================

MODE_LABELS = ['native', 'xfrc', 'adapted', 'jpQ-DER']
N_MODES = 4

if new_start:
    t_total = np.zeros((len(r_pieces_list), N_MODES))
    t_applyFT = np.zeros((len(r_pieces_list), N_MODES))
    t_rest = np.zeros((len(r_pieces_list), N_MODES))

    for i in range(len(r_pieces_list)):
        print(f"Testing speed (isolate) for {r_pieces_list[i]} pieces.. ..")

        # native: Cable plugin
        print("native:")
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
        out = env_native.run_speedtest2_isolate()
        t_total[i, 0] = out["total"]
        t_applyFT[i, 0] = out["applyFT"]
        t_rest[i, 0] = out["rest"]

        # xfrc: Python convert_and_update_force_timed
        print("xfrc:")
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
        out = env_xfrc.run_speedtest2_isolate()
        t_total[i, 1] = out["total"]
        t_applyFT[i, 1] = out["applyFT"]
        t_rest[i, 1] = out["rest"]

        # adapted: wire_qst plugin
        print("adapted:")
        env_adapted = TestPluginEnv(
            overall_rot=0.0,
            do_render=False,
            r_pieces=r_pieces_list[i],
            r_len=r_len,
            r_thickness=r_thickness,
            test_type=test_type,
            alpha_bar=alpha_val,
            beta_bar=beta_val,
            plugin_name="wire_qst",
        )
        out = env_adapted.run_speedtest2_isolate()
        t_total[i, 2] = out["total"]
        t_applyFT[i, 2] = out["applyFT"]
        t_rest[i, 2] = out["rest"]

        # jpq-DER: wire plugin
        print("jpQ-DER:")
        env_jpq = TestPluginEnv(
            overall_rot=0.0,
            do_render=False,
            r_pieces=r_pieces_list[i],
            r_len=r_len,
            r_thickness=r_thickness,
            test_type=test_type,
            alpha_bar=alpha_val,
            beta_bar=beta_val,
            plugin_name="wire",
        )
        out = env_jpq.run_speedtest2_isolate()
        t_total[i, 3] = out["total"]
        t_applyFT[i, 3] = out["applyFT"]
        t_rest[i, 3] = out["rest"]

    speedtest_data = [r_pieces_list, t_total, t_applyFT, t_rest]
    input("Saving pickle. Press 'Enter' to confirm.. ..")
    with open(speedtest_isolate_all_picklename, 'wb') as f:
        pickle.dump(speedtest_data, f)
    print("Pickle saved!")

else:
    with open(speedtest_isolate_all_picklename, 'rb') as f:
        data = pickle.load(f)
    r_pieces_list, t_total, t_applyFT, t_rest = data

    print("Pickle loaded!")

plot_isolate_timing_split(
    r_pieces_list,
    t_total,
    t_applyFT,
    t_rest,
    mode_labels=MODE_LABELS,
)
