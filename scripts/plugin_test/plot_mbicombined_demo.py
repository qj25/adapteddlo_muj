"""
To-do:
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from adapteddlo_muj.envs.our_rope_valid_test import TestRopeEnv
from adapteddlo_muj.envs.our_xfrc_rope_valid_test import TestRopeXfrcEnv
from adapteddlo_muj.envs.native_cable_valid_test import TestCableEnv

base_plotted = False
plt.rcParams.update({'pdf.fonttype': 42})
# plt.style.use('ggplot')
plt.style.use('seaborn-v0_8')
fig_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "adapteddlo_muj/data/figs/plgn/" + 'mbi_combined.pdf'
)

def mbi_plot(b_a, theta_crit, c='k'):
    b_a_base = b_a.copy()
    theta_crit_base = 2*np.pi*np.sqrt(3)/(b_a_base)

    max_devi_theta_crit = np.max(np.abs(theta_crit_base - theta_crit))
    avg_deviation = np.linalg.norm(theta_crit_base - theta_crit) / len(theta_crit)
    print(f"max_devi_theta_crit = {max_devi_theta_crit}")
    print(f"avg_deviation = {avg_deviation}")

    if not base_plotted:
        plt.plot(b_a_base, theta_crit_base, c='k', linewidth="2", alpha=0.5, zorder=5)
    else:
        next(iter(plt.rcParams['axes.prop_cycle']))
    plt.plot(b_a, theta_crit, alpha=0.7, color=c)
    

lopbal_type_list = ['wire','wire_qst','native']
legend_type_list = ['jpQ-DER','adapted','native']
# c_list = ['steelblue', 'sienna']
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
c_list = [
    color_cycle[1],
    color_cycle[0],
    color_cycle[2],
]
# input(color_cycle)

fig = plt.figure("Michell's Buckling Instability", figsize=(6,4))
ax = fig.add_subplot(111)
ax.set_xlabel(r"$\beta/\alpha$")
ax.set_ylabel(r'$\theta^n$ (rad)')
legend_str = []
legend_str.append('analytical')

for i in range(len(lopbal_type_list)):
    print(f"For {lopbal_type_list[i]}:")
    if lopbal_type_list[i] != 'native':
        mbi_picklename = '/adapt/plgn/' + lopbal_type_list[i] + '/mbi1.pickle'
    else:
        mbi_picklename = lopbal_type_list[i] + '/plgn/cable/mbi1.pickle'
    mbi_picklename = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "adapteddlo_muj/data/mbi/" + mbi_picklename
    )
    print('Loading MBI test...')
    with open(mbi_picklename, 'rb') as f:
        pickle_mbidata = pickle.load(f)
    idhalf_pickle = round(len(pickle_mbidata)/2)
    b_a = pickle_mbidata[:idhalf_pickle]
    theta_crit = pickle_mbidata[idhalf_pickle:]
    mbi_plot(b_a=b_a, theta_crit=theta_crit, c=c_list[i])
    legend_str.append(legend_type_list[i])
    base_plotted = True

ax.legend(legend_str)
plt.grid(True)
plt.tight_layout()
plt.savefig(fig_dir,bbox_inches='tight')
plt.show()
