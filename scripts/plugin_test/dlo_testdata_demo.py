"""
NOTE: This version is not ready for use. Requires unreleased wire.cc plugin.
To-do:
"""

import os
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt

from adapteddlo_muj.envs.our_rope_valid_test import TestRopeEnv
from adapteddlo_muj.envs.our_xfrc_rope_valid_test import TestRopeXfrcEnv
from adapteddlo_muj.envs.native_cable_valid_test import TestCableEnv
from adapteddlo_muj.envs.validitytest_env import TestPluginEnv
from adapteddlo_muj.utils.argparse_utils import dtd_parse

#======================| Settings |======================

lopbal_dict = dict(
    native="cable",
    adapt="wire_qst",
    adapt2="wire",
    xfrc="xfrc"
)

parser = dtd_parse()
# Parse the arguments
args = parser.parse_args()

lopbal_type = args.stiff
test_type_g = args.test
do_render_g = bool(int(args.render))
new_start_g = bool(int(args.newstart))
lfp_g = bool(args.loadresults)

plugin_name = lopbal_dict[lopbal_type]
lopbal_name = lopbal_type
if lopbal_name == 'adapt2':
    lopbal_name = 'adapt'
plugin_datafolder = "plgn/" + plugin_name

if test_type_g == 'mbi' and args.newstart == 2:
    new_start_g = False

# if test_type_g == 'lhb' and lopbal_type == 'native':
    # new_start_g = True
#======================| End Settings |======================

def lhb_data(
    alpha_bar,
    beta_bar,
    s,
    m,
    t,
    e_x,
):
    # for analytical results, tanh^2(s_ss)
    phi = np.arccos(np.dot(t,e_x))
    phi_0 = max(phi)
    dal = (
        beta_bar * m / (2. * alpha_bar)
        * np.sqrt((1 - np.cos(phi_0))/(1 + np.cos(phi_0)))
    ) * s
    
    func_phi = (
        (np.cos(phi) - np.cos(phi_0))
        / (1 - np.cos(phi_0))
    )
    
    return dal, func_phi

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

    plt.figure(f"Michell's Buckling Instability for {plugin_name}")
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
    alpha_val=1.,
    beta_val=1.,
    do_render=False,
    new_start=False,
):
    r_pieces = 51
    r_len = 2*np.pi * r_pieces / (r_pieces-1)
    r_thickness = 0.05
    r_mass = r_len * 2.
    r_mass = r_len
    if lopbal_type == 'native':
        env = TestPluginEnv(
            overall_rot=overall_rot,
            do_render=do_render,
            r_pieces=r_pieces,
            r_len=r_len,
            r_thickness=r_thickness,
            test_type='mbi',
            alpha_bar=alpha_val,
            beta_bar=beta_val,
            # r_mass=r_mass,
            new_start=new_start,
            plugin_name=plugin_name
        )
    elif lopbal_type == 'xfrc':
        env = TestRopeXfrcEnv(
            overall_rot=overall_rot,
            do_render=do_render,
            r_pieces=r_pieces,
            r_len=r_len,
            r_thickness=r_thickness,
            test_type='mbi',
            alpha_bar=alpha_val,
            beta_bar=beta_val,
            # r_mass=r_mass,
            new_start=new_start,
        )
    else:
        env = TestPluginEnv(
            overall_rot=overall_rot,
            do_render=do_render,
            r_pieces=r_pieces,
            r_len=r_len,
            r_thickness=r_thickness,
            test_type='mbi',
            alpha_bar=alpha_val,
            beta_bar=beta_val,
            # r_mass=r_mass,
            new_start=new_start,
            plugin_name=plugin_name
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

def mbi_test(new_start=False, load_from_pickle=False, do_render=False):
    mbi_picklename = lopbal_name + '/' + plugin_datafolder + '/mbi1.pickle'
    mbi_picklename = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "adapteddlo_muj/data/mbi/" + mbi_picklename
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
        beta_bar_step = (beta_bar_lim[1] - beta_bar_lim[0]) / (n_data_mbi - 1)
        beta_bar = np.zeros(n_data_mbi)
        alpha_bar = np.zeros(n_data_mbi)
        for i in range(n_data_mbi):
            beta_bar[i] = beta_bar_step * i + beta_bar_lim[0]
            alpha_bar[i] = 1.
        beta_bar = beta_bar[::-1]
        b_a = beta_bar / alpha_bar
        theta_crit = np.zeros(n_data_mbi)
        if new_start:
            theta_crit[i] = mbi_indivtest(
                alpha_val=alpha_bar[i],
                beta_val=beta_bar[i],
                new_start=new_start,
                do_render=do_render
            )
        # overall_rot = 9.0
        overall_rot = 8.5
        # overall_rot = 0.0
        for i in range(n_data_mbi):
            # overall_rot = 7.5
            print(f'b_a = {b_a[i]}')
            while theta_crit[i] < 1e-7:
                print(f'overall_rot = {overall_rot}')
                theta_crit[i] = mbi_indivtest(
                    overall_rot=overall_rot,
                    alpha_val=alpha_bar[i],
                    beta_val=beta_bar[i],
                    do_render=do_render
                )
                overall_rot += 0.1 # * (np.pi/180)
        pickle_mbidata = np.concatenate((b_a,theta_crit))
        with open(mbi_picklename, 'wb') as f:
            pickle.dump(pickle_mbidata,f)
            print('mbi test data saved!')
    
    mbi_plot(b_a=b_a, theta_crit=theta_crit)

def _pickle2data(ax, r_pieces):
    pickledata_path = lopbal_name + '/' + plugin_datafolder + '/lhb{}.pickle'.format(r_pieces)
    pickledata_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "adapteddlo_muj/data/lhb/" + pickledata_path
    )
    # input(pickledata_path)
    with open(pickledata_path, 'rb') as f:
        pickledata = pickle.load(f)
    ## Using 18.26 as m/l = 2*np.pi*27/9.29
    # const_denom = 2.0*np.pi*27.0/9.238388888888865
    # const_denom = 2.0*np.pi*27.0/9.223642857142869
    # const_denom = 2.0*np.pi*27.0/9.29
    # const_denom = 18.26
    # s_ss_base2 = pickledata[1]/12*const_denom
    s_ss_base2 = pickledata[1].copy() # /12*const_denom
    fphi_base2 = (np.tanh(s_ss_base2))**2.
    id_counter = 0

    # print(s_ss_base2)
    while id_counter < len(s_ss_base2):
        # print(id_counter)
        # print(s_ss_base2[id_counter])
        # print(s_ss_base2[id_counter] <= 6)
        # print(s_ss_base2[id_counter] > 6)
        if s_ss_base2[id_counter] <= -6 or s_ss_base2[id_counter] > 6:
            s_ss_base2 = np.delete(s_ss_base2, id_counter, axis=0)
            fphi_base2 = np.delete(fphi_base2, id_counter, axis=0)
            pickledata[0] = np.delete(pickledata[0], id_counter, axis=0)
            # print(s_ss_base2)
        else:
            id_counter += 1

    print(f'r_pieces = {r_pieces}')
    print(f'fphi = {pickledata[0]}')
    print(f's_ss = {s_ss_base2}')

    avg_devi_lhb = np.linalg.norm(pickledata[0] - fphi_base2) / len(fphi_base2)
    print(f"Average deviation for {r_pieces} pieces = {avg_devi_lhb}")
    s_ss_base2 = np.insert(s_ss_base2, 0, -6.)
    fphi_base2 = np.insert(fphi_base2, 0, 1.)
    pickledata[0] = np.insert(pickledata[0], 0, 1.)
    s_ss_base2 = np.insert(s_ss_base2, len(s_ss_base2), 6.)
    fphi_base2 = np.insert(fphi_base2, len(fphi_base2), 1.)
    pickledata[0] = np.insert(pickledata[0], len(pickledata[0]), 1.)
    ax.plot(s_ss_base2, pickledata[0], alpha=0.5)
    # input()

def lhb_plot(r_pieces_list):
    s_ss_base = np.arange(-6., 6., 0.01)
    fphi_base = (np.tanh(s_ss_base))**2.
    plt.rcParams.update({'pdf.fonttype': 42})
    # plt.style.use('ggplot')
    plt.style.use('seaborn-v0_8')
    fig_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "adapteddlo_muj/data/figs/" + plugin_datafolder + '/' + 'lhb_adapted.pdf'
    )
    fig = plt.figure("Localized Helical Buckling for " + plugin_name, figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.set_xlabel("s/s*")
    ax.set_ylabel(r'$f(\varphi)$')
    ax.plot(s_ss_base, fphi_base, color='k', linewidth="2", alpha=0.5)
    
    ax.spines['left'].set_position(('data',0))
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.labelpad = 0.

    legend_str = []
    legend_str.append('Analytical')
    for r_pieces in r_pieces_list:
        _pickle2data(ax, r_pieces=r_pieces)
        legend_str.append(str(r_pieces))

    ax.legend(legend_str)
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_dir,bbox_inches='tight')
    plt.show()

def lhb_indivtest(
    r_pieces=20,
    do_render=False,
    new_start=False
):
    alpha_val = 1.345
    beta_val = 0.789
    overall_rot = 27. * (2*np.pi)
    r_thickness = 0.04
    # overall_rot = 0.0
    r_len = 9.29
    # r_mass = r_len/5
    if lopbal_type == 'native':
        env = TestPluginEnv(
            do_render=do_render,
            r_pieces=r_pieces,
            r_len=r_len,
            r_thickness=r_thickness,
            test_type='lhb',
            overall_rot=overall_rot,
            alpha_bar=alpha_val,
            beta_bar=beta_val,
            # r_mass=r_mass,
            new_start=new_start,
            plugin_name=plugin_name
        )
    elif lopbal_type == 'xfrc':
        env = TestRopeXfrcEnv(
            do_render=do_render,
            r_pieces=r_pieces,
            r_len=r_len,
            r_thickness=r_thickness,
            test_type='lhb',
            overall_rot=overall_rot,
            alpha_bar=alpha_val,
            beta_bar=beta_val,
            # r_mass=r_mass,
            new_start=new_start,
            limit_f=True,
        )
    else:
        env = TestPluginEnv(
            do_render=do_render,
            r_pieces=r_pieces,
            r_len=r_len,
            r_thickness=r_thickness,
            test_type='lhb',
            overall_rot=overall_rot,
            alpha_bar=alpha_val,
            beta_bar=beta_val,
            # r_mass=r_mass,
            new_start=new_start,
            plugin_name=plugin_name
        )
    # env.do_render = False
    # env.reset()
    # env.viewer._paused = True
    if do_render:
        env.viewer.close()
    env.close()

def lhb_test(new_start=True, load_from_pickle=False, do_render=False):
    print('Starting LHB test.')
    n_pieces = [40, 60, 80, 110, 140]
    n_pieces = [180]
    # n_pieces = [40]
    n_pieces = [40, 60, 80, 110, 140, 180]
    if not load_from_pickle:
        for i in n_pieces:
            print(f"LHB test for {i} pieces.. ..")
            lhb_indivtest(
                r_pieces=i,
                new_start=new_start,
                do_render=do_render    
            )
    lhb_plot(r_pieces_list=n_pieces)

if test_type_g == 'lhb':
    lhb_test(new_start=new_start_g, load_from_pickle=lfp_g, do_render=do_render_g)
if test_type_g == 'mbi':
    mbi_test(new_start=new_start_g, load_from_pickle=lfp_g, do_render=do_render_g)