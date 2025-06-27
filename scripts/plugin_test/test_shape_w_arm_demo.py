import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from time import time
import adapteddlo_muj.utils.transform_utils as T
# from adapteddlo_muj.envs.rnrvalid import ValidRnREnv
# from adapteddlo_muj.envs.rnrvalid2 import ValidRnR2Env
from adapteddlo_muj.envs.rnrvalid3_plugin import ValidRnR3Env
from adapteddlo_muj.utils.argparse_utils import tswa_parse

# from adapteddlo_muj.utils.transform_utils import IDENTITY_QUATERNION

# init_pos = np.array([0.35, 0., 0.3])
# init_quat = np.array([0.,1.,0.,0.])
# init_qpos = np.array([
    # 1.23530872e-08,  5.05769873e-01,
    # 1.92988983e+00, -1.04436514e-07,
    # 7.05933183e-01,  3.14159125e+00
# ])
"""
DATA:   (A-adapted, X-direct, N-native)
white (last number is alpha row is the diff_pos)
-A- alpha = 6.047162706224905e-05 (stiff_scale = 0.015) - 0.007771766311904944
-A- beta = 7.438010128656632e-05 (b_a = 1.23)
-X- alpha = 6.047162706224905e-05 (stiff_scale = 0.015) - 0.007679568747663055
-X- beta = 7.438010128656632e-05 (b_a = 1.23)
-N- alpha = 7.619425009843381e-05 (stiff_scale = 0.0189) - 0.0068583062598774405
-N- beta = 7.771813510040248e-05 (b_a = 1.02)

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

getting_jointpos = False

parser = tswa_parse()
args = parser.parse_args()
stest_types = args.stiff
wire_colors = args.wirecolor
move_id = args.moveid
do_render = args.render
# note: for red3native, must reset vel every 100 steps to ensure final pos reachable (due to robot control)

if stest_types is None:
    stest_types = ['adapt','native','adapt2']
else:
    stest_types = [stest_types]
if wire_colors is None:
    wire_colors = ['black','red','white']
else:
    wire_colors = [wire_colors]

lopbal_dict = dict(
    native="cable",
    adapt="wire_qst",
    adapt2="wire"
)

n_testtypes = len(stest_types)
n_wirecolors = len(wire_colors)
r_len = 0.40
model_type2id = dict(
    adapt=0,
    xfrc=1,
    native=2,
)
# alpha_glob_arr 
# -- rows=model_type(adapt,xfrc,native)
# -- columns=wire_type(white,black,red)
# alpha_glob_arr = np.array([
    # [6.047162706224905e-05, 0.0009997975674291843, 0.00063696780505569],
    # [6.047162706224905e-05, 0.0009997975674291843, 0.00063696780505569],
    # [7.619425009843381e-05, 0.0010643006362955833, 0.0006732507812930395],
# ])
# b_a_glob_arr = np.array([
    # [1.23,1.10,1.89],
    # [1.23,1.10,1.89],
    # [1.02,1.05,1.80],
# ])
# alpha_glob_arr = np.array([
#     [6.547162706224905e-05, 0.0009997975674291843, 0.00063696780505569],
#     [6.547162706224905e-05, 0.0009997975674291843, 0.00063696780505569],
#     [7.619425009843381e-05, 0.0010643006362955833, 0.0006732507812930395],
# ])
# b_a_glob_arr = np.array([
#     [1.05,1.10,1.89],
#     [1.05,1.10,1.89],
#     [0.85,1.05,1.80],
# ])
# alpha_glob_arr = np.array([
    # [5.644018525809911e-05, 0.0009997975674291843, 0.00063696780505569],
    # [5.644018525809911e-05, 0.0009997975674291843, 0.00063696780505569],
    # [7.256595247469885e-05, 0.0010643006362955833, 0.0006732507812930395],
# ])
# b_a_glob_arr = np.array([
#     [0.68,1.10,1.89],
#     [0.68,1.10,1.89],
#     [0.71,1.05,1.80],
# ])

init_pos=np.array([0.385,0.0,0.3])
init_quat=np.array([0.707,0.,0.707,0.])
init_qpos = np.array([
    7.40857786e-07,  6.38881793e-01,
    2.27269786e+00, -3.14160763e+00,
    1.34089981e+00,  3.14161037e+00
])

move_pos = np.array([
    [0.20,-0.1,-0.15],
    [0.095,0.,0.],#r
    [0.2,0.1,0.1],
    [0.2,0.1,0.1],#b
])
move_pos[:,0] -= 0.035
move_aa = np.array([
    [30.,-40.,0.],
    [0.,0.,0.],#r
    [50.,10.,0.],
    [50.,20.,-10.],#b
])
z_rot = np.array([
    360.0,
    -720,
    360,
    -720,#b
])

if move_id is not None:
    move_pos = np.array([move_pos[move_id]])
    move_aa = np.array([move_aa[move_id]])
    z_rot = np.array([z_rot[move_id]])

# new one with pos_id = 2 and z_rot = -720 
z_rot *= np.pi/180
move_aa *= np.pi/180
move_quat = np.zeros((len(move_aa),4))
for i in range(len(move_aa)):
    move_quat[i] = T.axisangle2quat(move_aa[i])
n_pos = len(move_pos)

piece_multi = 5
r_pieces = piece_multi*10

class manip_rope_seq:
    def __init__(
        self,
        stest_type='adapt',
        wire_color='white',
        overall_rot=None,
    ):
        # self.env = gym.make("adapteddlo_muj:Test-v0")
        # self.env = FlingLRRLRandEnv()
        stest_name = stest_type
        if stest_type == 'adapt2':
            stest_name = 'adapt'
        stiff_picklename = wire_color + '_' + stest_name + '_stiff.pickle'
        stiff_picklename = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "adapteddlo_muj/data/dlo_muj_real/stiff_vals/" + stiff_picklename
        )
        with open(stiff_picklename, 'rb') as f:
            stiff_pickle = pickle.load(f)
        alpha_glob, b_a_glob = stiff_pickle

        # alpha_glob = 0.001196450659614982
        # b_a_glob = 1.45

        # alpha_glob = 0.0012514302099516786
        # b_a_glob = 1.39

        beta_glob = b_a_glob * alpha_glob

        print(f'alpha_glob = {alpha_glob}')
        print(f'beta_glob = {beta_glob}')

        if wire_color == 'white':
            massperlen = 0.087/5.0
            rgba_vals = np.concatenate((np.array([300,300,300])/300,[1]))
            # alpha_glob = alpha_glob_arr[model_type2id[stest_type],0]
            # beta_glob = alpha_glob * b_a_glob_arr[model_type2id[stest_type],0]
        elif wire_color == 'black':
            rgba_vals = np.concatenate((np.array([0,0,0])/300,[1]))
            massperlen = 0.079/2.98
            # alpha_glob = alpha_glob_arr[model_type2id[stest_type],1]
            # beta_glob = alpha_glob * b_a_glob_arr[model_type2id[stest_type],1]
        elif wire_color == 'red':
            rgba_vals = np.concatenate((np.array([300,0,0])/300,[1]))
            massperlen = 0.043/2.0
            # alpha_glob = alpha_glob_arr[model_type2id[stest_type],2]
            # beta_glob = alpha_glob * b_a_glob_arr[model_type2id[stest_type],2]
        plugin_name = lopbal_dict[stest_type]
        self.env = ValidRnR3Env(
            alpha_bar=alpha_glob,
            beta_bar=beta_glob,
            r_len=r_len,
            r_mass=r_len*massperlen,
            r_thickness=0.006,
            r_pieces=r_pieces,
            overall_rot=overall_rot,
            plugin_name=plugin_name,
            rgba_vals=rgba_vals,
            do_render=do_render
        )

node_pos_arr = np.zeros((n_testtypes,n_wirecolors,n_pos,r_pieces+1,3))
joint_pos_arr = np.zeros((n_testtypes,n_wirecolors,n_pos,6))
for i in range(n_testtypes):
    for j in range(n_wirecolors):
        for pos_id in range(n_pos):
            if move_id is None:
                i_move = pos_id
            else:
                i_move = move_id
            if wire_colors[j] == 'red' and i_move == 2:
                velreset = True
            else:
                velreset = False
            
            print(f"Now computing: {wire_colors[j]}{i_move}_{stest_types[i]}")
            if stest_types[i] == 'native':
                overall_rot = None
            else:
                # overall_rot = z_rot[pos_id]
                overall_rot = None
                # overall_rot = -3*np.pi
            env1 = manip_rope_seq(
                stest_type=stest_types[i],
                wire_color=wire_colors[j],
                overall_rot=overall_rot,
            )
            env1.env.velreset = velreset
            if getting_jointpos:
                env1.env.max_action = 0.02
            desired_pos = env1.env.init_pos + move_pos[pos_id]
            desired_quat = T.quat_multiply(env1.env.init_quat,move_quat[pos_id])
            env1.env.max_action = 0.02
            ## Move
            # env1.env.hold_pos(5.)
            env1.env.move_to_pose(
                desired_pos,
                desired_quat
                # env1.env.init_pos,
                # env1.env.init_quat
                # env1.env.init_pos,
                # env1.env.init_quat
            )
            # env1.env.step()
            # if stest_types[i] == 'native':
                # env1.env.rot_x_rads(z_rot[pos_id])
            if not getting_jointpos:
                # comment out when obtaining joint pos
                # if stest_types[i] == 'native':
                env1.env.rot_x_rads(z_rot[pos_id])
                env1.env.move_to_pose(
                    desired_pos,
                    desired_quat
                    # init_pos,
                    # init_quat
                    # env1.env.init_pos,
                    # env1.env.init_quat
                )
                print("holding_pos")
                env1.env.hold_pos(10.)
                print("held_pos")
            
            while True:
                j_pos = env1.env._jd.copy()
                if do_render:
                    env1.env.viewer._paused = True
                    env1.env.hold_pos(1.)
                # env1.env.hold_pos(10.)
    
                # env1.runabit()
                # j_pos = env1.env._jd.copy()
                joint_pos_arr[i,j,pos_id] = j_pos
                # print(repr(env1.env._jd))
                
                rob_qpos = env1.env.observations['qpos']
                # nodes_pos = np.zeros((11,3))
                # node_id = np.array(np.array(range(11))*piece_multi,dtype='int')
                # nodes_pos = env1.env.observations['rope_pose'][node_id]
                nodes_pos = env1.env.observations['rope_pose']
                node_pos_arr[i,j,pos_id] = nodes_pos
                # print(nodes_pos)
    
                simdata_picklename = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "adapteddlo_muj/data/simdata/plugin/" + f"simdata_{wire_colors[j]}{i_move}_{stest_types[i]}.pickle"
                )
                pickle_simdata = [init_qpos, z_rot[pos_id], nodes_pos,j_pos]
                with open(simdata_picklename, 'wb') as f:
                    pickle.dump(pickle_simdata,f)

            # excel_dir = os.path.join(
            #     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            #     "adapteddlo_muj/data/excel/"
            # )

            # np.savetxt(excel_dir + f"jointpos{i_move}.csv",j_pos,delimiter=",")
            # np.savetxt(excel_dir + "initqpos.csv",init_qpos,delimiter=",")
            # np.savetxt(excel_dir + f"z_rot{i_move}.csv",[z_rot[pos_id]],delimiter=",")

            # print(f"progress = {pos_id+1 + j*n_pos + i*n_pos*n_wirecolors}/{n_testtypes*n_wirecolors*n_pos}")

if move_id is None:
    move_id = 999
# init_qpos = env1.env.init_qpos.copy()
print('sim test csvdata saved!')