import os
import pickle
import numpy as np
import adapteddlo_muj.utils.finddepth as fd1
from adapteddlo_muj.utils.plotter import plot3d, plot_bars
from adapteddlo_muj.utils.argparse_utils import svr_parse

parser = svr_parse()
args = parser.parse_args()

wc = args.wirecolor

test_types = ['adapt','native']
wire_colors = ['black','red','white']
pos_type = ['0','1','2','3']
n_testtypes = len(test_types)
n_wirecolors = len(wire_colors)
n_pos = len(pos_type)

# init_qpos = np.array([
    # 7.40857786e-07,  6.38881793e-01,
    # 2.27269786e+00, -3.14160763e+00,
    # 1.34089981e+00,  3.14161037e+00
# ])
data_file = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "adapteddlo_muj/data/"
)
data_file = data_file + 'pts_all.pickle'

with open(data_file, 'rb') as f:
    real_pos_arr_all = pickle.load(f)
    print('real pos data loaded!')
n_pieces = len(real_pos_arr_all[0][0])-1

loop_new = True
for i in range(n_testtypes):
    for j in range(n_wirecolors):
        for pos_id in range(n_pos):
            simdata_picklename = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "adapteddlo_muj/data/simdata/" + f"simdata_{wire_colors[j]}{pos_id}_{test_types[i]}.pickle"
            )
            with open(simdata_picklename, 'rb') as f:
                pickle_simdata = pickle.load(f)
                print('sim test data loaded!')
            [init_qpos, z_rot, node_pos_arr_indiv, joint_pos_arr] = pickle_simdata
            if loop_new:
                node_pos_arr = np.zeros((
                    n_testtypes,
                    n_wirecolors,
                    n_pos,
                    n_pieces+1,
                    3
                ))
                r_len = fd1.len_pts(node_pos_arr_indiv)
                loop_new = False
            node_pos_arr_indiv = fd1.split_lines2(node_pos_arr_indiv,n_pieces)
            node_pos_arr[i,j,pos_id,:,:] = node_pos_arr_indiv[:,:].copy()

# np.savetxt("jointposarr.csv",joint_pos_arr[0,0],delimiter=",")
# np.savetxt("initqpos.csv",init_qpos,delimiter=",")
# np.savetxt("z_rot.csv",z_rot,delimiter=",")
# print('sim test csvdata saved!')

real_pos_arr = real_pos_arr_all.copy()

n_points = len(node_pos_arr[0][0][0])
error_arr = np.zeros((n_testtypes,n_wirecolors,n_pos))
for i in range(n_testtypes):
    print(f"Test type: {test_types[i]}| =======================================")
    for j in range(n_wirecolors):
        # if wire_colors[j] != 'white': continue
        print(f"wirecolor: {wire_colors[j]}")
        for pos_id in range(n_pos):
            # if pos_id != 1: continue
            node_pos_arr[i,j,pos_id] -= node_pos_arr[i,j,pos_id,0]
            node_pos_arr[i,j,pos_id,:,0] *= -1.0
            node_pos_arr[i,j,pos_id,:,1] *= -1.0
            real_pos_arr[j,pos_id] -= real_pos_arr[j,pos_id,0]
            # print(node_pos_arr[i,j,pos_id])
            old_node_pos = real_pos_arr[j,pos_id].copy()
            # print()
            se_axis = node_pos_arr[i,j,pos_id][-1]-node_pos_arr[i,j,pos_id][0]
            if i < 1 or i > 0:
                real_pos_arr[j,pos_id] = fd1.adjust_linestandard(
                    points_arr=real_pos_arr[j,pos_id].copy(),
                    startend_axis=se_axis.copy(),
                )
                proxy_realpos = real_pos_arr[j,pos_id].copy()
                proxy_realpos = fd1.optimize_through_axisscale(
                    real_pos_arr[j,pos_id].copy(),
                    se_axis,
                    r_len
                )
                real_pos_arr[j,pos_id] = proxy_realpos.copy()
            # real_pos_arr[j,pos_id] = scale_linestandard(
                # points_arr=real_pos_arr[j,pos_id].copy(),
                # startend_axis=se_axis.copy(),
            # )

            error_arr[i,j,pos_id] = np.sum(np.linalg.norm(
                node_pos_arr[i,j,pos_id]-real_pos_arr[j,pos_id],axis=1
            ))/n_points

            print(f"pos{pos_id} error = {error_arr[i,j,pos_id]}")

            # plot3d(
            #     node_pos_arr[i,j,pos_id],
            #     [old_node_pos,real_pos_arr[j,pos_id]]
            # )

# posid_excl = [0,1,3]
for j in range(n_wirecolors):
    if wc is not None:
        if wire_colors[j] != wc:
            continue
    print(f"wirecolor: {wire_colors[j]}")
    for pos_id in range(n_pos):
        # if pos_id in posid_excl: continue
        err1 = np.sum(np.linalg.norm(
            node_pos_arr[0,j,pos_id]-node_pos_arr[1,j,pos_id],axis=1
        ))/n_points
        print(real_pos_arr[j,pos_id])
        print(f"pos{pos_id} error = {err1}")
        plot3d(real_pos_arr[j,pos_id],[node_pos_arr[0,j,pos_id],node_pos_arr[1,j,pos_id]])

# for j in range(n_wirecolors):
#     for pos_id in range(n_pos):
#         plot3d(real_pos_arr[j,pos_id],node_pos_arr[:,j,pos_id])
plot_bars(error_arr[:,[2,0,1]])