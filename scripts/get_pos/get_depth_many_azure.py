import os
import numpy as np
import pickle
from adapteddlo_muj.utils.finddepth import locatepointin3d, rescale_points
from adapteddlo_muj.utils.plotter import plot3d

wire_colors = ['black','red','white']
pos_type = ['0','1','2','3']

r_len = 0.4
r_pieces = 10
# wire_colors = ['white']
# pos_type = ['3']
    
def get_3dpoints(data_name):
    # Read the CSV file into a numpy array
    data_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "adapteddlo_muj/data/data3d/"
    )
    data_file = data_file + data_name + '.csv'
    # data = np.genfromtxt(data_file, delimiter=',', dtype='double')
    data = np.genfromtxt(data_file, delimiter=',')
    
    # Print the numpy array
    # print("Data from CSV as numpy array:")
    # print(data)
    # for i in range(len(data)):
        # print(f'Point {i}: {data[i]}')
    # input()
    xyz_posall = np.zeros_like(data)
    for data_id in range(len(data)):
        xyz_posall[data_id] = locatepointin3d(data[data_id],
            img_size=np.array([640,576],dtype='int')
        )
    return xyz_posall

pt_arr = np.zeros((len(wire_colors),len(pos_type),r_pieces+1,3))
for i in range(len(wire_colors)):
    for j in range(len(pos_type)):
        data_name = wire_colors[i] + pos_type[j]
        xyz_posall = get_3dpoints(data_name)
        total_len = np.sum(np.linalg.norm(np.diff(xyz_posall,axis=0),axis=1))
        print(f"wc={i}")
        print(f"pt={j}")
        print(f"total_len = {total_len}")
        # print(f"xyz_posall = {xyz_posall}")
        pt_arr[i,j] = rescale_points(xyz_posall,r_len=r_len,r_pieces=r_pieces)
        # input(pt_arr[i,j])
        # plot3d(pt_arr[i,j])

data_file = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "adapteddlo_muj/data/"
)
data_file = data_file + 'pts_all.pickle'
with open(data_file, 'wb') as f:
    pickle.dump(pt_arr,f)
