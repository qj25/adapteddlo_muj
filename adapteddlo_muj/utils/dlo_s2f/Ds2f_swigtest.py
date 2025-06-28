import Dlo_s2f
import numpy as np
from time import time

# # define init variables
# n_test = 1000
# n_nodes = 5
# r_len = 1.

# theta_n = 0.
# overall_rot = 0.

# sec_len = r_len / (n_nodes - 1)
# node_pos = np.zeros((n_nodes,3))
# node_force = np.zeros((n_nodes,3))
# bf0sim = np.zeros((3,3))

# for j in range(3):
#     bf0sim[j,j] = 1.
# for i in range(n_nodes):
#     node_pos[i,0] = sec_len*i

# node_pos[2,1] = -0.01

# print(bf0sim)
# print(node_pos)
# print("Entering C++ ...")

# start_t = time()
# for i in range(n_test):
#     # flatten np array
#     bf0sim = bf0sim.flatten()
#     node_pos = node_pos.flatten()
#     node_force = node_force.flatten()

#     dt1 = Der_iso3.DER_iso3(node_pos, bf0sim, 0., 0.)

#     dt1.updateVars(node_pos, bf0sim, bf0sim)
#     dt1.updateTheta(0.)
#     dt1.calculateCenterlineF2(node_force)

#     node_force = node_force.reshape((n_nodes,3))
# end_t = time()
# print(f"time taken = {end_t - start_t}")

# print(node_force)
