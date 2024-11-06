import numpy as np

def compute_PCA(d1, d2, d3):
    n_data = len(d1)
    d123 = np.concatenate((
        [d1],
        [d2],
        [d3]
    ))


    d123_mean = np.mean(d123, axis=1, keepdims=True)
    svd = np.linalg.svd(d123 - np.mean(d123, axis=1, keepdims=True))
    left = svd[0]
    # print(left[:,-1])



    # print(d123)
    # d123_mean = np.zeros(3)
    # # normalizing and standardizing var
    # for i in range(3):
    #     d123_mean[i] = np.mean(d123[i])
    #     d123[i] -= d123_mean[i]
    #     if np.sum(np.square(d123[i])) != 0:
    #         d123[i] /= np.sqrt(np.sum(np.square(d123[i]))/(n_data-1))

    # covar_mat = np.dot(d123, np.transpose(d123)) / (n_data-1)
    # print(d123)
    # print(covar_mat)
    # eig_val, eig_vec = np.linalg.eig(covar_mat)
    # id_min = min(range(len(eig_val)), key=eig_val.__getitem__)
    # plane_norm = eig_vec[:,id_min]
    plane_norm = left[:,-1]
    plane_norm /= np.linalg.norm(plane_norm)
    plane_const = np.dot(plane_norm, d123_mean)
    # use formula for plane dist to calc dist norm of all points
    e_tot = 0.
    for i in range(n_data):
        e_tot += abs(np.dot(plane_norm, d123[:,i]) - plane_const)
        # print(abs(np.dot(plane_norm, d123[:,i]) - plane_const))
    # print(plane_norm)
    # plane_norm /= np.linalg.norm(plane_norm)
    # print(np.linalg.norm(plane_norm))
    return e_tot

def centralize_devdata(x_data, dev_data):
    n_graddata = 6 # MUST be even
    center_id = np.argmax(dev_data)
    if dev_data[center_id+1] > dev_data[center_id-1]:
        center0_id = -1 # away from the int(n_graddata/2)
    else:
        center0_id = 0
    id_start = center_id - (int(n_graddata/2) + center0_id)
    grad_data = np.zeros(n_graddata)
    # for i in range(n_graddata):
        # print(dev_data[i+id_start])
        # print(x_data[i+id_start])
        # print()
        
    for i in range((int(n_graddata/2)-1)):
        # print((dev_data[id_start+i+1] - dev_data[id_start+i]))
        # print((x_data[id_start+i+1] - x_data[id_start+i]))
        grad_data[i] = (
            (dev_data[id_start+i+1] - dev_data[id_start+i])
            / (x_data[id_start+i+1] - x_data[id_start+i])
        )
        grad_data[n_graddata-1-i] = - (
            (dev_data[id_start+n_graddata-1-i-1] - dev_data[id_start+n_graddata-1-i])
            / (x_data[id_start+n_graddata-1-i-1] - x_data[id_start+n_graddata-1-i])
        )
    if center0_id < 0:
        id_ends = np.array([0,-1,-2])
        id_raw_ends = id_start+n_graddata + id_ends
        id_raw_ends[0] = id_start + id_ends[0]
    else:
        id_ends = np.array([-1,0,1])
        id_raw_ends = id_start + id_ends
        id_raw_ends[0] = id_start+n_graddata + id_ends[0]
    new_x1 = (
        x_data[id_raw_ends[1]]
        - (grad_data[id_ends[1]]-grad_data[id_ends[0]]) 
        / (grad_data[id_ends[1]]-grad_data[id_ends[2]])
        * (x_data[id_raw_ends[1]]-x_data[id_raw_ends[2]])
    )
    # print(center_id)
    # print(id_start)
    # print('new')
    # print(id_raw_ends)
    # print(x_data[id_raw_ends[0]])
    # print(x_data[id_raw_ends[1]])
    # print(x_data[id_raw_ends[2]])
    new_xmid = (new_x1 + x_data[id_raw_ends[0]]) / 2.0
    # print('test')
    # print(new_x1)
    # print(new_xmid)
    x_data -= new_xmid
    n_shootsteps = 10
    shootstep_len = np.min(np.abs(x_data)) / n_shootsteps
    new_dev = dev_data[center_id]
    if center0_id < 0:
        grad_step = grad_data[(int(n_graddata/2)-2)]
    else:
        grad_step = grad_data[(int(n_graddata/2)+1)]
    for i in range(n_shootsteps):
        new_dev = (
            grad_step*(n_shootsteps-i)/n_shootsteps
            * shootstep_len
            + new_dev 
        )
        # print(grad_step)
        # print(shootstep_len)
        # input(new_dev)
    dev_data = np.insert(dev_data,center_id-center0_id,new_dev)
    x_data = np.insert(x_data,center_id-center0_id,0.0)

    # grad_data[2] = grad_data[1]-grad_data[0] + grad_data[1]
    # grad_data[3] = grad_data[4]-grad_data[5] + grad_data[4]
    # print(grad_data)
    # print(dev_data)
    # print(x_data)
    # input()

    return x_data, dev_data

if __name__ == "__main__":
    d1 = [0., 1., 2.]
    d2 = [0., 2., 0.]
    d3 = [0., 0., 1.]
    print(compute_PCA(d1, d2, d3))
    # t1 = (np.cross([1,2,0],[2,0,1]))
    # mag_1 = (np.linalg.norm(t1))
    # t1 = t1 / mag_1
    # print(t1)
    # print(np.linalg.norm(t1))