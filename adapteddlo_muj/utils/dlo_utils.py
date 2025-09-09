import numpy as np

def rot_vec(vec_0, u, a):
    # for rotation of 3d vector 
    # vec_0: vector
    # u: anchor vector (rotate about this vector)
    # a: angle in radians
    u = u / np.linalg.norm(u)
    R = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            if i == j:
                R[i,j] = (
                    np.cos(a)
                    + ((u[i]**2) * (1 - np.cos(a)))
                )
            else:
                ss = 1.
                if i < j:
                    ss *= -1
                if ((i+1)*(j+1)) % 2 != 0:
                    ss *= -1
                R[i,j] = (
                    u[i]*u[j] * (1 - np.cos(a))
                    + ss * u[3-(i+j)] * np.sin(a)
                )
    return np.dot(R,vec_0)

def ang_btwn(v1, v2):
    # vectors point away from the point where angle is taken
    # if np.linalg.norm(v1-v2) < 1e-6:
    #     return 0.
    cos_ab = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    if cos_ab > 1:
        return 0.
    ab = np.arccos(cos_ab)
    if np.isnan(ab):
        print(np.linalg.norm(v1))
        input(np.linalg.norm(v2))
    # if np.isnan(ab):
    #     print(np.dot(v1, v2)
    #     / (
    #         np.linalg.norm(v1)
    #         * np.linalg.norm(v2)
    #     ))
    #     print(np.dot(v1, v2))
    #     print(v1)
    #     print(np.linalg.norm(v1))
    #     print(v2)
    #     print(np.linalg.norm(v2))
    #     print(np.linalg.norm(v1)*np.linalg.norm(v2))
    #     print(ab)
    #     print('here')
    return ab

def ang_btwn2(v1, v2, v_anchor):
    # use sin and cos to find angle diff from -np.pi to np.pi
    # rotation angle of v1 to v2 wrt to axis v_anchor
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    v_anchor /= np.linalg.norm(v_anchor)
    e_tol = 1e-3
    if np.linalg.norm(v1-v2) < e_tol:
        return 0.
    sab = 0.
    cross12 = np.cross(v1, v2)
    # if np.linalg.norm(np.cross(cross12, v_anchor)) > e_tol:
    #     print(np.linalg.norm(np.cross(cross12, v_anchor)))
    #     print(np.linalg.norm(v1-v2))
    #     print('v1v2')
    #     print(v1)
    #     print(v2)
    #     print(cross12)
    #     print(v_anchor)
    #     raise Exception("inaccurate anchor")
    n_div = 0.
    for i in range(len(v1)):
        if abs(v_anchor[i]) < e_tol:
            continue
        sab += (
            cross12[i]
            / (
                np.linalg.norm(v1)
                * np.linalg.norm(v2)
            )
            / v_anchor[i]
        )
        n_div += 1.
    sab /= n_div
    cab = (
        np.dot(v1, v2)
        / (np.linalg.norm(v1) * np.linalg.norm(v2))
    )
    ab = np.arctan2(sab, cab)
    return ab

def ang_btwn3(v1, v2, v_anchor):
    theta_diff = np.arccos(v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    if np.cross(v1,v2).dot(v_anchor) < 0:
        theta_diff *= -1.
    return theta_diff

def create_skewsym(v1):
    return np.array(
        [
            [0., -v1[2], v1[1]],
            [v1[2], 0., -v1[0]],
            [-v1[1], v1[0], 0.]
        ]
    )

def create_distmatrix(adjacent_distvec):
    # converts a nx3 matrix of distvecs between
    # adjacent nodes into a distvec matrix between all nodes

    # distmat is such that distmat[a,b] is the dist vec
    # from node a to b (b-a)
    n_nodes = len(adjacent_distvec)+1
    distmat = np.zeros((n_nodes,n_nodes,3))
    dist_arr = adjacent_distvec.copy()
    indcs = np.arange(len(dist_arr))
    distmat[:-1,1:][indcs,indcs,:] = dist_arr.copy()
    """
    [0,A,a,x]
    [0,0,B,b]
    [0,0,0,C]
    [0,0,0,0]

    a = A + B
    b = B + C
    x = a + C = A + B + C
    """
    for i in range(2,n_nodes):
        distmat[:i-1,i] = distmat[:i-1,i-1] + distmat[i-1,i]
    distmat = distmat - np.transpose(distmat, (1, 0, 2))
    return distmat

def get_torqvec(forcevec, distmat, ids_i):
    """
    distmat is such that distmat[a,b] is the dist vec
    from node a to b (b-a)

    MuJoCo model of DER is such that B_N is the parent of B_(N+1)
    Since DER computation is from B_N to B_first(B_0),
    a reversed order from the MuJoCo model,

            F_2 (Upwards)
    B_3 --> B_2 --> B_1 --> B_0

    Case A (moment induced at B_(N+1) by force at B_N):
    For a force of F_2 upwards at B_2, a CCW moment is induced at B_3
    M3 = [F_2 cross (B_2 - B_3)]
    this moment is applied on B_3 from B_2. 
    To apply that torque through applyFT to qfrc_passive,
    we must find the torque on B_2 from B_3, which is the negative of M3

    Case B (moment induced at B_N by force at B_(N+1)):
    For a force of F_2 upwards at B_2, a CW moment is induced at B_1
    M1 = [F_2 cross (B_2 - B_1)]
    this moment is applied on B_1 from B_2. 
    To apply that torque through applyFT to qfrc_passive,
    we must find the torque on B_1 from B_2, which is exactly M1

    Thereforce to simplify calculations,
    we will make the upper triangular matrix 
    (distances from a to b (b-a) such that b > a) negative.
        Note: This has the effect of making the whole distance matrix
        give only one way distances, i.e.
        (distmat[a,b] = distmat[b,a] = distvec of b to a (a-b) such that a < b)
        a < b because index of DER is reversed 
            -- derbodyid_a < derbodyid_b is vecbodyid_a > vecbodyid_a
    """
    triu_distmat = (
        distmat 
        * np.triu(
            np.ones((distmat.shape[0], distmat.shape[1]), dtype=bool)
        )[:,:,np.newaxis]
    )
    distmat -= 2. * triu_distmat

    n_nodes = len(distmat)
    torq_vec = np.zeros((n_nodes,3))
    for i in ids_i:
    # for i in range(len(forcevec)):
        # print(i)
        # print(distmat[:,i])
        # print(forcevec[i])
        # print(np.cross(
            # distmat[:,i],
            # forcevec[i]
        # ))
        # input()
        torqvec_indiv = np.cross(
            distmat[:,i],
            forcevec[i]
        )
        # for i2 in range(len(torqvec_indiv)):
        #     if i2 < i:
        #         torqvec_indiv[i2] = np.zeros(3)
        torq_vec += torqvec_indiv
    # for i in range(len(torqvec_indiv)):
    #     print(i)
    #     print(torq_vec[i])

    return torq_vec

def force2torq(forcevec, adjacent_distvec, ids_i):
    distmat = create_distmatrix(adjacent_distvec)
    return get_torqvec(forcevec, distmat, ids_i)

def nearest_point_on_wire(wire_points, query_point):
    """
    Find the nearest point on a discretized wire (polyline) to a given query point.
    
    Parameters
    ----------
    wire_points : (N,3) ndarray
        The 3D points describing the wire in order.
    query_point : (3,) array_like
        The 3D coordinates of the query point.
    
    Returns
    -------
    nearest_point : (3,) ndarray
        The closest point on the wire (may lie inside a segment, not necessarily a node).
    segment_index : int
        Index of the starting vertex of the segment containing the nearest point.
    distance : float
        Euclidean distance from the query point to the nearest point.
    t : float
        Relative position along the segment [0,1].
    """
    wire_points = np.asarray(wire_points, dtype=float)
    q = np.asarray(query_point, dtype=float)

    if wire_points.shape[0] < 2:
        raise ValueError("Wire must contain at least 2 points")

    min_dist = np.inf
    nearest_point = None
    segment_index = -1
    t_best = 0.0

    for i in range(len(wire_points) - 1):
        p0, p1 = wire_points[i], wire_points[i+1]
        seg_vec = p1 - p0
        seg_len2 = np.dot(seg_vec, seg_vec)
        
        if seg_len2 == 0:  # degenerate segment
            proj = p0
            t = 0.0
        else:
            t = np.dot(q - p0, seg_vec) / seg_len2
            t = np.clip(t, 0.0, 1.0)  # restrict to segment
            proj = p0 + t * seg_vec

        dist = np.linalg.norm(q - proj)
        if dist < min_dist:
            min_dist = dist
            nearest_point = proj
            segment_index = i
            t_best = t

    return nearest_point, segment_index, min_dist, t_best

def curvature_binormals(positions):
    """
    Compute curvature binormal vectors at each interior node of a discretized curve.

    Args:
        positions (ndarray): shape (N, 3), node positions

    Returns:
        kb (ndarray): shape (N, 3), curvature binormal vectors (0 at endpoints)
    """
    N = len(positions)
    kb = np.zeros((N, 3))

    for i in range(1, N-1):  # only interior nodes
        e1 = positions[i] - positions[i-1]
        e2 = positions[i+1] - positions[i]
        denom = np.linalg.norm(e1) * np.linalg.norm(e2) + np.dot(e1, e2)
        if denom > 1e-12:  # avoid division by zero
            kb[i] = 2.0 * np.cross(e1, e2) / denom

    return kb

def kb_similarity_metric(positions, n=2):
    """
    Compute similarity metric of curvature binormals across neighbors.
    Magnitudes are normalized away, only directional coherence remains.

    Args:
        positions (ndarray): shape (N, 3), node positions
        n (int): neighborhood size

    Returns:
        kb_similarity (ndarray): shape (N,), similarity scores in [0,1]
    """
    kb = curvature_binormals(positions)
    N = len(kb)

    kb_similarity = np.zeros(N)
    for i in range(N):
        start = max(0, i - n)
        end = min(N, i + n + 1)
        neighborhood = kb[start:end]

        # Normalize all nonzero vectors, keep zeros as [0,0,0]
        norms = np.linalg.norm(neighborhood, axis=1, keepdims=True)
        unit_vecs = np.divide(
            neighborhood, norms, out=np.zeros_like(neighborhood), where=norms>1e-12
        )
        avg_unit_vec = np.mean(unit_vecs, axis=0)
        kb_similarity[i] = np.linalg.norm(avg_unit_vec)

    return kb_similarity

if __name__ == "__main__":
    v1 = np.array([1.5, 1.5, 0.])
    v2 = np.array([-1., 0., 0.])
    va = np.array([0., 1., 0.])
    va = np.cross(v1,v2)
    va *= 1.

    print(ang_btwn2(v1,v2,va)/np.pi*180)
    print(ang_btwn3(v1,v2,va)/np.pi*180)