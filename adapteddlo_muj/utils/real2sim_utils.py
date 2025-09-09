import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def compute_wire_frames(wire_pos):
    """
    Given wire_pos (N,3), returns:
    - body_positions: (N-1, 3) positions of each segment's center
    - body_quats: (N-1, 4) quaternions (w, x, y, z) for each segment
    - joint_quats: (N-2, 4) quaternions for each ball joint (rotation from previous to next segment)
    The capsule geoms are assumed to be aligned with the x-axis (tangential), and the material frame is parallel transported.
    """
    N = wire_pos.shape[0]
    tangents = np.diff(wire_pos, axis=0)
    tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)

    # Initial frame: x = tangent, y = arbitrary, z = cross(x, y)
    frames = []
    x0 = tangents[0]
    y0 = np.array([0, 1, 0])
    if np.abs(np.dot(x0, y0)) > 0.99:
        y0 = np.array([0, 0, 1])
    z0 = np.cross(x0, y0)
    z0 /= np.linalg.norm(z0)
    y0 = np.cross(z0, x0)
    y0 /= np.linalg.norm(y0)
    frames.append(np.stack([x0, y0, z0], axis=1))  # 3x3 matrix

    angle_lim = np.array([2*np.pi,0.0])
    # Parallel transport frames
    for i in range(1, N-1):
        prev_x = frames[-1][:,0]
        new_x = tangents[i]
        axis = np.cross(prev_x, new_x)
        if np.linalg.norm(axis) < 1e-8:
            frames.append(frames[-1])
            continue
        axis /= np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(prev_x, new_x), -1, 1))
        rot = R.from_rotvec(axis * angle)
        new_frame = rot.apply(frames[-1].T).T
        frames.append(new_frame)

        if angle < angle_lim[0]:
            angle_lim[0] = angle
        if angle > angle_lim[1]:
            angle_lim[1] = angle
        # print(f"newframe_{i} = {new_frame}")
        # new_x = tangents[i]
        # prev_frame = frames[-1]
        # prev_y = prev_frame[:,1]
        # # Project previous y onto plane perpendicular to new_x
        # y_proj = prev_y - np.dot(prev_y, new_x) * new_x
        # if np.linalg.norm(y_proj) < 1e-8:
        #     # If degenerate, pick an arbitrary perpendicular
        #     y_proj = np.cross(new_x, prev_frame[:,2])
        #     if np.linalg.norm(y_proj) < 1e-8:
        #         y_proj = np.cross(new_x, [1,0,0])
        #     y_proj /= np.linalg.norm(y_proj)
        # else:
        #     y_proj /= np.linalg.norm(y_proj)
        # z_new = np.cross(new_x, y_proj)
        # z_new /= np.linalg.norm(z_new)
        # y_new = np.cross(z_new, new_x)
        # y_new /= np.linalg.norm(y_new)
        # new_frame = np.stack([new_x, y_new, z_new], axis=1)
        # # print("new_x:", new_x)
        # # print("new_frame[:,0]:", new_frame[:,0])
        # # print("dot product:", np.dot(new_x, new_frame[:,0]))
        # frames.append(new_frame)

    # # Body positions
    # body_positions = 0.5 * (wire_pos[:-1] + wire_pos[1:])
    body_positions = wire_pos.copy()

    # Compute body quaternions (MuJoCo: [w, x, y, z])
    body_quats = []
    for f in frames:
        rot = R.from_matrix(f)
        q = rot.as_quat()  # [x, y, z, w]
        q = np.roll(q, 1)  # [w, x, y, z]
        # print(f"quat = {q}")
        body_quats.append(q)
    body_quats.append(np.array([1.0,0,0,0]))
    body_quats = np.array(body_quats)

    # Compute joint quaternions (rotation from previous to next frame)
    import adapteddlo_muj.utils.transform_utils as T
    cum_joint_quats = body_quats[0].copy()
    joint_quats = []
    # frames[0] = np.array([
        # [0,1,0.0],
        # [1,0,0.0],
        # [0,0,1.0],
    # ])
    # cum_joint_quats = T.mat2quat(frames[0])
    for i in range(len(frames)-1):
        rot_prev = R.from_matrix(frames[i])
        rot_next = R.from_matrix(frames[i+1])
        rel_rot = rot_prev.inv() * rot_next
        # rel_rot = rot_next * rot_prev.inv()
        q = rel_rot.as_quat()
        q = np.roll(q, 1)
        joint_quats.append(q)

        # print(T.quat_multiply(cum_joint_quats,q))
        q_rel = T.quat_multiply(
            T.quat_inverse(T.mat2quat(frames[i])),
            T.mat2quat(frames[i+1]),
        )
        # print(T.quat_multiply(cum_joint_quats,q_rel))
        # # print()
        # print(f"q = {q}")
        # print(f"rp_{i} = {frames[i]}")
        # print(f"rp_{i} = {rot_prev.as_matrix()}")
        # print(f"rn_{i} = {frames[i+1]}")
        # print(f"rn_{i} = {rot_next.as_matrix()}")
        # print(f"rr_{i} = {rel_rot.as_matrix()}")
        # print(f"cjq_{i} = {cum_joint_quats}")
        # input()
    joint_quats = np.array(joint_quats)

    # Let's assume you already have body_positions (N+1, 3) and body_quats (N+1, 4)
    N = wire_pos.shape[0] - 1
    body_relpos = np.zeros((N+1, 3))
    body_relpos[0] = np.zeros(3)  # The first body has no parent, so relpos is zero

    for i in range(1, N+1):
        delta = body_positions[i] - body_positions[i-1]
        prev_quat = body_quats[i-1]  # [w, x, y, z] or [x, y, z, w] depending on your convention
        # If your quats are [w, x, y, z], convert to [x, y, z, w] for scipy
        prev_quat_scipy = np.array([prev_quat[1], prev_quat[2], prev_quat[3], prev_quat[0]])

        ## REPLACED due to bad checks and confirm [w, x, y, z] format for body_quats
        # if prev_quat.shape[0] == 4 and prev_quat[0] != 0:  # likely [w, x, y, z]
            # prev_quat_scipy = np.array([prev_quat[1], prev_quat[2], prev_quat[3], prev_quat[0]])
        # else:
            # prev_quat_scipy = prev_quat

        r = R.from_quat(prev_quat_scipy)
        relpos = r.inv().apply(delta)
        body_relpos[i] = relpos

    # return as before, but add body_relpos
    return body_positions, body_quats, joint_quats, body_relpos, angle_lim


def compute_edge_offset_points(n1, n2, n3, e1, e2, da, h_in, h_out):
    """
    For 3 consecutive nodes (n1, n2, n3) and edge frames (e1, e2), compute 8 points (4 per edge) at midpoints,
    offset by ±da in y/z of the edge's basic frame, with orientation as specified, and return their positions and
    quaternions relative to n2/e2. For the offset point displaced toward negative z, displace it a further h_in/2 in +x (edge_idx==0) or -x (edge_idx==1). For the other points, displace by h_out/2 in the same manner. The two points offset by h_in are placed first in the returned arrays.
    Args:
        n1, n2, n3: np.ndarray, shape (3,)
        e1, e2: np.ndarray, shape (3,3) (columns are axes)
        da: float
        h_in, h_out: float
    Returns:
        positions: (8, 3) np.ndarray (relative to n2, in e2 frame)
        quaternions: (8, 4) np.ndarray (relative to e2, as [w, x, y, z])
    """
    from scipy.spatial.transform import Rotation as R
    # print(np.linalg.norm(e1[0]))
    # print(np.linalg.norm(e1[:,0]))
    # input(e1)
    # input(e2)
    points = []
    quats = []
    h_in_points = []
    h_in_quats = []
    global_pts = []
    # Compute curvature binormal (for both edges, same binormal)
    t1 = n2 - n1
    t2 = n3 - n2
    t1n = t1 / np.linalg.norm(t1)
    t2n = t2 / np.linalg.norm(t2)
    binormal = np.cross(t1n, t2n)
    if np.linalg.norm(binormal) < 1e-8:
        # If straight, pick arbitrary binormal
        binormal = np.array([0, 0, 1]) if abs(t1n[2]) < 0.9 else np.array([0, 1, 0])
    binormal = binormal / np.linalg.norm(binormal)
    # For each edge
    for edge_idx, (a, b, e_mat) in enumerate([(n1, n2, e1), (n2, n3, e2)]):
        edge = b - a
        x = edge / np.linalg.norm(edge)
        y = binormal
        z = np.cross(x, y)
        z = z / np.linalg.norm(z)
        y = np.cross(z, x)  # re-orthogonalize
        y /= np.linalg.norm(y)
        # e_basic: columns are [x, y, z]
        e_basic = np.stack([x, y, z], axis=1)
        midpoint = (a + b) / 2
        # 4 offsets: +y, -y, +z, -z
        for k, (offset_axis, sign) in enumerate([(1, +1), (1, -1), (2, +1), (2, -1)]):
            offset_vec = np.zeros(3)
            offset_vec[offset_axis] = sign * da
            # For -z offset (k==3): use h_in, others use h_out

            if offset_axis == 2 and sign == -1:
                # -z direction
                if edge_idx == 0:
                    offset_vec[0] += h_in/2
                else:
                    offset_vec[0] -= h_in/2
            else:
                if edge_idx == 0:
                    offset_vec[0] += h_out/2
                else:
                    offset_vec[0] -= h_out/2

            # Point in e_basic frame
            pt = midpoint + e_basic @ offset_vec
            global_pts.append(pt)
            # if offset_axis == 1:
            #     y_new = e_basic @ offset_vec
            #     y_new /= np.linalg.norm(y_new)
            #     # print(e_basic)
            #     # print(offset_vec)
            #     print(sign*y_new)
            #     print(f"yaxis = {y}")
            y_new = e_basic @ offset_vec
            # print()
            # print(f"e_basic = {e_basic}")
            # print(f"offset_vec = {offset_vec}")
            # print(f"y_new = {y_new}")
            # print(f"mp = {midpoint}")
            # print(f"pt = {pt}")
            # print()
            # Orientation: x = edge x, y = -offset direction, z = x cross y
            x_axis = x
            y_axis = -sign * e_basic[:, offset_axis]
            z_axis = np.cross(x_axis, y_axis)
            z_axis = z_axis / np.linalg.norm(z_axis)
            y_axis = np.cross(z_axis, x_axis)  # re-orthogonalize
            R_pt = np.stack([x_axis, y_axis, z_axis], axis=1)


            # Check: y_axis should point from point to edge and be perpendicular to x_axis
            # Vector from point to edge (project pt onto edge, then pt_proj - pt)
            edge_point = midpoint  # closest point on edge to pt (since pt is offset from midpoint)
            vec_to_edge = edge_point - pt
            y_axis_norm = y_axis / np.linalg.norm(y_axis)
            vec_to_edge_norm = vec_to_edge / (np.linalg.norm(vec_to_edge) + 1e-12)
            dot_y_edge = np.dot(y_axis_norm, vec_to_edge_norm)
            dot_y_x = np.dot(y_axis_norm, x_axis / np.linalg.norm(x_axis))
            # if dot_y_edge < 0.95 or abs(dot_y_x) > 1e-6:
                # print(f"Warning: y-axis of orientation for point {k} (edge {edge_idx}) may not point to edge or is not perpendicular to x-axis. dot_y_edge={dot_y_edge:.3f}, dot_y_x={dot_y_x:.3e}")
                # print(y_axis)
                # print(binormal)
            dotxx = np.dot(
                x_axis/np.linalg.norm(x_axis),
                e_mat[:,0]/np.linalg.norm(e_mat[:,0])
            )
            if dotxx < 0.95:
                print("BAD X")
                print(x)
                print(e_mat[:,0])
            
            # Transform position and orientation to n2/e2 frame
            pt_rel = pt - n2
            # Express in e2 frame
            pt_rel_e2 = e2.T @ pt_rel
            # Orientation relative to e2: R_rel = e2.T @ R_pt
            R_rel = e2.T @ R_pt
            quat = R.from_matrix(R_rel).as_quat()  # [x, y, z, w]
            quat = np.roll(quat, 1)  # [w, x, y, z]
            if offset_axis == 2 and sign == -1:
                h_in_points.append(pt_rel_e2)
                h_in_quats.append(quat)
            else:
                points.append(pt_rel_e2)
                quats.append(quat)
            # print(k)
            # print(R_rel)
            # print(quat)
    return np.vstack(h_in_points + points), np.vstack(h_in_quats + quats)# , np.array(global_pts) 


if __name__ == "__main__":
    # Test compute_edge_offset_points
    import numpy as np
    n1 = np.array([0.0, 0.0, 0.0])
    n2 = np.array([1.0, 0.0, 0.0])
    n3 = np.array([1.0, 0.707, 0.707])
    n3 = n2 + 0.5773502691896257
    # n3 = np.array([1.0, 0.0, -1.0])
    da = 0.1
    h_in = 0.0
    h_out = 0.0
    # Edge directions
    x1 = (n2 - n1) / np.linalg.norm(n2 - n1)
    x2 = (n3 - n2) / np.linalg.norm(n3 - n2)
    # Curvature binormal
    binormal = np.cross(x1, x2)
    if np.linalg.norm(binormal) < 1e-8:
        binormal = np.array([0, 0, 1])
    binormal = binormal / np.linalg.norm(binormal)
    # y axes
    y1 = binormal
    z1 = np.cross(x1, y1)
    z1 = z1 / np.linalg.norm(z1)
    y1 = np.cross(z1, x1)
    y1 = y1 / np.linalg.norm(y1)
    e1 = np.stack([x1, y1, z1], axis=1)
    y2 = binormal
    z2 = np.cross(x2, y2)
    z2 = z2 / np.linalg.norm(z2)
    y2 = np.cross(z2, x2)
    y2 = y2 / np.linalg.norm(y2)
    e2 = np.stack([x2, y2, z2], axis=1)
    print(f"e = {e1}")
    print(f"e = {e2}")
    # Call function
    pos, quat, pt = compute_edge_offset_points(n1, n2, n3, e1, e2, da, h_in, h_out)
    print("Positions:")
    print(pos)
    print("Quaternions:")
    print(quat) 

    # Example points (replace with your actual values)
    # n1 = np.array([0, 0, 0])
    # n2 = np.array([1, 0, 0])
    # n3 = np.array([1, 1, 0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot n1, n2, n3
    ax.scatter(*n1, color='r', label='n1')
    ax.scatter(*n2, color='g', label='n2')
    ax.scatter(*n3, color='b', label='n3')
    ax.scatter(pt[:,0], pt[:,1], pt[:,2], color='k', label='pt', s=80, marker='x')

    # Optionally, connect the nodes
    ax.plot([n1[0], n2[0], n3[0]], [n1[1], n2[1], n3[1]], [n1[2], n2[2], n3[2]], 'c--', label='chain')

    # Annotate
    ax.text(*n1, 'n1', color='r')
    ax.text(*n2, 'n2', color='g')
    ax.text(*n3, 'n3', color='b')
    # ax.text(pt[:,0], pt[:,1], pt[:,2], 'pt', color='k')

    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show() 

    # Set equal aspect ratio for all axes
    def set_axes_equal(ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc.  This is one possible solution to Matplotlib's ax.set_aspect('equal') and ax.axis('equal') not working for 3D.'''
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    set_axes_equal(ax) 