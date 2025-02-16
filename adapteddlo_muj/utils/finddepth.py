import numpy as np

def locatepointin3d(
    xydepth_pt,
    img_size=np.array([1280,720], dtype='int'),
    img_fov=None
):
    if img_fov is None:
        img_fov = np.array([75.0,65.0])
    img_fov *= np.pi/180.0
    half_size = np.array(img_size/2, dtype='int')
    half_fov = img_fov/2

    xy_angle = (xydepth_pt[:2] - half_size) / half_size * half_fov

    # print(f"half_fov = {img_fov}")
    # print(f"xy_angle = {xy_angle}")

    xyz_pos = np.zeros(3)
    xyz_pos[1] = xydepth_pt[2] / np.sqrt(1 + np.tan(xy_angle[0])**2 + np.tan(xy_angle[1])**2)
    xyz_pos[1] = xydepth_pt[2]
    xyz_pos[0] = xyz_pos[1] * np.tan(xy_angle[0])
    xyz_pos[2] = - xyz_pos[1] * np.tan(xy_angle[1])
    # print(f"z_angle = {np.tan(xy_angle[1])}")
    # print(f"xyz_pos = {xyz_pos}")
    return xyz_pos

def rescale_points(points_arr,r_len=1.0,r_pieces=10):
    total_imglen = 0.0
    for i in range(len(points_arr)-1):
        total_imglen += np.linalg.norm(points_arr[i]-points_arr[i+1])
    wire_img2real_dist = r_len/total_imglen
    points_arr *= wire_img2real_dist
    seg_len = r_len/r_pieces
    new_points_arr = split_lines(points_arr,seg_len,0.0)
    return new_points_arr

def rotate_points(points, axis, angle_radians):
    """
    Rotate a list of 3D points around a given axis by a certain angle.

    :param points: List or numpy array of 3D points (each point is a list or tuple of 3 values)
    :param axis: Axis of rotation, should be a 3D unit vector (e.g., [1, 0, 0] for X-axis)
    :param angle_degrees: Angle to rotate by, in degrees
    :return: A numpy array of the rotated points
    """
    
    # Normalize the rotation axis
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    
    # Rodrigues' rotation formula components
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    ux, uy, uz = axis
    
    # Rotation matrix
    rotation_matrix = np.array([
        [cos_theta + ux**2 * (1 - cos_theta), ux*uy * (1 - cos_theta) - uz*sin_theta, ux*uz * (1 - cos_theta) + uy*sin_theta],
        [uy*ux * (1 - cos_theta) + uz*sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy*uz * (1 - cos_theta) - ux*sin_theta],
        [uz*ux * (1 - cos_theta) - uy*sin_theta, uz*uy * (1 - cos_theta) + ux*sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
    ])
    
    # Convert points to a numpy array
    points = np.array(points)
    
    # Apply the rotation to all points
    rotated_points = np.dot(points, rotation_matrix)
    
    return rotated_points

def scale_points_custom_axis(points, axis_vector, scale_factor):
    # Ensure points is a numpy array for easier manipulation
    points = np.array(points)
    # Normalize the axis vector to ensure proper scaling along the unit direction
    axis_vector = np.array(axis_vector)
    axis_vector = axis_vector / np.linalg.norm(axis_vector)
    # Project each point onto the axis vector and then scale along this axis
    for i in range(len(points)):
        # Find the projection of the point onto the axis vector
        projection = np.dot(points[i], axis_vector) * axis_vector
        # Scale the projection component by the scale factor
        scaled_projection = projection * scale_factor
        # Update the point by adjusting its projection along the axis
        points[i] = points[i] + (scaled_projection - projection)
    return points

def find_rotaxis_to_neutral(start_pt, end_pt, end_pt_axis=np.array([1.0,0.0,0.0])):
    # finds the axis to rotate a straight line with start and end point defined
    # into a new straight line with same start_pt and is parallel to end_pt_axis
    # anchor all points - start_ptt is origin
    end_pt -= start_pt
    end_pt_goal = np.linalg.norm(end_pt)*end_pt_axis/np.linalg.norm(end_pt_axis)
    start_pt -= start_pt
    # print(end_pt)
    # print(start_pt)
    # print(end_pt_axis/np.linalg.norm(end_pt_axis))
    # print(end_pt_goal)
    end_pt_mid = (end_pt + end_pt_goal) / 2.0
    ax1 = end_pt_mid - start_pt
    ax2 = end_pt_goal - end_pt
    rot_axis = np.cross(ax1,ax2)
    rot_angle = - np.arcsin(np.linalg.norm(end_pt_mid-end_pt)/np.linalg.norm(end_pt))*2.0
    return rot_axis, rot_angle

def adjust_linestandard(
        points_arr,
        startend_axis,  #end_pt-star_pt
    ):
    # rotate all points about start_pt (origin) to match sim 
    rot_axis, rot_angle = find_rotaxis_to_neutral(
        points_arr[0].copy(),
        points_arr[-1].copy(),
        startend_axis.copy()
    )
    new_ptarr = rotate_points(points_arr,rot_axis,rot_angle)
    new_seaxis = points_arr[-1]-points_arr[0]
    new_ptarr = scale_points_custom_axis(
        new_ptarr,
        startend_axis,
        scale_factor=(
            np.linalg.norm(startend_axis)
            / np.linalg.norm(new_seaxis)
        )
    )
    # print(f"rot_axis = {rot_axis}")
    # print(f"rot_angle = {rot_angle}")
    return new_ptarr

def scale_linestandard(
        points_arr,
        startend_axis,  #end_pt-star_pt
    ):
    all_scaled = True
    # scale each axis according to sim values
    for i in range(3):
        real_robend = points_arr[-1,i] - points_arr[0,i]
        if abs(real_robend) <1e-3 or abs(startend_axis[i])<1e-3: 
            all_scaled = False
            continue
        real2sim_scale = startend_axis[i]/real_robend
        points_arr[:,i] *= real2sim_scale
    if not all_scaled:
        points_arr = adjust_linestandard(points_arr=points_arr, startend_axis=startend_axis)

    return points_arr

def split_lines(points_arr,seg_len,leftover_len=None):
    new_pts_arr = []
    new_pts_arr.append(points_arr[0])
    # leftover_len = 0.0
    if leftover_len is None:
        leftover_len = seg_len/2
    for i in range(len(points_arr)-1):
        part_len = np.linalg.norm(points_arr[i]-points_arr[i+1])
        part_dir = (points_arr[i+1]-points_arr[i]) / part_len
        part_len += leftover_len
        n_seg = int(part_len/seg_len)
        for j in range(n_seg):
            new_pts_arr.append(
                points_arr[i] + ((j+1)*seg_len-leftover_len)*part_dir
            )
        leftover_len = part_len - n_seg*seg_len
    if leftover_len > 1e-5:
        new_pts_arr.append(points_arr[-1])
    new_pts_arr = np.array(new_pts_arr)
    return new_pts_arr

def split_lines2(points_arr, n_pieces):
    total_len = 0.0
    for i in range(len(points_arr)-1):
        total_len += np.linalg.norm(points_arr[i]-points_arr[i+1])
    seg_len = total_len/n_pieces
    return split_lines(points_arr, seg_len, leftover_len=0.0)
   
def scale_along_axis_3d(axis, points, scale_factor):
    # Normalize the axis vector to make sure it's a unit vector
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)

    # Function to scale a single point in 3D
    def scale_point(point):
        # Convert point to a numpy array
        point = np.array(point)

        # Project the point onto the axis
        projection = np.dot(point, axis) * axis

        # Translate the point by subtracting the projection
        translated_point = point - projection

        # Scale the translated point
        scaled_translated_point = scale_factor * translated_point

        # Translate it back by adding the projection
        scaled_point = scaled_translated_point + projection

        return scaled_point

    # Apply the scaling to all points
    scaled_points = np.array([scale_point(p) for p in points])

    return scaled_points

def optimize_through_axisscale(og_posarr, se_axis, r_len):
    proxy_realpos = og_posarr.copy()
    lbub = None
    len_real = len_pts(og_posarr.copy())
    l_diff = len_real - r_len
    old_ldiff = l_diff
    scale_step = [0.9,1.1]
    ss_old = 1.0
    loop_cnt = 0
    while abs(l_diff) > 1e-5:
        if loop_cnt == 0:
            if l_diff > 0: 
                ss_now = scale_step[0]
            else: 
                ss_now = scale_step[1]
            ss_og = ss_now
        loop_cnt += 1
        proxy_realpos = scale_along_axis_3d(
            se_axis, og_posarr, ss_now
        )
        # print(proxy_realpos)
        len_real = len_pts(proxy_realpos)
        l_diff = len_real - r_len
        if lbub is None:
            if old_ldiff*l_diff<0:
                lbub = [ss_old, ss_now]
                last_id = 1
                ss_now = (lbub[0]+lbub[1])/2
            else:
                ss_old = ss_now
                ss_now *= ss_og
        else:
            if old_ldiff*l_diff<0:
                last_id = 1 - last_id
                lbub[last_id] = ss_now
            else:
                lbub[last_id] = ss_now
            ss_now = (lbub[0]+lbub[1])/2
        old_ldiff = l_diff
    return proxy_realpos

def len_pts(points):
    return np.sum(np.linalg.norm(points[1:]-points[:-1], axis=1))