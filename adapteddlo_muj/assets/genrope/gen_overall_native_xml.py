import os
import numpy as np
from adapteddlo_muj.utils.real2sim_utils import compute_wire_frames, compute_edge_offset_points
from scipy.spatial.transform import Rotation as R
from adapteddlo_muj.utils.transform_utils import scipy_quat_to_matrix

def generate_overall_native_xml(
    wire_pos,
    n_pieces=13,
    thickness=0.03,
    mass=1.0,
    j_damp=0.5,
    con_val=(0,0),
    stiff_bend=3.38e6,
    stiff_twist=9.92e6,
    init_pos=None,
    init_quat=None,
    xml_path=None,
    calcEnergy=True,
    someStiff=0,
    rgba="0.1 0.0533333 0.673333 1",
):
    """
    Generate a MuJoCo XML for a discretized wire with welded ends.
    """
    if xml_path is None:
        obj_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'overall_s2f.xml'
        )
    else:
        obj_path = xml_path

    # Calculate the total length of the wire from wire_pos
    diffs = np.diff(wire_pos, axis=0)
    length = np.sum(np.linalg.norm(diffs, axis=1))
    if wire_pos is None:
        wire_pos = np.zeros((n_pieces+1, 3))
        wire_pos[:, 0] = np.linspace(0, length, n_pieces+1)
    body_positions, body_quats, joint_quats, body_relpos,_ = compute_wire_frames(wire_pos)
    # print(f"jcalc = {joint_quats}")
    mass_per = mass / n_pieces
    if init_pos is None:
        init_pos = np.zeros(3)
    init_pos += body_positions[0]
    if init_quat is None:
        init_quat = np.array([1.0,0,0,0])
    (contype, conaffinity) = con_val
    # contype = 1 if collision_on else 0
    # conaffinity = 1 if collision_on else 0

    # Compute world position and quaternion for B_first and B_last
    def accumulate_chain_world_pose(init_pos, init_quat, body_relpos, body_quats, idx):
        pos = np.array(init_pos)
        quat = np.array(init_quat)

        for i in range(idx+1):
            quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
            r = R.from_quat(quat_scipy)
            pos = pos + r.apply(body_relpos[i])
            if i < len(body_quats):
                q_child = body_quats[i]
                w0, x0, y0, z0 = quat
                w1, x1, y1, z1 = q_child
                quat = np.array([
                    w0*w1 - x0*x1 - y0*y1 - z0*z1,
                    w0*x1 + x0*w1 + y0*z1 - z0*y1,
                    w0*y1 - x0*z1 + y0*w1 + z0*x1,
                    w0*z1 + x0*y1 - y0*x1 + z0*w1
                ])
                quat = quat / np.linalg.norm(quat)
        return pos, quat
    

    # def compute_body_quats_from_joints(init_quat, joint_quats):
    #     # All quats in [w, x, y, z] (MuJoCo)
    #     n = joint_quats.shape[0] + 1
    #     body_quats = np.zeros((n, 4))
    #     # Convert to scipy format [x, y, z, w]
    #     q = np.array([init_quat[1], init_quat[2], init_quat[3], init_quat[0]])
    #     body_quats[0] = init_quat
    #     for i in range(1, n):
    #         q_joint = joint_quats[i-1]
    #         q_joint_s = np.array([q_joint[1], q_joint[2], q_joint[3], q_joint[0]])
    #         q = (R.from_quat(q) * R.from_quat(q_joint_s)).as_quat()
    #         # Convert back to MuJoCo format
    #         body_quats[i] = np.array([q[3], q[0], q[1], q[2]])
    #     return body_quats
    
    # Stack init_quat and joint_quats into a (n_pieces, 4) array
    body_relquat = np.vstack((body_quats[0], joint_quats))
    b_0_pos, b_0_quat = accumulate_chain_world_pose(init_pos, init_quat, body_relpos, body_relquat, 0)
    b_last2_pos, b_last2_quat = accumulate_chain_world_pose(init_pos, init_quat, body_relpos, body_relquat, n_pieces)

    xml = []
    xml.append('<mujoco model="base">')
    xml.append('  <compiler angle="radian"/>')
    xml.append('  <option timestep="0.0005" tolerance="1e-10" integrator="RK4" cone="elliptic" jacobian="sparse" iterations="30"/>')
    xml.append('  <size njmax="5000" nconmax="5000"/>')
    xml.append('  <visual>')
    xml.append('    <quality shadowsize="2048"/>')
    xml.append('    <map stiffness="700" fogstart="10" fogend="15" znear="0.001" zfar="40" shadowscale="0.5"/>')
    xml.append('    <rgba haze="0.15 0.25 0.35 1"/>')
    xml.append('  </visual>')
    xml.append('  <statistic meansize="0.05" extent="2"/>')
    xml.append('  <extension>')
    xml.append('    <plugin plugin="mujoco.elasticity.wire">')
    xml.append('      <instance name="composite">')
    xml.append(f'        <config key="twist" value="{stiff_twist}"/>')
    xml.append(f'        <config key="bend" value="{stiff_bend}"/>')
    xml.append('        <config key="flat" value="true"/>')
    xml.append(f'        <config key="calcEnergy" value="{"true" if calcEnergy else "false"}"/>')
    xml.append('        <config key="pqsActive" value="true"/>')
    if someStiff > 0:
        xml.append(f'        <config key="someStiff" value="{someStiff}"/>')
    xml.append('      </instance>')
    xml.append('    </plugin>')
    xml.append('  </extension>')
    xml.append('  <asset>')
    xml.append('    <texture type="skybox" builtin="gradient" rgb1="1 1 0.9" rgb2="0.9 0.9 0.81" width="512" height="3072"/>')
    xml.append('    <texture type="2d" name="texplane" builtin="checker" mark="cross" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2" markrgb="0.8 0.8 0.8" width="512" height="512"/>')
    xml.append('    <material name="matplane" texture="texplane" texuniform="true" reflectance="0.3"/>')
    xml.append('  </asset>')
    xml.append('  <worldbody>')
    xml.append('    <geom name="floor" size="3 3 0.125" pos="0 0 -10" type="plane" condim="1" material="matplane"/>')
    xml.append('    <geom name="axis_x" type="capsule" fromto="0 0 0 0.03 0 0" size="0.002" rgba="1 0 0 1" contype="0" conaffinity="0"/>')
    xml.append('    <geom name="axis_y" type="capsule" fromto="0 0 0 0 0.03 0" size="0.002" rgba="0 1 0 1" contype="0" conaffinity="0"/>')
    xml.append('    <geom name="axis_z" type="capsule" fromto="0 0 0 0 0 0.03" size="0.002" rgba="0 0 1 1" contype="0" conaffinity="0"/>')

    xml.append(f'    <body name="eef_body2" pos="{b_0_pos[0]} {b_0_pos[1]} {b_0_pos[2]}" quat="{b_0_quat[0]} {b_0_quat[1]} {b_0_quat[2]} {b_0_quat[3]}">')
    xml.append(f'      <geom name="eef_geom2" size="{thickness} {thickness} {thickness*2}" type="box" contype="0" conaffinity="0" mass="10" rgba="0.1 0.8 0.2 0.7"/>')
    xml.append('      <site name="eef_body2_site" pos="0 0 0" size="0.01" group="2" rgba="0 0 0 0"/>')
    xml.append('      <body name="eef_body2_sensor">')
    xml.append('        <site name="sensor_site2" pos="0 0 0" size="0.01" group="2" rgba="0 0 0 0"/>')
    xml.append('      </body>')
    xml.append('    </body>')
    xml.append(f'    <body name="eef_body" pos="{b_last2_pos[0]} {b_last2_pos[1]} {b_last2_pos[2]}" quat="{b_last2_quat[0]} {b_last2_quat[1]} {b_last2_quat[2]} {b_last2_quat[3]}">')
    xml.append(f'      <geom name="eef_geom" size="{thickness} {thickness} {thickness*2}" type="box" contype="0" conaffinity="0" mass="10" rgba="0.8 0.2 0.1 0.7"/>')
    xml.append('      <site name="eef_body_site" pos="0 0 0" size="0.01" group="1" rgba="0 0 0 0"/>')
    xml.append('      <body name="eef_body_sensor">')
    xml.append('        <site name="sensor_site1" pos="0 0 0" size="0.01" group="1" rgba="0 0 0 0"/>')
    xml.append('      </body>')
    xml.append('    </body>')

    xml.append(f'    <body name="stiffrope" pos="{init_pos[0]} {init_pos[1]} {init_pos[2]}" quat="{init_quat[0]} {init_quat[1]} {init_quat[2]} {init_quat[3]}">')
    xml.append('      <joint name="freejoint_A" type="free" limited="false" actuatorfrclimited="false"/>')

    indent = "      "
    for i in range(n_pieces):
        if i == 0:
            i_name = 'first'
        elif i == n_pieces-1:
            i_name = 'last'
        else:
            i_name = str(i)
        name = f'B_{i_name}'
        if name == 'B_last' or name == 'B_first':
            mass_used = 10.0
        else:
            mass_used = mass_per
        # Use joint_quats for body quats
        pos = body_relpos[i]
        g_length = np.linalg.norm(body_relpos[i+1])/2.0
        if i-1 < 0:
            quat = body_quats[0]
        else:
            quat = joint_quats[i-1] if (i-1) < len(joint_quats) else np.array([1.0,0,0,0])
        # print(f"quatgen_{i} = {quat}")
        xml.append(f'{indent}<body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}">')
        if i > 0:
            jname = f'J_{i}' if i < n_pieces-1 else 'J_last'
            # Do not specify quat for joint
            xml.append(f'{indent}  <joint name="{jname}" pos="0 0 0" type="ball" group="3" actuatorfrclimited="false" damping="{j_damp}"/>')
        gname = f'G{i}'
        xml.append(f'{indent}  <geom name="{gname}" size="{thickness/2} {g_length}" pos="{g_length} 0 0" quat="0.707107 0 -0.707107 0" type="capsule" contype="{contype}" conaffinity="{conaffinity}" condim="1" solref="0.001" mass="{mass_used}" rgba="{rgba}"/>')
        xml.append(f'{indent}  <plugin instance="composite"/>')
        indent += "  "  # Increase indent for next nested body

    # Add B_last2 as a child of B_last
    # pos = body_positions[n_pieces]-body_positions[n_pieces-1]
    pos = body_relpos[n_pieces]
    indent = "      " + "  " * n_pieces  # One more indent than the last body
    xml.append(f'{indent}<body name="B_last2" pos="{pos[0]} {pos[1]} {pos[2]}">')
    xml.append(f'{2*indent}<plugin instance="composite"/>')
    xml.append(f'{indent}</body>')

    # After the loop, close all bodies in reverse order
    for i in range(n_pieces):
        indent = "      " + "  " * (n_pieces - i - 1)
        xml.append(f'{indent}</body>')

    xml.append('    </body>')  # Close "stiffrope"
    xml.append('  </worldbody>')

    xml.append('  <equality>')
    xml.append('    <weld name="weld_end" body1="B_last" body2="eef_body_sensor" anchor="0 0 0" relpose="0 0 0 0 0 0 0" torquescale="1" solref="0.001"/>')
    xml.append('    <weld name="weld_start" body1="B_first" body2="eef_body2_sensor" anchor="0 0 0" relpose="0 0 0 0 0 0 0" torquescale="1" solref="0.001"/>')
    xml.append('  </equality>')

    xml.append('  <sensor>')
    xml.append('    <torque site="sensor_site1" name="torque_A"/>')
    xml.append('    <force site="sensor_site1" name="force_A"/>')
    xml.append('    <torque site="sensor_site2" name="torque_B"/>')
    xml.append('    <force site="sensor_site2" name="force_B"/>')
    xml.append('  </sensor>')
    xml.append('</mujoco>')

    with open(obj_path, 'w') as f:
        f.write('\n'.join(xml))
    print(f'Wrote {obj_path}')

def generate_smoothingpipe_xml(
    wire_pos,
    n_pieces=13,
    thickness=0.03,
    dr_range = [1.1, 5.0],   # distratio_range
    mass=1.0,
    con_val=(0,0),
    init_pos=None,
    init_quat=None,
    xml_path=None,
    smooth_multi=1.0,
    rgba="0.1 0.0533333 0.673333 1",
):
    """
    Generate a MuJoCo XML for a discretized smooth pipe (no plugins, joints, or extra bodies),
    with 4 box geoms per body forming a square tube around the axis.
    """
    if xml_path is None:
        obj_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'smoothingpipe.xml'
        )
    else:
        obj_path = xml_path
        
    (contype, conaffinity) = con_val

    # Calculate the total length of the wire from wire_pos
    diffs = np.diff(wire_pos, axis=0)
    length = np.sum(np.linalg.norm(diffs, axis=1))
    l_half = np.mean(np.linalg.norm(diffs, axis=1))/2.0

    if wire_pos is None:
        wire_pos = np.zeros((n_pieces+1, 3))
        wire_pos[:, 0] = np.linspace(0, length, n_pieces+1)
    body_positions, body_quats, joint_quats, body_relpos, angle_lim = compute_wire_frames(wire_pos)
    mass_per = mass / n_pieces
    if init_pos is None:
        init_pos = np.zeros(3)
    init_pos += body_positions[0]
    if init_quat is None:
        init_quat = np.array([1.0,0,0,0])

    # Stack init_quat and joint_quats into a (n_pieces, 4) array
    body_relquat = np.vstack((body_quats[0], joint_quats))
    # print(f"quat = {body_quats}")

    xml = []
    xml.append('<mujoco model="smoothingpipe">')
    xml.append('  <worldbody>')
    xml.append(f'    <body name="smp" pos="{init_pos[0]} {init_pos[1]} {init_pos[2]}" quat="{init_quat[0]} {init_quat[1]} {init_quat[2]} {init_quat[3]}">')

    indent = "      "
    thin = thickness / 4
    for i in range(n_pieces):
        name = f'smp_B_{i}' if i < n_pieces-1 else 'smp_B_last'
        pos = body_relpos[i]
        g_length = np.linalg.norm(body_relpos[i+1])/2.0
        if i-1 < 0:
            quat = body_quats[0]
        else:
            quat = joint_quats[i-1] if (i-1) < len(joint_quats) else np.array([1.0,0,0,0])
        xml.append(f'{indent}<body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}">')
        
        # box define
        if i > 0 and i < (n_pieces):
            # give error if curvature binormal of any points more than pi/2
            v1 = body_positions[i] - body_positions[i-1]
            v2 = body_positions[i+1] - body_positions[i]
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)
            cos_theta = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            theta_acute = np.arccos(cos_theta)  
            theta = np.pi - np.arccos(cos_theta)  

            distaway = (
                (dr_range[0]-dr_range[1]) / (angle_lim[1]-angle_lim[0])
                * (theta_acute - angle_lim[0]) + dr_range[1]
            ) * thickness

            angle_tol = 2 * np.arctan(distaway / l_half)
            if theta < angle_tol:
                raise ValueError(f"Bend angle at segment {i} is {np.degrees(theta):.2f} deg, which exceeds the threshold of {np.degrees(angle_tol):.2f} deg.")
            
            # get length of boxes
            h_in = max(l_half - distaway/np.tan(theta/2),0.001)
            h_out = max(l_half + distaway/np.tan(theta/2),0.001)

            # get points from which to define box
            ppos, pquat = compute_edge_offset_points(
                body_positions[i-1],
                body_positions[i],
                body_positions[i+1],
                scipy_quat_to_matrix(body_quats[i-1]),
                scipy_quat_to_matrix(body_quats[i]),
                distaway,
                h_in, h_out
            )

            # write boxes
            # inner points
            mass_offset_box = 10.0
            write_offset_box_geoms(
                xml, ppos[:2], pquat[:2], h_in, thickness, distaway*smooth_multi, indent,
                f"smp_G{i}", rgba, mass_offset_box/8, contype, conaffinity, k_start=0,
            )
            # # other points
            # write_offset_box_geoms(
            #     xml, ppos[2:], pquat[2:], h_out, thickness, distaway, indent, f"smp_G{i}", rgba, mass_per/8, contype, conaffinity, k_start=2
            # )

        # for sign in [+1, -1]:
        #     box_pos = f"{g_length} {sign*j} 0"
        #     box_size = f"{g_length} {thin} {2*j}"
        #     xml.append(f'{indent}  <geom name="smp_G{i}_y{sign}" type="box" pos="{box_pos}" size="{box_size}" mass="{mass_per/4}" contype="{contype}" conaffinity="{conaffinity}" rgba="{rgba}"/>')
        #     box_pos = f"{g_length} 0 {sign*j}"
        #     box_size = f"{g_length} {2*j} {thin}"
        #     xml.append(f'{indent}  <geom name="smp_G{i}_z{sign}" type="box" pos="{box_pos}" size="{box_size}" mass="{mass_per/4}" contype="{contype}" conaffinity="{conaffinity}" rgba="{rgba}"/>')
        indent += "  "  # Increase indent for next nested body

    # Add B_last2 as a child of B_last
    pos = body_relpos[n_pieces]
    indent = "      " + "  " * n_pieces  # One more indent than the last body
    xml.append(f'{indent}<body name="smp_B_last2" pos="{pos[0]} {pos[1]} {pos[2]}">')
    xml.append(f'{indent}</body>')

    # After the loop, close all bodies in reverse order
    for i in range(n_pieces):
        indent = "      " + "  " * (n_pieces - i - 1)
        xml.append(f'{indent}</body>')

    xml.append('    </body>')  # Close "stiffrope"
    xml.append('  </worldbody>')
    xml.append('</mujoco>')

    with open(obj_path, 'w') as f:
        f.write('\n'.join(xml))
    print(f'Wrote {obj_path}')

def write_offset_box_geoms(xml, ppos, pquat, h, thickness, distaway, indent, gname_prefix, rgba, mass, contype, conaffinity, k_start=0):
    """
    Appends <geom> elements to xml for each of the offset points.
    Each box has size (h, thickness, distaway*2), is placed at ppos[k] with orientation pquat[k].
    k_start: starting index for the geom name suffix (default 0)
    """
    for k in range(ppos.shape[0]):
        pos_str = f"{ppos[k,0]} {ppos[k,1]} {ppos[k,2]}"
        quat = pquat[k]
        quat_str = f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"
        size_str = f"{h/2} {thickness/2} {distaway}"
        gname = f"{gname_prefix}_box{(k_start + k)}"
        xml.append(f'{indent}<geom name="{gname}" type="box" pos="{pos_str}" quat="{quat_str}" size="{size_str}" mass="{mass}" group="3" contype="{contype}" conaffinity="{conaffinity}" rgba="{rgba}" solref="0.001"/>')

if __name__ == '__main__':
    generate_overall_native_xml() 