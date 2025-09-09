import os 
import mujoco
import mujoco_viewer
import numpy as np
import adapteddlo_muj.utils.mjc2_utils as mjc2
from adapteddlo_muj.utils.xml_utils import XMLWrapper
# from adapteddlo_muj.controllers.wire_plugin.WireStandalone import WireStandalone
from adapteddlo_muj.utils.mjc_utils import MjSimWrapper
# from adapteddlo_muj.utils.real2sim_utils import compute_wire_frames
from adapteddlo_muj.assets.genrope.gen_overall_native_xml import generate_overall_native_xml
from adapteddlo_muj.utils.dlo_utils import nearest_point_on_wire, kb_similarity_metric

def smoothing_compare(init_pos, curr_pos, prev_E, curr_E, m_p):
    # compare (position error) * m_p + (energy decrease)
    # if < 0: continue smoothing (return True)
    pos_diff = np.zeros(len(curr_pos))
    # pos_diff = np.linalg.norm((curr_pos - init_pos),axis=1)
    for i in range(len(pos_diff)):
        _,_,pos_diff[i],_ = nearest_point_on_wire(init_pos, curr_pos[i])
    poserr_norm = np.mean(pos_diff)
    reldec_val = (
        poserr_norm * m_p
        + (curr_E - prev_E)
    )
    print(f"poserr_norm = {poserr_norm}")
    print(f"E_diff = {(curr_E-prev_E)}")
    print(f"Smoothing compare val = {reldec_val}")
    # print(curr_E-prev_E)
    if reldec_val < 10000: return True
    else: return False 

def smoothing_compare2(pos_diff, prev_E, curr_E, m_p):
    # input uses pos_diff
    # compare (position error) * m_p + (energy decrease)
    poserr_norm = np.mean(pos_diff)
    reldec_val = (
        poserr_norm * m_p
        + (curr_E - prev_E)
    )
    print(f"poserr_norm = {poserr_norm}")
    print(f"E_diff = {(curr_E-prev_E)}")
    print(f"Smoothing compare val = {reldec_val}")
    # print(curr_E-prev_E)
    if reldec_val < 10000: return True
    else: return False 


def get_initialsmooth(
    wire_pos, mass_per_length, n_pieces, thickness, j_damp,
    stiff_vals, xml_path, do_render, n_maxsmoothsteps,
    n_avg=1,
    m_p=1.0e-2,
    f_multi=5.0,
    reset_freq=1,
    pos_tol=1e-7,
    someStiff=0
):
    # Smoothing the wire in 0 gravity based on energy tradeoff and velocity reset at each step
    # Generate wire xml
    # includes conversion of wire xpos to main body pose and subsequent connected quaternions
    # Compute total arc length
    segment_lengths = np.linalg.norm(np.diff(wire_pos, axis=0), axis=1)
    total_length = np.sum(segment_lengths)

    # Mass per unit length
    mass = mass_per_length * total_length
    # init_pos = np.array([0.0, 0.0, 0.5])
    # init_quat = np.array([1.0, 0.0, 0.0, 0.0])
    rgba_wire = "0.1 0.0533333 0.673333 1"
    # rgba_pipe = "1.0 1.0 1.0 0.1"
    generate_overall_native_xml(
        n_pieces=n_pieces-1,
        thickness=thickness,
        mass=mass,
        j_damp=j_damp,
        con_val=(1,0),
        stiff_bend=stiff_vals[1],
        stiff_twist=stiff_vals[0],
        wire_pos=wire_pos,
        xml_path=xml_path,
        rgba=rgba_wire,
        someStiff=someStiff
    )
    xml = XMLWrapper(xml_path)

    # # Load MuJoCo model and data
    # model = mujoco.MjModel.from_xml_path(xml_path)
    xml_string = xml.get_xml_string()
    model = mujoco.MjModel.from_xml_string(xml_string)
    mujoco.mj_saveLastXML(xml_path,model)
    data = mujoco.MjData(model)
    # model.opt.gravity[-1] = 1.0
    # model.opt.gravity[-1] = -9.81
    model.opt.gravity[-1] = 0.0

    known_body_name = "B_first"
    # plgn_instance = model.body_plugin[
    #     mjc2.obj_name2id(model, "body", known_body_name)
    # ]
    # start = model.plugin_stateadr[plgn_instance]
    # r_len = total_length
    r_pieces = len(wire_pos) - 1
    vec_bodyid = np.zeros(r_pieces, dtype=int)
    for i in range(r_pieces):
        if i == 0:
            i_name = 'first'
        elif i == r_pieces-1:
            i_name = 'last'
        else:
            i_name = str(i)
        vec_bodyid[i] = mjc2.obj_name2id(
            model,"body",'B_' + i_name
        )
    vec_bodyid_full = np.concatenate((
        vec_bodyid, [mjc2.obj_name2id(
            model,"body",'B_last2'
        )]
    ))

    sim = MjSimWrapper(model, data)
    if do_render:
        viewer = mujoco_viewer.MujocoViewer(model, data)
        viewer.vopt.geomgroup[3] ^= 1
        dist=0.5
        azi=90.0
        elev=0.0
        lookat=np.array([0.0, 0.0, 0.0])
        viewer.cam.distance = dist
        viewer.cam.azimuth = azi
        viewer.cam.elevation = elev
        viewer.cam.lookat = lookat
        viewer.render()
        viewer._paused = True

    known_body_name = "B_first"
    plgn_instance = model.body_plugin[
        mjc2.obj_name2id(model, "body", known_body_name)
    ]
    start = model.plugin_stateadr[plgn_instance]

    def get_storedtorque():
        if plgn_instance == model.nplugin - 1:
            # Last plugin - use remaining state
            stored_torques = data.plugin_state[start:start+model.nv]
            E_total = data.plugin_state[-1]
        else:
            # Not last plugin - use next plugin's start as end
            # stored_torques = data.plugin_state[start:model.plugin_stateadr[plgn_instance+1]-1]
            stored_torques = data.plugin_state[start:start+model.nv]
            E_total = data.plugin_state[start+model.nv]
        return stored_torques, E_total

    sim.forward()
    sim.step()
    if do_render:
        viewer.render()
    sim.forward()
    init_xpos = data.xpos[vec_bodyid_full].copy()
    kb_avg_mag = kb_similarity_metric(positions=init_xpos,n=2)

    n_count = 0
    prev_xpos = np.zeros_like(data.xpos[vec_bodyid_full])
    prev_E = 0.0
    curr_xpos = np.zeros_like(data.xpos[vec_bodyid_full])
    curr_E = 0.0

    for i in range(n_avg):
        sim.step()
        sim.forward()
        if do_render:
            viewer.render()
        if i%reset_freq==0:
            data.qvel[:] = np.zeros(len(data.qvel[:]))
            sim.forward()

        stored_torques, E_total = get_storedtorque()
        prev_xpos += data.xpos[vec_bodyid_full].copy()
        prev_E += E_total
        n_count += 1

    prev_xpos /= n_count
    prev_E /= n_count
    n_count = 0

    for i in range(n_maxsmoothsteps):
        # if i == 10000: data.eq_active[0] = 0
        # print(i)
        sim.step()
        sim.forward()
        if do_render:
            viewer.render()
        if i%reset_freq==0:
            data.qvel[:] = np.zeros(len(data.qvel[:]))
            sim.forward()

        stored_torques, E_total = get_storedtorque()

        # Apply restoring force
        curr_pos = data.xpos[vec_bodyid_full]
        pos_diff = np.zeros(len(curr_pos))
        nearest_pt = np.zeros((len(curr_pos),3))
        for i in range(len(pos_diff)):
            nearest_pt[i],_,pos_diff[i],_ = nearest_point_on_wire(init_xpos, curr_pos[i])
        restoring_vec = nearest_pt - curr_pos
        print(f"kb_avg_mag = {kb_avg_mag}")
        restoring_vec = np.multiply(restoring_vec, kb_avg_mag[:, None])
        restoring_vec *= f_multi
        data.xfrc_applied[vec_bodyid_full, :3] = restoring_vec.copy()

        # print("qfrc_passive main:",data.qfrc_passive.reshape((-1,3)))
        # print("stored_torques:",stored_torques.reshape((-1,3)))
        # print("total energy:",E_total)

        ## Smoothing compare stopping condition |=============================
        # if n_count > (n_avg-1):
        #     curr_xpos /= n_count
        #     curr_E /= n_count
        #     if not smoothing_compare2(
        #         pos_diff=pos_diff,
        #         prev_E=prev_E, curr_E=E_total, m_p=m_p
        #     ):
        #         # continue
        #         print(f"Total steps = {i}")
        #         if do_render:
        #             viewer._paused = True
        #         break
        #     prev_xpos = curr_xpos.copy()
        #     prev_E = curr_E
        #     curr_xpos = np.zeros_like(data.xpos[vec_bodyid_full])
        #     curr_E = 0.0
        #     n_count = 0
        # else:
        #     curr_xpos += data.xpos[vec_bodyid_full].copy()
        #     curr_E += E_total
        #     n_count += 1
        ## |==================================================================

        pos_change = np.mean(
            np.linalg.norm(prev_xpos-data.xpos[vec_bodyid_full],axis=1)
        )
        print(f"pos_change = {pos_change}")
        if pos_change < pos_tol:
            break
        prev_xpos = data.xpos[vec_bodyid_full].copy()
    # pos_diff = np.linalg.norm((curr_pos - init_pos),axis=1)
        
        # if viewer._paused:
            # break
        # input(data.xpos[vec_bodyid_full].copy().flatten())
    npos = data.xpos[vec_bodyid_full].copy().flatten()
    return npos.reshape((-1,3))