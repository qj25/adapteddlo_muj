import numpy as np
import mujoco

import time

# for i in range(26):
#     print(f"id={i}:  type={mujoco.mju_type2Str(i)}")
    
def obj_id2name(model, type_str, obj_id):
    type_id = mujoco.mju_str2Type(type_str)
    return mujoco.mj_id2name(model,type_id,obj_id)

def obj_name2id(model, type_str, obj_name):
    type_id = mujoco.mju_str2Type(type_str)
    obj_id = mujoco.mj_name2id(model,type_id,obj_name)
    if obj_id < 0:
        print(f"{type_str} of name '{obj_name}' does not exist.")
        input()
    return obj_id

def obj_getvel(model, data, type_str, obj_id):
    type_id = mujoco.mju_str2Type(type_str)
    obj_vel = np.zeros(6)
    mujoco.mj_objectVelocity(
        model,
        data,
        type_id,
        obj_id,
        obj_vel,
        0
    )
    return obj_vel

def get_site_jac(model, data, site_id):
    """Return the Jacobian' translational component of the end-effector of
    the corresponding site id.
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    jac = np.vstack([jacp, jacr])

    return jac

def get_fullM(model, data):
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    return M

def change_eq_pos(model, pos1, pos2, obj1_name):
    # obj1 is a 'body' object
    obj1_id = obj_name2id(model, 'body', obj1_name)
    for i in range(len(model.eq_type)):
        if obj1_id == model.eq_obj1id[i]:
            eq_id = i
            break
    model.eq_data[eq_id][:6] = np.concatenate((pos1,pos2))

def update_weld_relpose(model, data, weld_name):
    """Update the relative pose of a weld constraint to match current body positions.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        weld_name (str): Name of the weld constraint in the XML model
    """
    # Update latest position
    mujoco.mj_fwdPosition(model, data)
    # Get weld constraint ID
    weld_id = obj_name2id(model, "equality", weld_name)
    if weld_id == -1:
        raise ValueError(f"Weld constraint {weld_name} not found in model")
    
    # Get the two bodies involved in the weld
    body1_id = model.eq_obj1id[weld_id]
    body2_id = model.eq_obj2id[weld_id]
    
    # Get current positions and orientations
    body1_pos = data.xpos[body1_id].copy()
    body1_quat = data.xquat[body1_id].copy()
    body2_pos = data.xpos[body2_id].copy()
    body2_quat = data.xquat[body2_id].copy()
    inv_body1_quat = np.zeros(4)

    mujoco.mju_negQuat(inv_body1_quat, body1_quat)
    
    # Compute relative position in body1's frame
    rel_pos = np.zeros(3)
    mujoco.mju_rotVecQuat(
        rel_pos,
        body2_pos - body1_pos,
        inv_body1_quat
    )
    # input(rel_pos)
    # if abs(rel_pos[0] + 10.5) < 1e-6: 
        # rel_pos[0] -= 0.02
    # input(rel_pos)
    
    # Compute relative orientation (body2 relative to body1)
    rel_quat = np.zeros(4)
    mujoco.mju_mulQuat(
        rel_quat,
        inv_body1_quat,
        body2_quat
    )
    # Update the weld constraint data
    model.eq_data[weld_id][6:10] = rel_quat  # Update quaternion
    model.eq_data[weld_id][3:6] = rel_pos   # Update position
    # model.eq_data[weld_id][0:3] = np.array([[0.5,0,0.5]])

def pause_sim(viewer, run_t):
    viewer._paused = True
    pt_start = time.time()
    while viewer._paused:
        viewer.render()
    pt_end = time.time()
    return run_t + (pt_end-pt_start)

class viewer_wrapper:
    def __init__(self, model, data, env):
        self.viewer = mujoco.viewer.launch_passive(
            model, data, key_callback=self.key_callback
        )
        self._paused = False
        self.env = env

    def key_callback(self, keycode):
        if chr(keycode) == ' ':
            self._paused = not self._paused
        if chr(keycode) == chr(256):
            print('hi')
            # self.viewer.close()
            # self.env.close()
            exit(0)

    def sync(self):
        if not self._paused:
            self.viewer.sync()
        while self._paused:
            time.sleep(1)