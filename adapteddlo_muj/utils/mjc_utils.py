import numpy as np
import mujoco
import adapteddlo_muj.utils.transform_utils as T

# object indicator in mujoco
MJ_SITE_OBJ = 6  # `site` objec
MJ_BODY_OBJ = 1  # `body` object
MJ_GEOM_OBJ = 5  # `geom` object
# geom types
MJ_CAPSULE = 3
MJ_CYLINDER = 5
MJ_BOX = 6
MJ_MESH = 7

"""
Notes:
- force sensor senses the forces on the body that it is defined on
    INCLUDING its weight. 
    - to bypass this issue, create a new body with 0 mass/geom. 
        (works for both weld and fixed)
- how does weld affect torque sensor:
    A-A2 weld=0.3 B-0.2-B2
    if 1N applied at B2, torq at B is 0.2Nm
    torqueA2sensor = torqB*2 (doubled as torque felt at weld is double) 
                    + torqaboutA2fromforceatB = 0.2Nm*2 + 0.3Nm
    - to bypass this issue of double torq,
        weld body at [0,0,0] relpos (so force at B has no contribution)
        and divide measured torque by 2.
"""

def get_contact_force(mj_model, mj_data, body_name, frame_pos, frame_quat):
    """Get the force acting on a body, with respect to a frame.
    Note that mj_rnePostConstraint should be called before this function
    to update the simulator state.

    :param str body_name: Body name in mujoco xml model.
    :return: force:torque format.
    :rtype: np.array(6)

    """
    bodyId = mujoco.mj_name2id(mj_model, MJ_BODY_OBJ, body_name)
    force_com = mj_data.cfrc_ext[bodyId, :]
    # contact force frame
    # orientation is aligned with world frame
    qf = np.array([1, 0, 0, 0.0])
    # position of origin in the world frame
    body_rootid = mj_model.body_rootid[bodyId]
    pf = mj_data.subtree_com[body_rootid, :]

    # inverse com frame
    pf_inv, qf_inv = np.zeros(3), np.zeros(4)
    mujoco.mju_negPose(pf_inv, qf_inv, pf, qf)
    # T^com_target
    p_ct, q_ct = np.zeros(3), np.zeros(4)
    mujoco.mju_mulPose(
        p_ct,
        q_ct,
        pf_inv,
        qf_inv,
        frame_pos.astype(np.float64),
        frame_quat.astype(np.float64),
    )
    # q_ct -> mat
    mat_ct = np.zeros(9)
    mujoco.mju_quat2Mat(mat_ct, q_ct)

    # transform to desired frame
    trn_force = force_com.copy()
    mujoco.mju_transformSpatial(
        trn_force, force_com, 1, p_ct, np.zeros(3), mat_ct
    )

    # reverse order to get force:torque format
    return np.concatenate((trn_force[3:], trn_force[:3]))

def get_sensor_id(mj_model, site_name):
    type_id = mujoco.mju_str2Type("site")
    sensorsite_id = mujoco.mj_name2id(mj_model,type_id,site_name)
    sensor_ids = []
    for i in range(mj_model.nsensor):
        if mj_model.sensor_objtype[i] == mujoco.mjtObj.mjOBJ_SITE:
            site_id = mj_model.sensor_objid[i]
            if sensorsite_id == site_id:
                sensor_ids.append(i)
    return sensor_ids

def get_sensor_force(mj_model, mj_data, sensor_id, body_name, frame_pos, frame_quat):
    ## IMPORTANT: sensordata give you the force wrt sensorsite.
    ## Section (A) in this function changes that to world frame.

    ## DOESNT WORK! takes subtree for finding CoM but is not necessary!
    ## use get_sensor_force2 instead
    """Get the force acting on a body, with respect to a frame.
    Note that mj_rnePostConstraint should be called before this function
    to update the simulator state.

    :param str body_name: Body name in mujoco xml model.
    :return: force:torque format.
    :rtype: np.array(6)

    """
    # In the XML, define torque, then force sensor
    bodyId = mujoco.mj_name2id(mj_model, MJ_BODY_OBJ, body_name)
    force_com = np.array([])
    for i in range(len(sensor_id)):
        dim_sensor = mj_model.sensor_dim[sensor_id[i]]
        force_com = np.concatenate((
            force_com,
            mj_data.sensordata[
                mj_model.sensor_adr[sensor_id[i]]
                :mj_model.sensor_adr[sensor_id[i]] + dim_sensor
            ]
        ))
    # force_com = mj_data.sensordata[sensor_id*6:sensor_id*6+6]
    # print(f"force_com={force_com}")
    # contact force frame
    # orientation is aligned with world frame
    qf = np.array([1, 0, 0, 0.0])
    # position of origin in the world frame
    body_rootid = mj_model.body_rootid[bodyId]
    pf = mj_data.subtree_com[body_rootid, :]
    # pf = np.zeros(3)
    # print(qf)
    # input(frame_quat)
    # inverse com frame
    pf_inv, qf_inv = np.zeros(3), np.zeros(4)
    mujoco.mju_negPose(pf_inv, qf_inv, pf, qf)
    # T^com_target
    p_ct, q_ct = np.zeros(3), np.zeros(4)
    mujoco.mju_mulPose(
        p_ct,
        q_ct,
        pf_inv,
        qf_inv,
        frame_pos.astype(np.float64),
        frame_quat.astype(np.float64),
    )
    # q_ct -> mat
    # Section (A)
    p_ct_n, q_ct_n = np.zeros(3), np.zeros(4)
    mujoco.mju_negPose(p_ct_n, q_ct_n, p_ct, q_ct)

    mat_ct = np.zeros(9)
    mujoco.mju_quat2Mat(mat_ct, q_ct_n)

    # transform to desired frame
    trn_force = force_com.copy()
    # print(f"trn_force = {p_ct_n}")
    mujoco.mju_transformSpatial(
        trn_force, force_com, 1, p_ct_n, np.zeros(3), mat_ct
    )
    # print(f"trn_force = {trn_force}")
    # reverse order to get force:torque format
    return np.concatenate((trn_force[3:], trn_force[:3]))

def fix_ftsensor_weld(f_raw, t_raw, dist_weld):
    # See notes above. fixes the error with doubled torque
    # and non-zero weld distance (dist_weld is from A to B)
    # print(np.cross(dist_weld,f_raw))
    return (t_raw - np.cross(dist_weld,f_raw)) / 2.0

def get_sensor_force2(mj_model, mj_data, sensor_id, sensor_site_name):
    """Get the force and torque in world frame from a sensor site.
    
    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        sensor_id: List of sensor IDs to read from
        sensor_site_name: Name of the sensor site in the XML model
    
    Returns:
        np.array(6): Force and torque in world frame [force_x, force_y, force_z, torque_x, torque_y, torque_z]
    """
    # Get sensor site ID and its pose in world frame
    site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, sensor_site_name)
    if site_id == -1:
        raise ValueError(f"Site {sensor_site_name} not found in model")
    body_id = mj_model.site_bodyid[site_id]

    # Get sensor site pose in world frame
    site_pos = mj_data.site_xpos[site_id].copy()  # Position in world frame
    site_pos = np.zeros(3)
    site_quat = T.mat2quat(mj_data.site_xmat[site_id].copy())  # Orientation in world frame
    body_pos = mj_data.xpos[body_id].copy()  # Position in world frame
    body_quat = mj_data.xquat[body_id].copy()  # Position in world frame
    body_pos = np.zeros(3)

    # Get raw sensor data
    force_com = np.array([])
    for i in range(len(sensor_id)):
        dim_sensor = mj_model.sensor_dim[sensor_id[i]]
        force_com = np.concatenate((
            force_com,
            mj_data.sensordata[
                mj_model.sensor_adr[sensor_id[i]]
                :mj_model.sensor_adr[sensor_id[i]] + dim_sensor
            ]
        ))
    # print(f"bodyquat = {mj_data.xquat[body_id]}")
    # print(f"bodymat = {mj_data.xmat[body_id]}")
    # Transform force and torque to world frame
    p_ct_n, q_ct_n = np.zeros(3), np.zeros(4)
    mujoco.mju_negPose(p_ct_n, q_ct_n, body_pos, body_quat)
    
    # print(f"{sensor_site_name}_val_pre = {force_com}")
    # Convert quaternion to rotation matrix
    invmat = np.zeros(9)
    mujoco.mju_quat2Mat(invmat, q_ct_n)
    # print(p_ct_n)
    # print(body_quat)
    # input()
    trn_force = force_com.copy()
    mujoco.mju_transformSpatial(
        trn_force, 
        force_com, 
        1,  # 1 for force, 0 for velocity
        p_ct_n,  # Position of the frame
        np.zeros(3),  # Velocity of the frame (zero for static frame)
        invmat  # Rotation matrix
    )
    # print(f"{sensor_site_name}_val_post = {trn_force}")
    
    # Return in force:torque format
    return np.concatenate((trn_force[3:], trn_force[:3]))

class MjSimWrapper:
    """A simple wrapper to remove redundancy in forward() and step() calls
    Typically, we call forward to update kinematic states of the simulation, then set the control
    sim.data.ctrl[:], finally call step
    """

    def __init__(self, model, data) -> None:
        # self.sim = sim
        self.model = model
        self.data = data
        self._is_forwarded_current_step = False

    def forward(self):
        if not self._is_forwarded_current_step:
            mujoco.mj_step1(self.model, self.data)
            mujoco.mj_rnePostConstraint(self.model, self.data)
            self._is_forwarded_current_step = True

    def step(self):
        self.forward()
        mujoco.mj_step2(self.model, self.data)
        self._is_forwarded_current_step = False

    def get_state(self):
        return self.sim.get_state()

    # def reset(self):
    #     self._is_forwarded_current_step = False
    #     return self.sim.reset()

    # @property
    # def model(self):
        # return self.sim.model

    # @property
    # def data(self):
        # return self.sim.data

# class MjSimPluginWrapper(MjSimWrapper):
#     """A simple wrapper to remove redundancy in forward() and step() calls
#     Typically, we call forward to update kinematic states of the simulation, then set the control
#     sim.data.ctrl[:], finally call step
#     """

#     def __init__(self, xml) -> None:
#         model = mujoco.MjModel.from_xml_string(xml)
#         data = mujoco.MjData(model)
#         super().__init__(model, data)