import numpy as np
import mujoco

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

def get_sensor_id(mj_model, body_name):
    type_id = mujoco.mju_str2Type("body")
    body_id = mujoco.mj_name2id(mj_model,type_id,body_name)
    sensor_ids = []
    for i in range(mj_model.nsensor):
        if mj_model.sensor_objtype[i] == mujoco.mjtObj.mjOBJ_SITE:
            site_id = mj_model.sensor_objid[i]
            site_body_id = mj_model.site_bodyid[site_id]
            if site_body_id == body_id:
                sensor_ids.append(i)
    return sensor_ids

def get_sensor_force(mj_model, mj_data, sensor_id, body_name, frame_pos, frame_quat):
    ## IMPORTANT: sensordata give you the force wrt sensorsite.
    ## Section (A) in this function changes that to world frame.
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