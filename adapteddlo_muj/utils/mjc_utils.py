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

def get_sensor_force(mj_model, mj_data, body_name, frame_pos, frame_quat):
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
    force_com = mj_data.sensordata
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
