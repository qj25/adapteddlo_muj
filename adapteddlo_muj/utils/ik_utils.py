import os
import mujoco
import numpy as np
from der_muj4.utils.xml_utils import XMLWrapper
import der_muj4.utils.mjc2_utils as mjc2
import der_muj4.utils.transform_utils as T

class ik_denso:
    def __init__(self, eef_site_name="eef_site", init_qpos=np.zeros(6)):
        xml, arm_xml = self._get_xmlstr()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.eef_site_name = eef_site_name
        self.eef_site_idx = mjc2.obj_name2id(
            self.model,
            "site",
            self.eef_site_name
        )

        #Init variables.
        jacp = np.zeros((3, self.model.nv)) #translation jacobian
        jacr = np.zeros((3, self.model.nv)) #rotational jacobian
        step_size = 0.5
        tol = 0.01
        alpha = 0.5
        self.prev_qpos = init_qpos
        damping = 0.15

        self.lmik = LevenbegMarquardtIK(
            self.model, self.data, step_size, tol,
            alpha, jacp, jacr, damping
        )
    
    def calc_ik(self, goal, prev_qpos=None):
        if prev_qpos is None:
            prev_qpos = self.prev_qpos.copy()
        self.prev_qpos = goal.copy()
        #calculate the qpos
        return self.lmik.calculate(goal, prev_qpos, self.eef_site_idx) 

    def _get_xmlstr(self):
        # load model
        # update rope model
        world_base_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets/world.xml"
        )
        robot_path = os.path.join(
            os.path.dirname(world_base_path),
            "densovs060/densovs060.xml"
        )
        self.xml = XMLWrapper(world_base_path)
        robotarm = XMLWrapper(robot_path)
        self.xml.merge_multiple(
            robotarm, ["worldbody", "asset"]
        )
        xml_string = self.xml.get_xml_string()
        return xml_string, robotarm

#Levenberg-Marquardt method
class LevenbegMarquardtIK:
    def __init__(self, model, data, step_size, tol, alpha, jacp, jacr, damping):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = jacp
        self.jacr = jacr
        self.damping = damping
    
    def check_joint_limits(self, q):
        """Check if the joints is under or above its limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], 
                       min(q[i], self.model.jnt_range[i][1]))

    #Levenberg-Marquardt pseudocode implementation
    def calculate(self, goal_pose, init_q, site_id):
        """
        Calculate the desire joints angles for goal
        by simulating another arm in MuJoCo.
        calculate considers bot pos and quat.
        """
        self.data.qpos = init_q
        mujoco.mj_forward(self.model, self.data)
        pose_error, eef_pose = self.calc_pose_error(goal_pose, site_id)
        

        i_compute = 0
        i_c_lim = 10000
        while not check_prox(goal_pose, eef_pose):
            i_compute += 1
            #calculate jacobian
            mujoco.mj_jacSite(
                self.model,
                self.data,
                self.jacp,
                self.jacr,
                site_id
            )
            J_full = np.vstack([self.jacp, self.jacr])[:, :6]
            delta_q = (
                # np.dot(J_full.T, pose_error)
                np.dot(np.linalg.pinv(J_full), pose_error)
                # np.dot(np.linalg.norm(J_full), pose_error)
            )
            #compute next step
            self.data.qpos += self.step_size * delta_q
            #check limits
            self.check_joint_limits(self.data.qpos)
            #compute forward kinematics
            mujoco.mj_forward(self.model, self.data)

            #calculate new error
            pose_error, eef_pose = self.calc_pose_error(goal_pose, site_id)

            # end loop if no soln found
            if i_compute > i_c_lim:
                return None

        # print(f"i_compute = {i_compute}")
        # print(f"final_lmik_pos = {eef_pos}")
        # print(f"final_lmik_quat = {eef_quat}")

        return self.data.qpos.copy()
    
    def calculate2(self, goal_pos, init_q, site_id):
        """
        Calculate the desire joints angles for goal
        by simulating another arm in MuJoCo.
        calculate2 only considers pos and not quat.
        """
        self.data.qpos = init_q
        mujoco.mj_forward(self.model, self.data)
        current_pose = np.array(
            self.data.site_xpos[site_id]
        )
        error = np.subtract(goal_pos, current_pose)

        while (np.linalg.norm(error) >= self.tol):
            #calculate jacobian
            mujoco.mj_jacSite(
                self.model,
                self.data,
                self.jacp,
                self.jacr,
                site_id
            )
            #calculate delta of joint q
            n = self.jacp.shape[1]
            I = np.identity(n)
            product = self.jacp.T @ self.jacp + self.damping * I
            
            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ self.jacp.T
            else:
                j_inv = np.linalg.inv(product) @ self.jacp.T
            
            delta_q = j_inv @ error
            #compute next step
            self.data.qpos += self.step_size * delta_q
            #check limits
            self.check_joint_limits(self.data.qpos)
            #compute forward kinematics
            mujoco.mj_forward(self.model, self.data) 
            #calculate new error
            error = np.subtract(
                goal_pos, 
                np.array(self.data.site_xpos[site_id])
            )  

        # print(f"final_lmik_pos = {eef_pos}")
        # print(f"final_lmik_quat = {eef_quat}")

        return self.data.qpos.copy()
    
    def calc_pose_error(self, goal, site_id):
        eef_pos = np.array(
            self.data.site_xpos[site_id]
        )
        eefmat = np.array(
            self.data.site_xmat[site_id].reshape((3, 3))
        )
        # print('here')
        # print(self._state['eef_pos'])
        eef_quat = T.mat2quat(eefmat)

        # error
        e_pos = goal[:3] - eef_pos
        e_ori = T.quat_error(eef_quat, goal[3:])
        # e_ori = np.zeros(3)
        # overall
        pose_error = np.concatenate((e_pos, e_ori))
        eef_pose = np.concatenate((eef_pos, eef_quat))
        return pose_error, eef_pose
    
def check_prox(goal, eef_pose, x_req=1.):
    eef_pos = eef_pose[:3]
    eef_quat = eef_pose[3:]
    return (
        T.check_proximity(
            eef_pos, goal[:3],
            d_tol=5e-6/x_req
        )
        and T.check_proximity(
            eef_quat, goal[3:],
            check_mode='quat',
            d_tol=5e-4/x_req
        )
    )