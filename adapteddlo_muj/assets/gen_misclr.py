# changed: mass of geom

import os
import numpy as np
import adapteddlo_muj.utils.transform_utils as T

class ObsGen:
    def __init__(
        self,
        obj_path=None,
        table_pos=[0., 0., 0.25],   # table top position
        h_gap=0.1,
        main_obs=True,
        more_obs=[[-0.4, 0.2, 0.25, 0.01, 0.01, 0.075]],
        add_marker=False,
    ):
        """
        Creates Table and Obstacles
        """
        self.h_gap = h_gap

        self.main_obs = main_obs

        self.obj_path = obj_path

        self.more_obs = more_obs
        self.add_marker = add_marker

        self._init_variables(table_pos)
        self._write_mainbody()

    def _init_variables(self, table_pos):

        if self.obj_path is None:
            self.obj_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "miscobj.xml"
            )
        
        # write to file
        # tab character
        self.t = "  "
        self.table_size = np.array([0.55, 0.8, 0.025])
        self.table_pos = table_pos  # table top position

        self.dens_val = 1000.
        self.obs_colli_type = [1, 0]
        self.obs_condim = 1
        self.table_colli_type = [1, 0]
        self.table_condim = 3
        
        self.curr_tab = 0

    def _write_table(self, f):
        f.write(
            self.curr_tab*self.t 
            + '<body name="table_body" pos="{} {} {}" quat="1 0 0 0">\n'.format(
                self.table_pos[0],
                self.table_pos[1],
                self.table_pos[2] - self.table_size[2],
            )
        )
        self.curr_tab += 1
        f.write(
            self.curr_tab*self.t
            + '<site name="tabletop" pos="0 0 {}" size="0.01 0.01 0.01" rgba="1 0 0 0" type="sphere" group="1" />\n'.format(
                self.table_size[2]
            )
        )
        f.write(
            (self.curr_tab)*self.t
            + '<geom friction="1 0.005 0.0001" group="0" name="table_collision" pos="0 0 0" size="{} {} {}" type="box" solref="0.001 1" contype="{}" conaffinity="{}" condim="{}"/>\n'.format(
                self.table_size[0],
                self.table_size[1],
                self.table_size[2],
                self.table_colli_type[0],
                self.table_colli_type[1],
                self.table_condim,
            )
        )
        self.curr_tab -= 1
        f.write(self.curr_tab*self.t + '</body>\n')

    def _write_box_obs(self, f, obs_id, obs_pos, obs_size, alpha_val=1):
    # def _write_box_obs(self, f, obs_id, obs_pos, obs_size, col_on=True, alpha_val=1):
        # if col_on:
        #     con_type = 1
        #     dens_val = 1000
        # else:
        #     con_type = 0
        #     dens_val = 1000
        
        # obs_pos[2] += obs_size[2]
        f.write(
            self.curr_tab*self.t 
            + '<body name="obstacle_{}" pos="{} {} {}" quat="1 0 0 0">\n'.format(
                obs_id,
                obs_pos[0],
                obs_pos[1],
                obs_pos[2],
            )
        )
        self.curr_tab += 1
        f.write(
            self.curr_tab*self.t
            + '<site name="obstacletop_{}" pos="0 0 {}" size="0.01 0.01 0.01" rgba="1 0 0 0" type="sphere" group="1" />\n'.format(
                obs_id,
                obs_size[2] * 2.
            )
        )
        f.write(
            self.curr_tab*self.t
            + '<site name="obstaclebtm_{}" pos="0 0 0" size="0.01 0.01 0.01" rgba="1 0 0 0" type="sphere" group="1" />\n'.format(
                obs_id,
            )
        )
        if obs_id == 0 and self.main_obs:   # main obs
            f.write(
                self.curr_tab*self.t
                + '<site name="goalmarker_{}" pos="0 0 0" size="0.01 0.01 0.01" rgba="1 0 0 0" type="sphere" group="1" />\n'.format(
                    obs_id
                )
            )
            obs_id += 0.1
            self._write_sidepillars(
                f=f,
                obs_id=obs_id,
                y_displace=self.h_gap/2,
                obs_size=obs_size,
                alpha_val=alpha_val,
            )
            obs_id += 0.1
            self._write_sidepillars(
                f=f,
                obs_id=obs_id,
                y_displace=-self.h_gap/2,
                obs_size=obs_size,
                alpha_val=alpha_val,
            )
            
        else:   # extra obs
            f.write(
                (self.curr_tab)*self.t
                + '<geom friction="1 0.005 0.0001" group="0" name="obscollision_{}" pos="0 0 0" size="{} {} {}" type="box" rgba="0.7 0.2 0.2 {}" density="{}" contype="{}" conaffinity="{}" condim="{}" solref="0.001 1"/>\n'.format(
                    obs_id,
                    obs_size[0],
                    obs_size[1],
                    obs_size[2],
                    alpha_val,
                    self.dens_val,
                    self.obs_colli_type[0],
                    self.obs_colli_type[1],
                    self.obs_condim
                )
            )
            
        self.curr_tab -= 1
        f.write(self.curr_tab*self.t + '</body>\n')

    def _write_mainbody(self):
        with open(self.obj_path, "w+") as f:
            f.write('<mujoco model="miscobj">\n')
            self.curr_tab += 1
            # triangular prism mesh
            f.write(self.curr_tab*self.t + "<worldbody>\n")
            self.curr_tab += 1

            self._write_table(f)
            # obs_pos = self.table_pos.copy()
            # obs_pos[0] -= 0.1
            # obs_pos[1] += 0.1
            obs_id = 0
            # self._write_box_obs(
            #     f, 
            #     obs_id=obs_id, 
            #     obs_pos=obs_pos,
            #     obs_size=[0.01, 0.01, 0.1]
            # )
            if self.add_marker:
                self._write_goalmarker(f)
            if self.more_obs is not None:
                for obs_data in self.more_obs:
                    obs_pos_i = obs_data[:3]
                    obs_size_i = obs_data[3:]
                    self._write_box_obs(
                        f, 
                        obs_id=obs_id, 
                        obs_pos=obs_pos_i,
                        obs_size=obs_size_i,
                        alpha_val=1.
                    )
                    obs_id += 1

            self.curr_tab -= 1
            f.write(self.curr_tab*self.t + "</worldbody>\n")
            self.curr_tab -= 1
            f.write(self.curr_tab*self.t + "</mujoco>\n")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Extra things ||~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _write_goalmarker(self, f):
        marker_pos = [-0.7, -0.15, 0.24]
        marker_size = [0.01, 0.01, 0.01]
        # obs_pos[2] += obs_size[2]
        f.write(
            self.curr_tab*self.t 
            + '<body name="goalmark" pos="{} {} {}" quat="1 0 0 0">\n'.format(
                marker_pos[0],
                marker_pos[1],
                marker_pos[2],
            )
        )
        self.curr_tab += 1
        f.write(
            self.curr_tab*self.t
            + '<site name="goalmark_top" pos="0 0 {}" size="0.01 0.01 0.01" rgba="1 0 0 0" type="sphere" group="1" />\n'.format(
                marker_size[2]
            )
        )
        f.write(
            self.curr_tab*self.t
            + '<site name="goalmark_btm" pos="0 0 {}" size="0.01 0.01 0.01" rgba="1 0 0 0" type="sphere" group="1" />\n'.format(
                - marker_size[2]
            )
        )
        f.write(
            (self.curr_tab)*self.t
            + '<geom friction="1 0.005 0.0001" group="0" name="goalmarker_vis" pos="0 0 0.08" size="0.009 0.009 0.08" type="box" rgba="0. 1. 0. 0.2" contype="0" conaffinity="0" solref="0.001 1"/>\n'
        )
        self.curr_tab -= 1
        f.write(self.curr_tab*self.t + '</body>\n')

    def _write_sidepillars(self,
        f,
        obs_id,
        y_displace,
        obs_size,
        alpha_val,
    ):
        obs_size2 = [0.025, 0.025, 0.007]
        obs_pos2 = [0., y_displace, obs_size[2]]
        f.write(
            self.curr_tab*self.t 
            + '<body name="obstacle_{}" pos="{} {} {}" quat="1 0 0 0">\n'.format(
                obs_id,
                obs_pos2[0],
                obs_pos2[1],
                obs_pos2[2],
            )
        )
        self.curr_tab += 1
        f.write(
            self.curr_tab*self.t
            + '<site name="obstacletop_{}" pos="0 0 {}" size="0.01 0.01 0.01" rgba="1 0 0 0" type="sphere" group="1" />\n'.format(
                obs_id,
                obs_size[2]
            )
        )
        # f.write(
        #     self.curr_tab*self.t
        #     + '<site name="obstacletop1_{}" pos="0 0 {}" size="0.001 0.001 1" rgba="1 0 0 1." type="box" group="1" />\n'.format(
        #         obs_id,
        #         obs_size2[2]
        #     )
        # )
        f.write(
            self.curr_tab*self.t
            + '<site name="obstaclebtm_{}" pos="0 0 {}" size="0.01 0.01 0.01" rgba="1 0 0 0" type="sphere" group="1" />\n'.format(
                obs_id,
                - obs_size[2]
            )
        )
        f.write(
            (self.curr_tab)*self.t
            + '<geom friction="1 0.005 0.0001" group="0" name="obscollision_{}" pos="0 0 0" size="{} {} {}" type="box" rgba="0.7 0.2 0.2 {}" density="{}" contype="{}" conaffinity="{}" condim="{}" solref="0.001 1"/>\n'.format(
                obs_id,
                obs_size[0],
                obs_size[1],
                obs_size[2],
                alpha_val,
                self.dens_val,
                self.obs_colli_type[0],
                self.obs_colli_type[1],
                self.obs_condim,
            )
        )
        ## top of one side pillar
        obs_id = str(obs_id) + 'T'
        obs_pos2 = [0., 0., obs_size[2] - obs_size2[2]]
        f.write(
            self.curr_tab*self.t 
            + '<body name="obstacle_{}" pos="{} {} {}" quat="1 0 0 0">\n'.format(
                obs_id,
                obs_pos2[0],
                obs_pos2[1],
                obs_pos2[2],
            )
        )
        self.curr_tab += 1
        f.write(
            self.curr_tab*self.t
            + '<site name="obstacletop_{}" pos="0 0 {}" size="0.01 0.01 0.01" rgba="1 0 0 0" type="sphere" group="1" />\n'.format(
                obs_id,
                obs_size2[2]
            )
        )
        # f.write(
        #     self.curr_tab*self.t
        #     + '<site name="obstacletop1_{}" pos="0 0 {}" size="0.001 0.001 1" rgba="1 0 0 1." type="box" group="1" />\n'.format(
        #         obs_id,
        #         obs_size2[2]
        #     )
        # )
        f.write(
            self.curr_tab*self.t
            + '<site name="obstaclebtm_{}" pos="0 0 {}" size="0.01 0.01 0.01" rgba="1 0 0 0" type="sphere" group="1" />\n'.format(
                obs_id,
                - obs_size2[2]
            )
        )
        f.write(
            (self.curr_tab)*self.t
            + '<geom friction="1 0.005 0.0001" group="0" name="obscollision_{}" pos="0 0 0" size="{} {} {}" type="box" rgba="0.7 0.2 0.2 {}" density="{}" contype="{}" conaffinity="{}" condim="{}" solref="0.001 1"/>\n'.format(
                obs_id,
                obs_size2[0],
                obs_size2[1],
                obs_size2[2],
                alpha_val,
                self.dens_val,
                self.obs_colli_type[0],
                self.obs_colli_type[1],
                self.obs_condim
            )
        )
        
        self.curr_tab -= 1
        f.write(self.curr_tab*self.t + '</body>\n')
        ## end of top of one pillar
        self.curr_tab -= 1
        f.write(self.curr_tab*self.t + '</body>\n')

if __name__ == "__main__":
    ObsGen()