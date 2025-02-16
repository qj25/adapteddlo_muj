import os
import numpy as np
import adapteddlo_muj.utils.transform_utils as T

class GenKin_O_weld2:
    def __init__(
        self,
        r_len=0.9,
        r_thickness=0.02,
        r_pieces=20,
        j_stiff=0.0,
        j_damp=0.7,
        r_mass=None,
        init_pos=[1.3, 0.0, 0.13],
        init_quat=[1.0, 0., 0., 0.],
        coll_on=False,
        d_small=0.,
        rope_type="capsule",
        attach_bodyname="eef_body",
        vis_subcyl=True,
        obj_path=None,
    ):
        """
        connected by kinematic chain

        z-axis pointing from start section towards other sections.
        Displacement of other sections in the local positive z-direction.
        local axes:
        x - blue
        y - yellow
        params:
        - init_pos: starting position for base of rope/box
        - init_angle: rotation angle of the box about the z axis 
        """

        self.r_len = r_len
        self.r_thickness = r_thickness
        self.r_pieces = r_pieces
        self.j_stiff = j_stiff
        self.j_damp = j_damp
        self.r_mass = r_mass
        self.init_pos = init_pos
        self.init_quat = T.quat_multiply(init_quat,np.array([0., 0., 1., 0.]))
        if coll_on:
            self.con_data = [1, 1]
        else:
            self.con_data = [0, 0]

        self.grow_dirn = np.dot(np.array([1.,0,0]),T.quat2mat(init_quat))
        # self.init_angle = init_angle
        self.d_small = d_small
        self.rope_type = rope_type
        self.attach_bodyname = attach_bodyname
        self.vis_subcyl = vis_subcyl
        self.obj_path = obj_path

        if self.vis_subcyl:
            self.subcyl_alpha = 0.3
        else:
            self.subcyl_alpha = 0.0
        rgb_vals = np.array([30,16,202])/300
        self.rgba_linkgeom = f'{rgb_vals[0]} {rgb_vals[1]} {rgb_vals[2]} 1'

        self._init_variables()
        self._write_mainbody()
        self._write_anchorbox()
        self._write_weldweight()

    def _init_variables(self):
        self.max_pieces = self.r_len / self.r_thickness
        
        if self.r_pieces >= self.max_pieces:
            raise ValueError(
                'Too many sections for requested thickness.\n'
                + 'No.of must be strictly less than Max.\n'
                + f'No.of sections = {self.r_pieces}.\n'
                + f'Max sections = {self.max_pieces}.'
            )

        # check if rope is to be attached to box
        # self.init_quat = np.array([0.5, 0.5, -0.5, 0.5])
        self.attach_pos = self.init_pos.copy()
        self.solref_val = "0.001 1"

        # mass stuff
        if self.r_mass is not None:
            self.mass_link = self.r_mass / self.r_pieces
        else:
            self.mass_link = None

        # misc
        self.displace_link = self.r_len / self.r_pieces
        self.subcyl_len = 0.05
        self.box_size = [
                self.r_thickness, self.r_thickness,
                2*self.r_thickness,
            ]

        if self.obj_path is None:
            self.obj_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "derrope1dkin.xml"
            )
        self.obj_path2 = os.path.join(
            os.path.dirname(self.obj_path),
            "anchorbox.xml"
        )
        self.obj_path3 = os.path.join(
            os.path.dirname(self.obj_path),
            "weldweight.xml"
        )
        
        # write to file
        # tab character
        self.t = "  "
        self.cap_size = np.zeros(2)
        self.cap_size[0] = self.r_thickness / 2
        if self.rope_type == "capsule":
            self.cap_size[1] = (self.r_len / self.r_pieces - self.r_thickness) / 2
            self.cap_size[1] = (self.r_len / self.r_pieces) / 2
        elif self.rope_type == "cylinder":
            self.cap_size[1] = (self.r_len / self.r_pieces - self.d_small) / 2
        self.cap_size[1] -= self.d_small
        self.joint_size = self.cap_size[0]
        if self.cap_size[0] > 0.8 * self.cap_size[1]:
            print(f"Warning: thickness too large -ratio- {self.cap_size[0]/self.cap_size[1]}")
            print("Consider reducing to prevent unwanted collisions")
            input()
        # body_names
        self.str_names = list(str(x_n) for x_n in range(self.r_pieces))
        self.str_names[0] = 'first'
        self.str_names[-1] = 'last'
        
        self.curr_tab = 0

    def _write_mainbody(self):
        with open(self.obj_path, "w+") as f:
            f.write('<mujoco model="stiff-rope">\n')
            self.curr_tab += 1
            f.write(self.curr_tab*self.t + "<worldbody>\n")
            self.curr_tab += 1
            f.write(
                self.curr_tab*self.t 
                + '<body name="stiffrope" pos="{} {} {}" quat="{} {} {} {}">\n'.format(
                    self.attach_pos[0],
                    self.attach_pos[1],
                    self.attach_pos[2],
                    self.init_quat[0],
                    self.init_quat[1],
                    self.init_quat[2],
                    self.init_quat[3]
                )
            )
            self.curr_tab += 1
            f.write(self.curr_tab*self.t + '<freejoint name="freejoint_A"/>\n')
            f.write(
                self.curr_tab*self.t
                + '<site name="ft_rope" pos="0 0 0" size="{} {} {}" rgba="1 0 0 0" type="sphere" group="1" />\n'.format(
                    self.r_thickness,
                    self.r_thickness/2,
                    self.r_thickness/2,
                )
            )
            if self.vis_subcyl:
                f.write(
                        (self.curr_tab)*self.t
                        + '<site name="suft_rbcyl_{}" pos="0 0 0" quat="0.707 0 0.707 0" size="0.001 {}" rgba="0 0 1 0.3" type="cylinder" group="1" />\n'.format(
                            0,
                            self.subcyl_len,
                        )
                    )
            for i_section in range(self.r_pieces):
                if i_section == 0:
                    f.write(
                        self.curr_tab*self.t
                        + '<body name="B_{}">\n'.format(
                            self.str_names[i_section],
                        )
                    )
                    self.curr_tab += 1
                    f.write(
                        (self.curr_tab)*self.t
                        + '<site name="S_{}" pos="0 0 0" size="{}"/>\n'.format(
                            self.str_names[i_section],
                            self.r_thickness*0.45
                        )
                    )
                else:
                    f.write(
                        self.curr_tab*self.t
                        + '<body name="B_{}" pos="{} 0 0">\n'.format(
                            self.str_names[i_section],
                            self.displace_link,
                        )
                    )
                    self.curr_tab += 1
                    f.write(
                        self.curr_tab*self.t
                        + '<joint name="J_{}" type="ball" pos="0 0 0" stiffness="{}" damping="{}" actuatorfrclimited="false"/>\n'.format(
                            self.str_names[i_section],
                            self.j_stiff,
                            self.j_damp,
                        )
                    )
                if self.mass_link is None:
                    f.write(
                        self.curr_tab*self.t
                        + '<geom name="G{}" pos="{} 0 0" quat="0.707107 0 -0.707107 0" type="{}" size="{:1.4f} {:1.4f}" rgba="{}" conaffinity="{}" contype="{}" solref="{}" condim="1"/>\n'.format(
                            i_section,
                            self.displace_link/2,
                            self.rope_type,
                            self.cap_size[0],
                            self.cap_size[1],
                            self.rgba_linkgeom,
                            self.con_data[0],
                            self.con_data[1],
                            self.solref_val
                        )
                    )
                else:
                    f.write(
                        self.curr_tab*self.t
                        + '<geom name="G{}" pos="{} 0 0" quat="0.707107 0 -0.707107 0" type="{}" size="{:1.4f} {:1.4f}" rgba="{}" mass="{}" conaffinity="{}" contype="{}" solref="{}" condim="1"/>\n'.format(
                            i_section,
                            self.displace_link/2,
                            self.rope_type,
                            self.cap_size[0],
                            self.cap_size[1],
                            self.rgba_linkgeom,
                            self.mass_link,
                            self.con_data[0],
                            self.con_data[1],
                            self.solref_val
                        )
                    )
                f.write(
                    (self.curr_tab)*self.t
                    + '<site name="S_{}" pos="0 0 0" size="{}" rgba="0 0 0 0"/>\n'.format(
                        i_section,
                        self.r_thickness*0.45
                    )
                )

                if self.vis_subcyl:
                    f.write(
                        (self.curr_tab)*self.t
                        + '<site name="subcylx_{}" pos="0 0 0" quat="0.707 0 0.707 0" size="0.001 {}" rgba="0 0 1 0.3" type="cylinder" group="1" />\n'.format(
                            i_section + 1,
                            self.subcyl_len,
                        )
                    )
                
                # f.write(self.curr_tab*self.t + '</body>\n')

                # self._write_mb_geom(f, i_section)

                # f.write(
                #     self.curr_tab*self.t
                #     + '<site name="r_link{}_site" type="sphere" size="0.0001 0.0001 0.0001"/>\n'.format(
                #         i_section + 1,
                #     )
                # )
                if (
                    i_section == 0 
                    or i_section == self.r_pieces - 1
                    or i_section == np.floor(self.r_pieces / 2)
                ):
                    f.write(
                        self.curr_tab*self.t
                        + '<site name="twistcylx_{}" pos="0 0 0.05" quat="0.707 0 0 -0.707" size="0.001 0.05" rgba="0 0 1 {}" type="cylinder" group="1" />\n'.format(
                            i_section + 1,
                            self.subcyl_alpha
                        )
                    )
                    f.write(
                        self.curr_tab*self.t
                        + '<site name="twistcyly_{}" pos="0 0.05 0" quat="0.707 -0.707 0 0" size="0.001 0.05" rgba="1 1 0 {}" type="cylinder" group="1" />\n'.format(
                            i_section + 1,
                            self.subcyl_alpha
                        )
                    )
            f.write(
                (self.curr_tab)*self.t
                + '<site name="S_{}" pos="{} 0 0" size="{}"/>\n'.format(
                    'last',
                    self.displace_link,
                    self.r_thickness*0.45
                )
            )
            f.write(
                self.curr_tab*self.t
                + '<body name="B_last2" pos="{} 0 0">\n'.format(
                    self.displace_link
                )
            )
            f.write(self.curr_tab*self.t + '</body>\n')
            self.curr_tab -= 1
            for i_section in range(self.r_pieces+1):
                f.write(self.curr_tab*self.t + '</body>\n')
                self.curr_tab -= 1
            f.write(self.t + "</worldbody>\n")
            f.write(self.t + '<sensor>\n')
            f.write(self.t + '</sensor>\n')
            self._write_equality(f)
            self._write_exclusion(f)
            f.write('</mujoco>\n')

    def _write_anchorbox(self):
        self.curr_tab = 0
        with open(self.obj_path2, "w+") as f:
            # create box
            f.write('<mujoco model="anchor-box">\n')
            self.curr_tab += 1
            f.write(self.curr_tab*self.t + "<worldbody>\n")
            self.curr_tab += 1
            
            # box at end
            f.write(self.curr_tab*self.t + '<body name="eef_body" pos="{} {} {}" quat="1.0 0.0 0.0 0.0">\n'.format(
                self.init_pos[0]-self.grow_dirn[0]*self.r_len,
                self.init_pos[1]-self.grow_dirn[1]*self.r_len,
                self.init_pos[2]-self.grow_dirn[2]*self.r_len
            ))
            f.write((self.curr_tab+1)*self.t + '<geom name="eef_geom" type="box" mass="1" size="{} {} {}" contype="0" conaffinity="0" rgba=".8 .2 .1 {}" friction="1 0.005 0.0001"/>\n'.format(
                self.r_thickness,
                self.r_thickness,
                self.r_thickness*2.0,
                self.subcyl_alpha
            ))
            f.write(self.curr_tab*self.t + '</body>')

            # box at start
            f.write(self.curr_tab*self.t + '<body name="eef_body2" pos="{} {} {}" quat="1.0 0.0 0.0 0.0">\n'.format(
                self.init_pos[0],
                self.init_pos[1],
                self.init_pos[2]
            ))
            f.write((self.curr_tab+1)*self.t + '<geom name="eef_geom2" type="box" mass="1" size="{} {} {}" contype="0" conaffinity="0" rgba=".8 .2 .1 {}" friction="1 0.005 0.0001"/>\n'.format(
                self.r_thickness,
                self.r_thickness,
                self.r_thickness*2.0,
                self.subcyl_alpha
            ))
            f.write(self.curr_tab*self.t + '</body>')

            self.curr_tab -= 1
            f.write(self.curr_tab*self.t + "</worldbody>\n")
            self._write_equality(f)
            self._write_exclusion(f)
            # f.write(t + '<contact>\n')
            # curr_tab += 1
            # for i in range(1,r_pieces+1):
            #     for j in range(1,i):
            #         if abs(i-j) > 1:
            #             f.write(
            #                 curr_tab*t + "<pair geom1='r_link{}_geom' geom2='r_link{}_geom'/>\n".format(
            #                     i, j
            #                 )
            #             )
            # curr_tab -= 1
            # f.write(t + '</contact>\n')
            f.write('</mujoco>\n')

    def _write_weldweight(self):
        ## ADDITIONAL MASS -- to strengthen weld constraint: more mass, stronger weld
        ww_massmulti = 40.
        mass_density = 1000.
        if self.mass_link is None:
            mass_ww = ww_massmulti * mass_density * self.r_len * (self.r_thickness/2)**2 * np.pi
        else:
            mass_ww = ww_massmulti * self.r_mass
        with open(self.obj_path3, "w+") as f:
            f.write("<worldbody>\n")
            f.write(
                '   <body pos="0. 0. 0.">\n'
            )
            f.write(
                # self.curr_tab*self.t
                # + '<geom name="Ganchor" pos="{} 0 0" type="{}" size="{:1.4f}" rgba="{}" mass="{}" conaffinity="0" contype="0" condim="1"/>\n'.format(
                '       <geom pos="{} 0 0" type="{}" size="{:1.4f}" rgba="{}" mass="{}" conaffinity="0" contype="0" condim="1"/>\n'.format(
                    self.displace_link,
                    'sphere',
                    self.cap_size[0]*1.0,
                    self.rgba_linkgeom,
                    mass_ww
                    # 1.0,
                    # self.con_data[0],
                    # self.con_data[1],
                )
            )
            f.write(
                '   </body>\n'
            )
            f.write("</worldbody>\n")

    def _write_equality(self, f):
        f.write(self.t + '<equality>\n')
        # f.write(
        #     # self.t + "   <connect body1='{}' body2='r_joint{}_body' solref='0.004 1' anchor='0 0 0'/>\n".format(
        #     # self.t + "   <connect body1='{}' body2='r_joint{}_body' anchor='0 0 0'/>\n".format(
        #     self.t + "   <weld body1='{}' body2='r_joint{}_body' solref='{}' relpose='{} 0 0 1 0 0 0'/>\n".format(
        #         # 25,
        #         self.attach_bodyname,
        #         self.r_pieces-1,
        #         self.solref_val,
        #         self.r_len / self.r_pieces,
        #     )
        # )
        f.write(
            # self.t + "   <connect body1='{}' body2='r_joint{}_body' solref='0.004 1' anchor='0 0 0'/>\n".format(
            # self.t + "   <connect body1='{}' body2='r_joint{}_body' anchor='0 0 0'/>\n".format(
            # self.t + "   <weld body1='{}' body2='r_joint{}_body' solref='{}' relpose='0. 0 0 1 0 0 0'/>\n".format(
            2*self.t + "<weld body1='B_{}' body2='{}' solref='{}'/>\n".format(
                # 25,
                self.str_names[self.r_pieces-1],
                self.attach_bodyname,
                self.solref_val
                # + self.r_len / self.r_pieces / 2,
            )
        )
        f.write(
            # self.t + "   <connect body1='B_last' body2='{}' solref='{}' anchor='{} 0 0'/>\n".format(
            # self.t + "   <connect body1='{}' body2='r_joint{}_body' anchor='0 0 0'/>\n".format(
            # self.t + "   <weld body1='B_last' body2='{}' solref='{}' relpose='0.0 0 0 0 0 -1 0'/>\n".format(
            self.t + "   <weld body1='B_first' body2='{}' solref='{}'/>\n".format(
                # 25,
                'eef_body2',
                self.solref_val,
                # anchor_pt
                # self.r_len / self.r_pieces
            )
        )
        f.write(self.t + '</equality>\n')

    def _write_exclusion(self, f):
        curr_tab = 1
        f.write(curr_tab*self.t + '<contact>\n')
        curr_tab += 1
        f.write(
            curr_tab*self.t
            + '<exclude body1="B_first" body2="B_last"/>\n'
        )
        f.write(
            curr_tab*self.t
            + '<exclude body1="B_1" body2="B_last"/>\n'
        )
        f.write(
            curr_tab*self.t
            + '<exclude body1="B_first" body2="B_{}"/>\n'.format(
                self.r_pieces-2
            )
        )
        for i_section in range(1,self.r_pieces):
            i1 = self.str_names[i_section-1]
            i2 = self.str_names[i_section]
            f.write(
                curr_tab*self.t
                + '<exclude body1="B_{}" body2="B_{}"/>\n'.format(
                    i1,
                    i2
                )
            )
        f.write(
            curr_tab*self.t
            + '<exclude body1="eef_body" body2="B_first"/>\n'.format(
                self.r_pieces
            )
        )
        f.write(
            curr_tab*self.t
            + '<exclude body1="eef_body" body2="B_last"/>\n'.format(
                self.r_pieces
            )
        )
        f.write(
            curr_tab*self.t
            + '<exclude body1="eef_body" body2="B_{}"/>\n'.format(
                self.r_pieces-2
            )
        )
        curr_tab -= 1
        f.write(curr_tab*self.t + '</contact>\n')