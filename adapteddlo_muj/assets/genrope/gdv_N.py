import os
import numpy as np
import adapteddlo_muj.utils.transform_utils as T
from adapteddlo_muj.utils.xml_utils import XMLWrapper
import mujoco

class GenKin_N:
    def __init__(
        self,
        r_len=0.9,
        r_thickness=0.02,
        r_pieces=20,
        stiff_vals=[5e5,1e6],
        j_damp=0.001,
        r_mass=None,
        init_pos=[1.3, 0.0, 0.13],
        init_quat=[1.0, 0., 0., 0.],
        coll_on=False,
        rope_type="capsule",
        attach_sitename="eef_body",
        vis_subcyl=True,
        obj_path=None,
        rgba_vals=None,
        plugin_name="cable",
        twist_displace=0.0
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
        self._init_plugins()
        self.plugin_name = plugin_name

        self.r_len = r_len
        self.r_thickness = r_thickness
        self.r_pieces = r_pieces
        self.stiff_vals = stiff_vals
        self.j_damp = j_damp
        self.r_mass = r_mass
        self.init_pos = init_pos
        self.init_quat = T.quat_multiply(init_quat,np.array([0., 0., 1., 0.]))
        self.twist_displace = twist_displace
        if coll_on:
            self.con_data = [1, 1]
        else:
            self.con_data = [0, 0]

        self.grow_dirn = np.dot(np.array([1.,0,0]),T.quat2mat(init_quat))
        # self.init_angle = init_angle
        self.rope_type = rope_type
        self.attach_sitename = attach_sitename
        self.vis_subcyl = vis_subcyl
        self.obj_path = obj_path

        if self.vis_subcyl:
            self.subcyl_alpha = 0.3
        else:
            self.subcyl_alpha = 0.0
        if rgba_vals is None:
            rgba_vals = np.concatenate((np.array([30,16,202])/300,[1.0]))
        self.rgba_linkgeom = f'{rgba_vals[0]} {rgba_vals[1]} {rgba_vals[2]} {rgba_vals[3]}'

        self._init_variables()
        with open(self.obj_path, "w+") as f:
            self._write_init(f)
        self._unpackcompositexml()
        self._write_anchorbox()
        self._write_weldweight()

    def _init_plugins(self):
        plugin_path = os.environ.get("MJPLUGIN_PATH")
        if plugin_path:
            plugin_file = os.path.join(plugin_path, "libelasticity.so")
            try:
                mujoco.mj_loadPluginLibrary(plugin_file)
            except Exception as e:
                print(f"Failed to load plugin: {e}")
        else:
            print("MJPLUGIN_PATH is not set.")

    def _unpackcompositexml(self):
        self.xml = XMLWrapper(self.obj_path)
        xml_string = self.xml.get_xml_string()
        model = mujoco.MjModel.from_xml_string(xml_string)
        mujoco.mj_saveLastXML(self.obj_path,model)

    def _write_init(self, f):
        f.write('<mujoco model="stiff-rope">\n')
        self.curr_tab += 1

        # extension
        f.write(self.curr_tab*self.t + "<extension>\n")
        self.curr_tab += 1
        f.write(self.curr_tab*self.t + f'<plugin plugin="mujoco.elasticity.{self.plugin_name}"/>\n')
        self.curr_tab -= 1
        f.write(self.curr_tab*self.t + "</extension>\n")

        # worldbody
        f.write(self.curr_tab*self.t + "<worldbody>\n")
        self.curr_tab += 1

        # main cable body
        f.write(
            self.curr_tab*self.t 
            + '<body name="stiffrope" pos="{} {} {}" quat="{} {} {} {}">\n'.format(
                self.attach_pos[0],
                self.attach_pos[1],
                self.attach_pos[2],
                self.init_quat[0],
                self.init_quat[1],
                self.init_quat[2],
                self.init_quat[3],
            )
        )
        self.curr_tab += 1

        # cable
        f.write(self.curr_tab*self.t + '<composite type="{}" curve="s" count="{} 1 1" size="{}" offset="0 0 0" initial="none">\n'.format(
            self.plugin_name,
            self.r_pieces+1,
            self.r_len,
        ))
        self.curr_tab += 1
        f.write(self.curr_tab*self.t + f'<plugin plugin="mujoco.elasticity.{self.plugin_name}">\n')
        self.curr_tab += 1
        f.write(self.curr_tab*self.t + '<config key="twist" value="{}"/>\n'.format(
            self.stiff_vals[0]
        ))
        f.write(self.curr_tab*self.t + '<config key="bend" value="{}"/>\n'.format(
            self.stiff_vals[1]
        ))
        if self.plugin_name != "cable":
            f.write(self.curr_tab*self.t + '<config key="twist_displace" value="{}"/>\n'.format(
                self.twist_displace
            ))
        # f.write(self.curr_tab*self.t + '<config key="vmax" value="0.05"/>\n')
        self.curr_tab -= 1
        f.write(self.curr_tab*self.t + '</plugin>\n')
        # joint
        f.write(self.curr_tab*self.t + '<joint kind="main" damping="{}"/>\n'.format(
            self.j_damp
        ))
        # geom 
        if self.mass_link is None:
            f.write(self.curr_tab*self.t + '<geom type="{}" size="{}" rgba="{}" condim="1" conaffinity="{}" contype="{}" solref="{}"/>\n'.format(
                self.rope_type,
                self.cap_size,
                self.rgba_linkgeom,
                self.con_data[0],
                self.con_data[1],
                self.solref_val
            ))
        else:
            f.write(self.curr_tab*self.t + '<geom type="{}" size="{}" rgba="{}" condim="1" conaffinity="{}" contype="{}" mass="{}" solref="{}"/>\n'.format(
                self.rope_type,
                self.cap_size,
                self.rgba_linkgeom,
                self.con_data[0],
                self.con_data[1],
                self.mass_link,
                self.solref_val
            ))
        self.curr_tab -= 1
        f.write(self.curr_tab*self.t + "</composite>\n")
        self.curr_tab -= 1
        f.write(self.curr_tab*self.t + "</body>\n")

        # eef_body (connected/welded body)
        self.curr_tab -= 1
        f.write(self.curr_tab*self.t + "</worldbody>\n")
        self.curr_tab -= 1
        f.write(self.curr_tab*self.t + "</mujoco>\n")


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
        self.box_size = [
                self.r_thickness, self.r_thickness,
                2*self.r_thickness,
            ]

        if self.obj_path is None:
            self.obj_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "nativerope1dkin.xml"
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
        self.cap_size = self.r_thickness / 2
        
        self.curr_tab = 0

    def _write_anchorbox(self):
        self.curr_tab = 0
        with open(self.obj_path2, "w+") as f:
            # create box
            f.write('<mujoco model="anchor-box">\n')
            self.curr_tab += 1
            f.write(self.curr_tab*self.t + "<worldbody>\n")
            self.curr_tab += 1
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
        self.displace_link = self.r_len / self.r_pieces
        ww_massmulti = 40.
        mass_density = 1000.0
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
                    self.r_thickness/2*1.0,
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
        f.write(
            # self.t + "   <connect body1='B_last' body2='{}' solref='{}' anchor='{} 0 0'/>\n".format(
            # self.t + "   <connect body1='{}' body2='r_joint{}_body' anchor='0 0 0'/>\n".format(
            # self.t + "   <weld body1='B_last' body2='{}' solref='{}' relpose='0.0 0 0 0 0 -1 0'/>\n".format(
            self.t + "   <weld body1='B_last' body2='{}' solref='{}'/>\n".format(
                # 25,
                self.attach_sitename,
                self.solref_val,
                # anchor_pt
                # self.r_len / self.r_pieces
            )
        )
        f.write(self.t + '</equality>\n')

    def _write_exclusion(self, f):
        f.write(self.curr_tab*self.t + '<contact>\n')
        self.curr_tab += 1
        f.write(
            self.curr_tab*self.t
            + '<exclude body1="B_first" body2="B_1"/>\n'
        )
        f.write(
            self.curr_tab*self.t
            + '<exclude body1="B_first" body2="B_last"/>\n'
        )
        f.write(
            self.curr_tab*self.t
            + '<exclude body1="B_1" body2="B_last"/>\n'
        )
        f.write(
            self.curr_tab*self.t
            + '<exclude body1="B_first" body2="B_{}"/>\n'.format(
                self.r_pieces-2
            )
        )
        f.write(
            self.curr_tab*self.t
            + '<exclude body1="eef_body" body2="B_first"/>\n'.format(
                self.r_pieces
            )
        )
        f.write(
            self.curr_tab*self.t
            + '<exclude body1="eef_body" body2="B_last"/>\n'.format(
                self.r_pieces
            )
        )
        f.write(
            self.curr_tab*self.t
            + '<exclude body1="eef_body" body2="B_{}"/>\n'.format(
                self.r_pieces-2
            )
        )
        self.curr_tab -= 1
        f.write(self.curr_tab*self.t + '</contact>\n')