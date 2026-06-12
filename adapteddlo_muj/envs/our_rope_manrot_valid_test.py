"""Validation env for rope models that apply LHB/MBI twist via manual end rotation."""

import os
import sys
import pickle

import numpy as np

import adapteddlo_muj.utils.transform_utils as T
import adapteddlo_muj.utils.mjc2_utils as mjc2
from adapteddlo_muj.envs.our_rope_valid_test import TestRopeEnv


class TestRopeManRotEnv(TestRopeEnv):
    """Like TestRopeEnv, but applies overall_rot by rotating rope ends (native cable style)."""

    def __init__(self, *args, overall_rot=None, **kwargs):
        self._manual_overall_rot = 0.0 if overall_rot is None else overall_rot
        super().__init__(*args, overall_rot=0.0, **kwargs)
        self.overall_rot = self._manual_overall_rot

    def lhb_init(self):
        l_shorten = 0.3
        n_steps = 100
        step_len = l_shorten / n_steps / 2

        pos_move = np.array([step_len, -step_len, -step_len])
        for _ in range(1):
            self.ropeend_pos_all(pos_move=pos_move.copy())
        pos_move = np.array([step_len, step_len, step_len])
        for _ in range(1):
            self.ropeend_pos_all(pos_move=pos_move.copy())

        self.rot_x_rads2(x_rads=self._manual_overall_rot)

        pos_move = np.array([step_len, 0., 0.])
        print('0')
        for i in range(n_steps - 2):
            sys.stdout.write(f"\033[{1}F")
            print(f"init stage: {i+1}/{n_steps-2}")
            self.ropeend_pos_all(pos_move=pos_move.copy())

    def start_lhbtest(self, new_start):
        self.overall_rot = self._manual_overall_rot
        if self.do_render:
            self.set_viewer_details(
                5.7628,
                -40.478,
                -12.434,
                np.array([-1.17009866, -1.37107526, 0.02327594]),
            )

        self.freq_velreset = 0.2
        lhb_picklename = 'lhbtest{}.pickle'.format(self.r_pieces)
        lhb_picklename = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/lhb/" + self.picklefolder + "/" + lhb_picklename,
        )
        if new_start:
            self.lhb_init()
            self.init_pickle = self.get_state()
            with open(lhb_picklename, 'wb') as f:
                pickle.dump(self.init_pickle, f)
            print('Pickle saved!')
        else:
            with open(lhb_picklename, 'rb') as f:
                self.init_pickle = pickle.load(f)
            self.set_state(self.init_pickle)

        s_ss_center, fphi_center, max_devi = self.lhb_testing()

        print(f"fphi = {fphi_center}")
        print(f"s_ss = {s_ss_center}")
        print(f"max_devi = {max_devi}")
        pickledata2 = [
            max_devi,
            np.array(self.data.site_xpos[self.joint_site_idx[:]]).copy(),
        ]
        pickledata_path2 = 'lhb{}_miscdata.pickle'.format(self.r_pieces)
        pickledata_path2 = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/lhb/" + self.picklefolder + "/" + pickledata_path2,
        )
        with open(pickledata_path2, 'wb') as f:
            pickle.dump(pickledata2, f)

        pickledata = [fphi_center, s_ss_center]
        pickledata_path = 'lhb{}.pickle'.format(self.r_pieces)
        pickledata_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/lhb/" + self.picklefolder + "/" + pickledata_path,
        )
        with open(pickledata_path, 'wb') as f:
            pickle.dump(pickledata, f)
        print('pickled data')
        print(f"r_pieces = {self.r_pieces}")

    def start_circletest(self, new_start):
        self.overall_rot = self._manual_overall_rot
        if self.do_render:
            self.set_viewer_details(
                4.5076,
                45.922,
                -19.810,
                np.array([-3.46078809, -0.26378238, 0.63761223]),
            )

        self.freq_velreset = 0.2
        self.stable_bool = True
        e_tol = 7.5
        circtest_picklename = 'mbitest1.pickle'
        circtest_picklename = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/mbi/" + self.picklefolder + "/" + circtest_picklename,
        )
        if new_start:
            self.circle_oop = False
            self.circle_init()
            self.init_pickle = self.get_state()
            with open(circtest_picklename, 'wb') as f:
                pickle.dump(self.init_pickle, f)
            input(f'Circle Pickle saved!')
        else:
            with open(circtest_picklename, 'rb') as f:
                self.init_pickle = pickle.load(f)

            self.set_state(self.init_pickle)
            self.rot_x_rads(x_rads=self._manual_overall_rot)

            norm_force = self.get_rope_normal()
            self.apply_force_t(t=1.0, force_dir=norm_force)

            max_e = 0.
            e_outofplane = 0.
            self.circle_oop = False
            print('0')
            for i in range(100):
                sys.stdout.write(f"\033[{1}F")
                print(f"Holding for {i+1} / {100}..")
                self.hold_pos(0.2, rend=True)
                if self.instability_check():
                    self.stable_bool = False
                e_outofplane = self.check_e_PCA_circle()
                if e_outofplane > max_e:
                    max_e = e_outofplane
                if max_e > e_tol:
                    print(f"e_tol = {e_tol}")
                    print(f"e_outofplane = {e_outofplane}")
                    print(f"max_e = {max_e}")

                    self.circle_oop = True
                    print(
                        f'b_a = {self.beta_bar/self.alpha_bar} '
                        f'=================================='
                    )
                    print(
                        f'out of plane theta_crit = {self.overall_rot} '
                        f'=================================='
                    )
                    return e_outofplane
            print(f"e_tol = {e_tol}")
            print(f"e_outofplane = {e_outofplane}")
            print(f"max_e = {max_e}")
            if max_e > e_tol:
                self.circle_oop = True
                print(
                    f'b_a = {self.beta_bar/self.alpha_bar} '
                    f'=================================='
                )
                print(
                    f'out of plane theta_crit = {self.overall_rot} '
                    f'=================================='
                )
            if not self.stable_bool:
                self.circle_oop = True
                input('Unstable sim: press "Enter" to continue..')
            return e_outofplane

    def rot_x_rads(self, x_rads):
        n_rotsteps = int(x_rads / np.pi * 180)
        rad_leftover = x_rads - (n_rotsteps / 180 * np.pi)
        print('0')
        for i in range(n_rotsteps):
            sys.stdout.write(f"\033[{1}F")
            print(f"init rot stage (degs): {i+1}/{n_rotsteps}")
            self.ropeend_rot(rot_axis=0)
        self.ropeend_rot(rot_a=rad_leftover, rot_axis=0)

    def rot_x_rads2(self, x_rads):
        x_rads /= 2.0
        body_id = mjc2.obj_name2id(self.model, "body", "eef_body")

        n_rotsteps = int(x_rads / np.pi * 180)
        rad_leftover = x_rads - (n_rotsteps / 180 * np.pi)
        print('0')
        for i in range(n_rotsteps):
            sys.stdout.write(f"\033[{1}F")
            print(f"init rot stage (degs): {2*(i+1)}/{2*n_rotsteps}")
            self.ropeend_rot2(body_id, rot_axis=0)
        self.ropeend_rot2(body_id, rot_a=rad_leftover, rot_axis=0)

    def ropeend_rot2(self, body_id, rot_a=np.pi / 180, rot_axis=0):
        rot_arr = np.zeros(3)
        rot_arr[rot_axis] = rot_a
        rot_quat = T.axisangle2quat(rot_arr)
        rot_quat2 = T.axisangle2quat(-rot_arr)
        new_quat = T.quat_multiply(
            rot_quat, self.model.body_quat[self.ropeend_body_id]
        )
        new_quat2 = T.quat_multiply(rot_quat2, self.model.body_quat[body_id])
        self.model.body_quat[self.ropeend_body_id] = new_quat
        self.model.body_quat[body_id] = new_quat2
        self.hold_pos(0.1)
