import numpy as np
from adapteddlo_muj.envs.realgrav_valid_test import TestRopeEnv

# Golden-section search algorithm in Python

def golden_section_search(f, a, b, tol=1e-5):
    # Golden ratio
    gr = (1 + 5**0.5) / 2
    
    # Initial points
    c = b - (b - a) / gr
    d = a + (b - a) / gr

    val_store = [None, None, None, None]
    
    while abs(b - a) > tol:
        print('new iter:')
        print([a,c,d,b])
        print(val_store)
        if val_store[1] is None:
            val_store[1] = f(c)
        if val_store[2] is None:
            val_store[2] = f(d)
        if val_store[1] < val_store[2]:
            b = d
            val_store[3] = val_store[2]
            val_store[2] = val_store[1]
            val_store[1] = None
        else:
            a = c
            val_store[0] = val_store[1]
            val_store[1] = val_store[2]
            val_store[2] = None

        # Recompute the new points
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    
    # Return the approximate minimum point
    return (a + b) / 2

def midpoint_rootfind(f,a,b,tol=1e-4):
    val_store = [f(a), f(b)]
    if val_store[0] == val_store[1]:
        print("Root is not in interval! Expand interval.")
    while abs(b - a) > tol:
        c = (a + b) / 2
        val_c = f(c)
        if val_c == val_store[0]:
            a = c
        else:
            b = c
        print([a,b])
    return (a + b) / 2

class mbi_stiff:
    def __init__(
        self,
        stest_type,
        rgba_vals,
        real_pos=None,
        massperlen=1.0,
        overall_rot=0.,
        r_len=1.0,
        grav_on=True,
        do_render=False,
        new_start=False,
    ):
        self.stest_type = stest_type
        self.rgba_vals = rgba_vals
        self.real_pos = real_pos
        self.massperlen = massperlen
        self.overall_rot = overall_rot
        self.r_len = r_len
        self.grav_on = grav_on
        self.do_render = do_render
        self.new_start = new_start
        self.r_pieces = 52
        self.r_len = self.r_len * self.r_pieces / (self.r_pieces-2)
        self.r_thickness = 0.01
        self.r_mass = self.massperlen * self.r_len
    
    def opt_func(self, stiff):
        if self.real_pos is None:
            print("self.real_pos not specified. Please specify or re_init class.")
            return None
        # optimization function to find alpha (bending stiffness)
        self.alpha_bar = stiff/(2*np.pi)**3
        self.beta_bar = stiff/(2*np.pi)**3

        env = TestRopeEnv(
            overall_rot=self.overall_rot,
            do_render=self.do_render,
            r_pieces=self.r_pieces,
            r_len=self.r_len,
            r_thickness=self.r_thickness,
            test_type='mbi',
            alpha_bar=self.alpha_bar,
            beta_bar=self.beta_bar,
            r_mass=self.r_mass,
            new_start=self.new_start,
            stifftorqtype=self.stest_type,
            grav_on=self.grav_on,
            rgba_vals=self.rgba_vals
        )
        if self.do_render:
            env.set_viewer_details(
                dist=1.5,
                azi=90.0,
                elev=0.0,
                lookat=np.array([-0.75,0.0,0.10])
            )
        sim_pos = env.observations['rope_pose'][1:-1].copy()[:,[0,2]]
        sim_pos -= sim_pos[0]
        sim_pos *= -1.0
        diff_pos = np.sum(
            np.linalg.norm(sim_pos-self.real_pos,axis=1)
        )/len(sim_pos)/self.r_len
        # print("|===|NEW DATA |================================================")
        # print(f"stiff_scale = {stiff}")
        # print(f"diff_pos = {diff_pos}")
        if self.do_render:
            env.viewer.close()
        env.close()

        return diff_pos
    
    def opt_func2(self,b_a):
        # optimization function to find b_a (twisting stiffness)
        # must have set self.alpha_bar and self.overall_rot before hand
        self.beta_bar = self.alpha_bar * b_a
        env = TestRopeEnv(
            overall_rot=self.overall_rot,
            do_render=self.do_render,
            r_pieces=self.r_pieces,
            r_len=self.r_len,
            r_thickness=self.r_thickness,
            test_type='mbi',
            alpha_bar=self.alpha_bar,
            beta_bar=self.beta_bar,
            r_mass=self.r_mass,
            new_start=self.new_start,
            stifftorqtype=self.stest_type,
            grav_on=self.grav_on,
            rgba_vals=self.rgba_vals
        )
        if self.do_render:
            env.set_viewer_details(
                dist=1.5,
                azi=90.0,
                elev=0.0,
                lookat=np.array([-0.75,0.0,0.10])
            )
        return env.circle_oop
    
if __name__ == "__main__":
    def square_func(x):
        print(x)
        return x*x
    
    opt_range = [-1,3]
    opt_val = golden_section_search(square_func,opt_range[0],opt_range[1])
    print(opt_val)