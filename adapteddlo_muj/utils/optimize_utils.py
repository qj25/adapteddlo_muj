import numpy as np

from adapteddlo_muj.envs.real2sim_paramiden.base import create_mbi_env, sim_pos_error

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
        model_name,
        rgba_vals,
        real_pos=None,
        massperlen=1.0,
        overall_rot=0.0,
        r_len=1.0,
        grav_on=True,
        do_render=False,
        new_start=False,
        stest_type=None,
        wire_color=None,
    ):
        self.model_name = stest_type or model_name
        self.rgba_vals = rgba_vals
        self.real_pos = real_pos
        self.massperlen = massperlen
        self.overall_rot = overall_rot
        self.r_len = r_len
        self.grav_on = grav_on
        self.do_render = do_render
        self.new_start = new_start
        self.wire_color = wire_color
        self.alpha_bar = None
        self.beta_bar = None
        self.env = None

    def _make_env(self, alpha_bar, beta_bar):
        self.alpha_bar = alpha_bar
        self.beta_bar = beta_bar
        self.env = create_mbi_env(
            self.model_name,
            alpha_bar=alpha_bar,
            beta_bar=beta_bar,
            rgba_vals=self.rgba_vals,
            massperlen=self.massperlen,
            wire_color=self.wire_color,
            overall_rot=self.overall_rot,
            rope_len=self.r_len,
            grav_on=self.grav_on,
            do_render=self.do_render,
            new_start=self.new_start,
        )
        return self.env

    def opt_func(self, stiff):
        if self.real_pos is None:
            print("self.real_pos not specified. Please specify or re_init class.")
            return None
        alpha_bar = stiff / (2 * np.pi) ** 3
        env = self._make_env(alpha_bar, alpha_bar)
        return sim_pos_error(env, self.real_pos, r_len=self.r_len)

    def opt_func2(self, b_a):
        if self.alpha_bar is None:
            raise ValueError("alpha_bar must be set before calling opt_func2")
        env = self._make_env(self.alpha_bar, self.alpha_bar * b_a)
        return env.circle_oop
    
if __name__ == "__main__":
    def square_func(x):
        print(x)
        return x*x
    
    opt_range = [-1,3]
    opt_val = golden_section_search(square_func,opt_range[0],opt_range[1])
    print(opt_val)