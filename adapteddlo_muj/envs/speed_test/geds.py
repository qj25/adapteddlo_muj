from adapteddlo_muj.envs.our_rope_valid_test import TestRopeEnv


def _run(settings):
    env = TestRopeEnv(
        overall_rot=0.0,
        do_render=False,
        r_pieces=settings["r_pieces"],
        r_len=settings["r_len"],
        r_thickness=settings["r_thickness"],
        test_type=settings["test_type"],
        alpha_bar=settings["alpha_val"],
        beta_bar=settings["beta_val"],
        model_name="geds",
    )
    if settings["test_type"] == "speedtest1":
        return env.run_speedtest1()
    return env.run_speedtest2()


def get_model_spec():
    return {"name": "geds", "run": _run}
