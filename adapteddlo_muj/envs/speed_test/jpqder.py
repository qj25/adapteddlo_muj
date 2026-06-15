from adapteddlo_muj.envs.validitytest_env import TestPluginEnv
from adapteddlo_muj.envs.speed_test.base import do_render_from_settings, run_env_speed_test


def _run(settings):
    env = TestPluginEnv(
        overall_rot=0.0,
        do_render=do_render_from_settings(settings),
        r_pieces=settings["r_pieces"],
        r_len=settings["r_len"],
        r_thickness=settings["r_thickness"],
        test_type=settings["test_type"],
        alpha_bar=settings["alpha_val"],
        beta_bar=settings["beta_val"],
        plugin_name="wire",
    )
    return run_env_speed_test(env, settings)


def get_model_spec():
    return {"name": "jpqder", "run": _run}
