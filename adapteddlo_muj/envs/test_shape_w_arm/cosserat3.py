from adapteddlo_muj.envs.test_shape_w_arm.base import create_rnr2_env


def create_env(wire_color: str, overall_rot, do_render: bool, model_name: str = "cosserat3"):
    return create_rnr2_env(wire_color, "cosserat3", overall_rot, do_render, model_name=model_name)


def get_model_spec():
    return {"name": "cosserat3", "create_env": create_env, "use_plugin": False}
