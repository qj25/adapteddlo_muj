from adapteddlo_muj.envs.test_shape_w_arm.base import create_jpqder_env


def create_env(wire_color: str, overall_rot, do_render: bool):
    return create_jpqder_env(wire_color, overall_rot, do_render)


def get_model_spec():
    return {"name": "jpqder", "create_env": create_env, "use_plugin": True}
