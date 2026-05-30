from adapteddlo_muj.envs.test_shape_w_arm.base import create_rnr2_env


def create_env(wire_color: str, overall_rot, do_render: bool):
    return create_rnr2_env(
        wire_color, "massspring", overall_rot, do_render, stiff_key="massspring"
    )


def get_model_spec():
    return {"name": "massspring", "create_env": create_env, "use_plugin": False}
