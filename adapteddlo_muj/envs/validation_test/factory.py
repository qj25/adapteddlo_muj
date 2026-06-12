import os
from typing import Any

from adapteddlo_muj.utils.wire_plugin import COSSERAT_WIRE_PLUGIN_CONFIGS

_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)

# Plugin models store pickles under adapt/plgn/<plugin>/.
PLUGIN_MODELS = frozenset({"jpqder", "cosserat"})
PLUGIN_NAME = "wire"
MANUAL_ROT_MODELS = frozenset({"massspring", "cosserat5"})


def data_subpath(model_name: str) -> str:
    if model_name in PLUGIN_MODELS:
        return f"adapt/plgn/{PLUGIN_NAME}"
    return model_name


def validation_data_dir(model_name: str, test_type: str) -> str:
    return os.path.join(_DATA_ROOT, test_type, data_subpath(model_name))


def ensure_validation_data_dirs(model_name: str) -> None:
    for test_type in ("mbi", "lhb"):
        os.makedirs(validation_data_dir(model_name, test_type), exist_ok=True)


def create_validation_env(model_name: str, test_type: str, **kwargs: Any):
    ensure_validation_data_dirs(model_name)
    if model_name == "native":
        from adapteddlo_muj.envs.native_cable_valid_test import TestCableEnv

        return TestCableEnv(test_type=test_type, **kwargs)
    if model_name == "xfrc":
        from adapteddlo_muj.envs.our_xfrc_rope_valid_test import TestRopeXfrcEnv

        return TestRopeXfrcEnv(test_type=test_type, **kwargs)
    if model_name == "jpqder":
        from adapteddlo_muj.envs.validitytest_env import TestPluginEnv

        return TestPluginEnv(
            test_type=test_type,
            plugin_name=PLUGIN_NAME,
            **kwargs,
        )
    if model_name == "cosserat":
        from adapteddlo_muj.envs.validitytest_env import TestPluginEnv

        return TestPluginEnv(
            test_type=test_type,
            plugin_name=PLUGIN_NAME,
            extra_plugin_configs=COSSERAT_WIRE_PLUGIN_CONFIGS,
            **kwargs,
        )

    if model_name in MANUAL_ROT_MODELS:
        from adapteddlo_muj.envs.our_rope_manrot_valid_test import TestRopeManRotEnv

        return TestRopeManRotEnv(
            test_type=test_type,
            model_name=model_name,
            **kwargs,
        )

    from adapteddlo_muj.envs.our_rope_valid_test import TestRopeEnv

    return TestRopeEnv(test_type=test_type, model_name=model_name, **kwargs)


def close_env(env, do_render: bool) -> None:
    if do_render and env.viewer is not None:
        env.viewer.close()
    env.close()
