import os

from adapteddlo_muj.envs.simvreal_test.base import load_sim_pickle


def _plugin_picklename(wire_color: str, pos_id: int, model_name: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "simdata",
        "plugin",
        f"simdata_{wire_color}{pos_id}_{model_name}.pickle",
    )


def _load_case(wire_color: str, pos_id: int):
    for model_name in ("jpqder", "adapt2"):
        if os.path.exists(_plugin_picklename(wire_color, pos_id, model_name)):
            return load_sim_pickle(wire_color=wire_color, pos_id=pos_id, model_name=model_name)
    return load_sim_pickle(wire_color=wire_color, pos_id=pos_id, model_name="adapt")


def get_model_spec():
    return {"name": "jpqder", "load_case": _load_case}
