import os

from adapteddlo_muj.envs.simvreal_test.base import load_sim_pickle


def _load_case(wire_color: str, pos_id: int):
    picklename = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "simdata",
        "plugin",
        f"simdata_{wire_color}{pos_id}_massspring.pickle",
    )
    if os.path.exists(picklename):
        return load_sim_pickle(wire_color=wire_color, pos_id=pos_id, model_name="massspring")
    return load_sim_pickle(wire_color=wire_color, pos_id=pos_id, model_name="adapt")


def get_model_spec():
    return {"name": "massspring", "load_case": _load_case}
