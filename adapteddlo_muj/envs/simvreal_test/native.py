from adapteddlo_muj.envs.simvreal_test.base import load_sim_pickle


def _load_case(wire_color: str, pos_id: int):
    return load_sim_pickle(wire_color=wire_color, pos_id=pos_id, model_name="native")


def get_model_spec():
    return {"name": "native", "load_case": _load_case}
