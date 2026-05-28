import os
import pickle


def load_sim_pickle(wire_color: str, pos_id: int, model_name: str):
    picklename = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "simdata",
        "plugin",
        f"simdata_{wire_color}{pos_id}_{model_name}.pickle",
    )
    with open(picklename, "rb") as f:
        return pickle.load(f)
