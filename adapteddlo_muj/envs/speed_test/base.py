from typing import Callable, Dict, List


SpeedTestRunFn = Callable[[Dict], float]


class SpeedModelSpec(dict):
    """Dict-like carrier for speed test model metadata and runner."""


def build_result_payload(
    model: str,
    test_type: str,
    r_pieces_list: List[int],
    times: List[float],
    settings: Dict,
    created_at: str,
) -> Dict:
    return {
        "model": model,
        "test_type": test_type,
        "r_pieces_list": r_pieces_list,
        "times": times,
        "r_len": settings["r_len"],
        "r_thickness": settings["r_thickness"],
        "alpha_val": settings["alpha_val"],
        "beta_val": settings["beta_val"],
        "created_at": created_at,
    }
