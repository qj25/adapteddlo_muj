from typing import Callable, Dict, List


SpeedTestRunFn = Callable[[Dict], float]


class SpeedModelSpec(dict):
    """Dict-like carrier for speed test model metadata and runner."""


def do_render_from_settings(settings: Dict) -> bool:
    return bool(settings.get("do_render", False))


def use_adapt_pickle_from_settings(settings: Dict) -> bool:
    return bool(settings.get("use_adapt_pickle", False))


def run_env_speed_test(env, settings: Dict) -> float:
    try:
        if settings["test_type"] == "speedtest1":
            return env.run_speedtest1()
        return env.run_speedtest2()
    finally:
        if getattr(env, "do_render", False) and getattr(env, "viewer", None) is not None:
            env.viewer.close()
            env.viewer = None


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
