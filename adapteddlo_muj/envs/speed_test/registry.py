from typing import Dict, List, Optional

from adapteddlo_muj.envs.speed_test import adapt, geds, jpqder, massspring, native, plain, xfrc, xpbd


DEFAULT_MODELS = ["plain", "native", "xfrc", "adapt", "massspring", "jpqder"]

MODEL_REGISTRY = {
    "plain": plain.get_model_spec(),
    "native": native.get_model_spec(),
    "xfrc": xfrc.get_model_spec(),
    "adapt": adapt.get_model_spec(),
    "massspring": massspring.get_model_spec(),
    "jpqder": jpqder.get_model_spec(),
    "xpbd": xpbd.get_model_spec(),
    "geds": geds.get_model_spec(),
}


def parse_models_arg(models_arg: Optional[str], default_models: Optional[List[str]] = None) -> List[str]:
    models = default_models or DEFAULT_MODELS
    if models_arg is None or models_arg.strip() == "":
        return models
    return [m.strip() for m in models_arg.split(",") if m.strip()]


def get_model_specs(model_names: List[str]) -> Dict:
    unknown = [m for m in model_names if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown speed test model(s): {unknown}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return {m: MODEL_REGISTRY[m] for m in model_names}
