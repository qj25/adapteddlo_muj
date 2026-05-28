from typing import Dict, List, Optional

from adapteddlo_muj.envs.simvreal_test import adapt, massspring, native, xpbd


DEFAULT_MODELS = ["adapt", "native", "massspring"]

MODEL_REGISTRY = {
    "adapt": adapt.get_model_spec(),
    "native": native.get_model_spec(),
    "massspring": massspring.get_model_spec(),
    "xpbd": xpbd.get_model_spec(),
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
            f"Unknown simvreal model(s): {unknown}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return {m: MODEL_REGISTRY[m] for m in model_names}
