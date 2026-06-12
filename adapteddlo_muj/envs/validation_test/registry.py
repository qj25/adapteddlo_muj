from typing import Dict, List, Optional

DEFAULT_MODELS = ["adapt", "native", "massspring"]

MODEL_REGISTRY = {
    "adapt": {},
    "native": {},
    "massspring": {},
    "xfrc": {},
    "geds": {},
    "jpqder": {},
    "cosserat": {},
    "cosserat3": {},
    "cosserat5": {},
    "xpbd": {},
}


def parse_models_arg(models_arg: Optional[str], default_models: Optional[List[str]] = None) -> List[str]:
    models = default_models or DEFAULT_MODELS
    if models_arg is None or models_arg.strip() == "":
        return models
    return [m.strip() for m in models_arg.split(",") if m.strip()]


def get_model_names(model_names: List[str]) -> List[str]:
    unknown = [m for m in model_names if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown validation test model(s): {unknown}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return model_names
