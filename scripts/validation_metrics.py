"""Shared loading and average-deviation metrics for MBI / LHB validation tests."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "adapteddlo_muj", "data")
RESULTS_ROOT = os.path.join(DATA_ROOT, "validation_test_results")
LHB_PIECES = [40, 60, 80, 110, 140, 180]
LHB_S_MIN = -6.0
LHB_S_MAX = 6.0
MIN_LHB_TRIM_SAMPLES = 5

_PLUGIN_MODELS = frozenset({"jpqder", "cosserat"})
_PLUGIN_NAME = "wire"


def data_subpath(model_name: str) -> str:
    if model_name in _PLUGIN_MODELS:
        return f"adapt/plgn/{_PLUGIN_NAME}"
    return model_name


COMBINED_MODELS: List[tuple[str, str]] = [
    ("adapt", "adapted"),
    ("jpqder", "jpQ-DER"),
    ("native", "native"),
    ("massspring", "massspring"),
    ("cosserat5", "cosserat"),
]


@dataclass
class SkipRecord:
    key: str
    reason: str
    detail: str = ""


@dataclass
class MbiResult:
    model: str
    b_a: np.ndarray
    theta_crit: np.ndarray
    skipped: List = field(default_factory=list)


@dataclass
class LhbPieceResult:
    status: str
    fphi: Optional[np.ndarray] = None
    s_ss: Optional[np.ndarray] = None
    reason: str = ""
    detail: str = ""


@dataclass
class LhbResult:
    model: str
    r_pieces_list: List[int]
    pieces: Dict[int, LhbPieceResult]
    skipped: List = field(default_factory=list)


def _mbi_result_path(model_name: str) -> str:
    return os.path.join(DATA_ROOT, "mbi", data_subpath(model_name), "mbi1.pickle")


def _legacy_mbi_results_path(model_name: str) -> str:
    return os.path.join(RESULTS_ROOT, "mbi", f"{model_name}.pickle")


def _result_path(test_type: str, model_name: str) -> str:
    if test_type == "mbi":
        return _mbi_result_path(model_name)
    return os.path.join(RESULTS_ROOT, test_type, f"{model_name}.pickle")


def _legacy_lhb_path(model_name: str, r_pieces: int) -> str:
    return os.path.join(
        DATA_ROOT,
        "lhb",
        data_subpath(model_name),
        f"lhb{r_pieces}.pickle",
    )


def _register_pickle_classes() -> None:
    import sys

    main = sys.modules["__main__"]
    for cls in (SkipRecord, MbiResult, LhbPieceResult, LhbResult):
        setattr(main, cls.__name__, cls)


def _load_pickle(path: str):
    _register_pickle_classes()
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_mbi_payload(model_name: str, path: str) -> MbiResult:
    payload = _load_pickle(path)
    if isinstance(payload, MbiResult):
        return payload
    blob = payload
    half = round(len(blob) / 2)
    return MbiResult(
        model=model_name,
        b_a=np.asarray(blob[:half], dtype=float),
        theta_crit=np.asarray(blob[half:], dtype=float),
    )


def load_result(test_type: str, model_name: str):
    path = _result_path(test_type, model_name)
    if os.path.exists(path):
        if test_type == "mbi":
            return _load_mbi_payload(model_name, path)
        return _load_pickle(path)
    if test_type == "mbi":
        legacy_results = _legacy_mbi_results_path(model_name)
        if os.path.exists(legacy_results):
            return _load_mbi_payload(model_name, legacy_results)
    if test_type == "lhb":
        pieces: Dict[int, LhbPieceResult] = {}
        for r_pieces in LHB_PIECES:
            legacy = _legacy_lhb_path(model_name, r_pieces)
            if not os.path.exists(legacy):
                continue
            with open(legacy, "rb") as f:
                pickledata = pickle.load(f)
            pieces[r_pieces] = LhbPieceResult(
                status="ok",
                fphi=np.asarray(pickledata[0], dtype=float),
                s_ss=np.asarray(pickledata[1], dtype=float),
            )
        if pieces:
            return LhbResult(
                model=model_name,
                r_pieces_list=sorted(pieces.keys()),
                pieces=pieces,
            )
    raise FileNotFoundError(f"No results found for {model_name} ({test_type})")


def mbi_avg_deviation(result: MbiResult) -> float:
    valid = np.isfinite(result.theta_crit)
    if not np.any(valid):
        raise ValueError(f"[{result.model}] No valid MBI points for deviation.")
    b_a = result.b_a[valid]
    theta_crit = result.theta_crit[valid]
    theta_crit_base = 2 * np.pi * np.sqrt(3) / b_a
    return float(np.linalg.norm(theta_crit_base - theta_crit) / len(theta_crit))


@dataclass
class LhbPieceDeviation:
    value: float
    n_total: int
    n_trimmed: int
    trimmed_value: float
    raw_value: float
    used_raw_fallback: bool = False

    @property
    def is_low_confidence(self) -> bool:
        return self.used_raw_fallback


def _lhb_arrays(fphi: np.ndarray, s_ss: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_ss_arr = np.asarray(s_ss, dtype=float)
    fphi_sim = np.asarray(fphi, dtype=float)
    fphi_base = (np.tanh(s_ss_arr)) ** 2.0
    return s_ss_arr, fphi_sim, fphi_base


def _avg_deviation(fphi_sim: np.ndarray, fphi_base: np.ndarray) -> float:
    if len(fphi_base) == 0:
        raise ValueError("No LHB samples available for deviation.")
    return float(np.linalg.norm(fphi_sim - fphi_base) / len(fphi_base))


def _trim_lhb_arrays(
    s_ss: np.ndarray,
    fphi_sim: np.ndarray,
    fphi_base: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_ss_trim = s_ss.copy()
    fphi_trim = fphi_sim.copy()
    fphi_base_trim = fphi_base.copy()
    idx = 0
    while idx < len(s_ss_trim):
        if s_ss_trim[idx] <= LHB_S_MIN or s_ss_trim[idx] > LHB_S_MAX:
            s_ss_trim = np.delete(s_ss_trim, idx, axis=0)
            fphi_trim = np.delete(fphi_trim, idx, axis=0)
            fphi_base_trim = np.delete(fphi_base_trim, idx, axis=0)
        else:
            idx += 1
    return s_ss_trim, fphi_trim, fphi_base_trim


def lhb_piece_deviation_info(fphi: np.ndarray, s_ss: np.ndarray) -> LhbPieceDeviation:
    s_ss_arr, fphi_sim, fphi_base = _lhb_arrays(fphi, s_ss)
    raw_value = _avg_deviation(fphi_sim, fphi_base)

    _, fphi_trim, fphi_base_trim = _trim_lhb_arrays(s_ss_arr, fphi_sim, fphi_base)
    n_trimmed = len(fphi_base_trim)
    if n_trimmed == 0:
        return LhbPieceDeviation(
            value=raw_value,
            n_total=len(fphi_base),
            n_trimmed=0,
            trimmed_value=float("nan"),
            raw_value=raw_value,
            used_raw_fallback=True,
        )

    trimmed_value = _avg_deviation(fphi_trim, fphi_base_trim)
    used_raw_fallback = n_trimmed < MIN_LHB_TRIM_SAMPLES
    value = raw_value if used_raw_fallback else trimmed_value
    return LhbPieceDeviation(
        value=value,
        n_total=len(fphi_base),
        n_trimmed=n_trimmed,
        trimmed_value=trimmed_value,
        raw_value=raw_value,
        used_raw_fallback=used_raw_fallback,
    )


def lhb_piece_avg_deviation(fphi: np.ndarray, s_ss: np.ndarray) -> float:
    return lhb_piece_deviation_info(fphi, s_ss).value


def lhb_model_avg_deviation(result: LhbResult) -> float:
    deviations: List[float] = []
    for r_pieces in result.r_pieces_list:
        piece = result.pieces[r_pieces]
        if piece.status != "ok" or piece.fphi is None or piece.s_ss is None:
            continue
        deviations.append(lhb_piece_avg_deviation(piece.fphi, piece.s_ss))
    if not deviations:
        raise ValueError(f"[{result.model}] No valid LHB piece counts for deviation.")
    return float(np.mean(deviations))


def load_mbi_avg_deviations(
    model_names: Optional[List[str]] = None,
) -> List[float]:
    names = model_names or [name for name, _ in COMBINED_MODELS]
    values: List[float] = []
    for model_name in names:
        result = load_result("mbi", model_name)
        values.append(mbi_avg_deviation(result))
    return values


def load_lhb_avg_deviations(
    model_names: Optional[List[str]] = None,
) -> List[float]:
    names = model_names or [name for name, _ in COMBINED_MODELS]
    values: List[float] = []
    for model_name in names:
        result = load_result("lhb", model_name)
        values.append(lhb_model_avg_deviation(result))
    return values


def load_lhb_piece_deviation(
    model_name: str,
    r_pieces: int,
) -> Optional[LhbPieceDeviation]:
    result = load_result("lhb", model_name)
    piece = result.pieces.get(r_pieces)
    if piece is None or piece.status != "ok" or piece.fphi is None or piece.s_ss is None:
        return None
    return lhb_piece_deviation_info(piece.fphi, piece.s_ss)


# Preset LHB table columns (native, adapt) — do not overwrite.
LHB_TABLE_PRESET: Dict[int, tuple[str, str]] = {
    40: ("0.0389", "0.0257"),
    60: ("0.0624", "0.0141"),
    80: ("0.00692", "0.00900"),
    110: ("0.00673", "0.00493"),
    140: ("0.00397", "0.00315"),
    180: ("0.00348", "0.00189"),
}


def format_lhb_deviation(val: float, *, low_confidence: bool = False) -> str:
    if val == 0.0 and not low_confidence:
        return "0"
    if abs(val) < 0.01:
        text = f"{val:.5f}".rstrip("0")
        if text.endswith("."):
            text += "0"
        return text
    return f"{val:.4g}"


def format_lhb_latex_table(
    *,
    massspring_model: str = "massspring",
    cosserat_model: str = "cosserat5",
    indent: str = "\t\t\t",
) -> str:
    lines: List[str] = []
    for idx, r_pieces in enumerate(LHB_PIECES):
        native, adapt = LHB_TABLE_PRESET[r_pieces]
        ms_info = load_lhb_piece_deviation(massspring_model, r_pieces)
        cs_info = load_lhb_piece_deviation(cosserat_model, r_pieces)
        ms_text = (
            format_lhb_deviation(ms_info.value, low_confidence=ms_info.is_low_confidence)
            if ms_info is not None
            else "---"
        )
        cs_text = (
            format_lhb_deviation(cs_info.value, low_confidence=cs_info.is_low_confidence)
            if cs_info is not None
            else "---"
        )
        prefix = "" if idx == 0 else indent
        suffix = " \\\\" if idx == len(LHB_PIECES) - 1 else " \\\\ \\hline"
        lines.append(
            f"{prefix}{r_pieces}  & {native}  & {adapt} & {ms_text} & {cs_text}{suffix}"
        )
    return "\n".join(lines)
