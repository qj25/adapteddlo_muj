"""DLO validation tests (MBI / LHB) with multi-model support."""

from __future__ import annotations

import builtins
import os
import pickle
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from adapteddlo_muj.envs.validation_test import (
    DEFAULT_MODELS,
    MODEL_REGISTRY,
    close_env,
    create_validation_env,
    data_subpath,
    get_model_names,
    parse_models_arg,
)
from adapteddlo_muj.utils.argparse_utils import dtd_parse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "adapteddlo_muj", "data")
RESULTS_ROOT = os.path.join(DATA_ROOT, "validation_test_results")
LHB_PIECES = [40, 60, 80, 110, 140, 180]
MASSSPRING_ALPHA_SCALE = 3.585
MBI_UNSTABLE_MODELS = frozenset({"massspring", "cosserat5"})
MBI_STIFFNESS_SCALE = 1.0
MBI_OVERALL_ROT_START = 2.5 # 8.5 for most, 6.5 for cosserat5, 2.5 for massspring
MBI_OVERALL_ROT_STEP = 0.1
MBI_OVERALL_ROT_MAX = 1000.0


class InstabilityError(RuntimeError):
    pass


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
    skipped: List[SkipRecord] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.skipped:
            return "ok"
        if np.all(np.isnan(self.theta_crit)):
            return "failed"
        return "partial"


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
    skipped: List[SkipRecord] = field(default_factory=list)

    @property
    def status(self) -> str:
        ok_count = sum(1 for p in self.pieces.values() if p.status == "ok")
        if ok_count == 0:
            return "failed"
        if ok_count < len(self.pieces):
            return "partial"
        return "ok"


def _effective_alpha_bar(model_name: str, alpha_bar: float) -> float:
    if model_name == "massspring":
        return alpha_bar * MASSSPRING_ALPHA_SCALE
    return alpha_bar


def _mbi_alpha_bar(model_name: str, alpha_bar: float) -> float:
    alpha = _effective_alpha_bar(model_name, alpha_bar)
    if model_name in MBI_UNSTABLE_MODELS:
        alpha *= MBI_STIFFNESS_SCALE
    return alpha


def _mbi_beta_bar(model_name: str, beta_bar: float) -> float:
    if model_name in MBI_UNSTABLE_MODELS:
        return beta_bar * MBI_STIFFNESS_SCALE
    return beta_bar


def _results_dir(test_type: str) -> str:
    return os.path.join(RESULTS_ROOT, test_type)


def _result_path(test_type: str, model_name: str) -> str:
    return os.path.join(_results_dir(test_type), f"{model_name}.pickle")


@contextmanager
def _noninteractive_sim():
    original_input = builtins.input
    builtins.input = lambda prompt="": ""
    try:
        yield
    finally:
        builtins.input = original_input


def _legacy_mbi_path(model_name: str) -> str:
    return os.path.join(DATA_ROOT, "mbi", data_subpath(model_name), "mbi1.pickle")


def _legacy_lhb_path(model_name: str, r_pieces: int) -> str:
    return os.path.join(
        DATA_ROOT,
        "lhb",
        data_subpath(model_name),
        f"lhb{r_pieces}.pickle",
    )


def save_result(test_type: str, payload: Any) -> str:
    out_path = _result_path(test_type, payload.model)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[{payload.model}] Saved results: {out_path}")
    return out_path


def load_result(test_type: str, model_name: str) -> Any:
    path = _result_path(test_type, model_name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    if test_type == "mbi":
        legacy = _legacy_mbi_path(model_name)
        if os.path.exists(legacy):
            with open(legacy, "rb") as f:
                blob = pickle.load(f)
            half = round(len(blob) / 2)
            return MbiResult(
                model=model_name,
                b_a=np.asarray(blob[:half], dtype=float),
                theta_crit=np.asarray(blob[half:], dtype=float),
            )
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


def _check_env_stability(env) -> None:
    if not getattr(env, "stable_bool", True):
        raise InstabilityError("simulation became unstable")


def _valid_numeric_array(arr: np.ndarray) -> bool:
    if arr is None:
        return False
    arr = np.asarray(arr, dtype=float)
    return arr.size > 0 and np.all(np.isfinite(arr))


def mbi_indivtest(
    model_name: str,
    overall_rot: float = 0.0,
    alpha_val: float = 1.0,
    beta_val: float = 1.0,
    do_render: bool = False,
    new_start: bool = False,
) -> float:
    r_pieces = 51
    r_len = 2 * np.pi * r_pieces / (r_pieces - 1)
    r_thickness = 0.05

    env = create_validation_env(
        model_name,
        test_type="mbi",
        overall_rot=overall_rot,
        do_render=do_render,
        r_pieces=r_pieces,
        r_len=r_len,
        r_thickness=r_thickness,
        alpha_bar=_mbi_alpha_bar(model_name, alpha_val),
        beta_bar=_mbi_beta_bar(model_name, beta_val),
        new_start=new_start,
    )
    try:
        _check_env_stability(env)
        if not env.circle_oop:
            return 0.0
        return float(overall_rot)
    finally:
        close_env(env, do_render)


def run_mbi_sim(
    model_name: str,
    new_start: bool = False,
    do_render: bool = False,
) -> MbiResult:
    print(f"[{model_name}] Starting MBI test...")
    n_data_mbi = 11
    beta_bar_lim = np.array([0.5, 1.25])
    beta_bar_step = (beta_bar_lim[1] - beta_bar_lim[0]) / (n_data_mbi - 1)
    beta_bar = np.zeros(n_data_mbi)
    alpha_bar = np.ones(n_data_mbi)
    for i in range(n_data_mbi):
        beta_bar[i] = beta_bar_step * i + beta_bar_lim[0]
    beta_bar = beta_bar[::-1]
    b_a = beta_bar / alpha_bar
    theta_crit = np.full(n_data_mbi, np.nan)
    skipped: List[SkipRecord] = []

    for i in range(n_data_mbi):
        print(f"[{model_name}] b_a = {b_a[i]}")
        overall_rot = MBI_OVERALL_ROT_START
        found = False
        while overall_rot <= MBI_OVERALL_ROT_MAX:
            print(f"[{model_name}] overall_rot = {overall_rot}")
            try:
                theta = mbi_indivtest(
                    model_name,
                    overall_rot=overall_rot,
                    alpha_val=alpha_bar[i],
                    beta_val=beta_bar[i],
                    do_render=do_render,
                    new_start=new_start and i == 0 and overall_rot == MBI_OVERALL_ROT_START,
                )
            except InstabilityError as exc:
                skipped.append(
                    SkipRecord(
                        key=f"b_a={b_a[i]:.4f}@rot={overall_rot:.2f}",
                        reason="instability",
                        detail=str(exc),
                    )
                )
                print(f"[{model_name}] Skipping unstable point b_a={b_a[i]:.4f} at rot={overall_rot:.2f}")
                break
            except Exception as exc:
                skipped.append(
                    SkipRecord(
                        key=f"b_a={b_a[i]:.4f}@rot={overall_rot:.2f}",
                        reason="error",
                        detail=str(exc),
                    )
                )
                print(f"[{model_name}] Error at b_a={b_a[i]:.4f}: {exc}")
                traceback.print_exc()
                break

            if theta > 1e-7:
                theta_crit[i] = theta
                found = True
                break
            overall_rot += MBI_OVERALL_ROT_STEP

        if not found and np.isnan(theta_crit[i]):
            skipped.append(
                SkipRecord(
                    key=f"b_a={b_a[i]:.4f}",
                    reason="max_rot",
                    detail=f"no buckling found up to rot={MBI_OVERALL_ROT_MAX}",
                )
            )
            print(f"[{model_name}] Skipping b_a={b_a[i]:.4f}: exceeded max overall_rot")

    return MbiResult(model=model_name, b_a=b_a, theta_crit=theta_crit, skipped=skipped)


def lhb_indivtest(
    model_name: str,
    r_pieces: int = 20,
    do_render: bool = False,
    new_start: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    alpha_val = 1.345
    beta_val = 0.789
    overall_rot = 27.0 * (2 * np.pi)
    r_thickness = 0.04
    r_len = 9.29

    env = create_validation_env(
        model_name,
        test_type="lhb",
        do_render=do_render,
        r_pieces=r_pieces,
        r_len=r_len,
        r_thickness=r_thickness,
        overall_rot=overall_rot,
        alpha_bar=_effective_alpha_bar(model_name, alpha_val),
        beta_bar=beta_val,
        new_start=new_start,
        limit_f=True,
    )
    try:
        _check_env_stability(env)
        legacy_path = _legacy_lhb_path(model_name, r_pieces)
        if not os.path.exists(legacy_path):
            raise FileNotFoundError(f"expected LHB output missing: {legacy_path}")
        with open(legacy_path, "rb") as f:
            pickledata = pickle.load(f)
        fphi = np.asarray(pickledata[0], dtype=float)
        s_ss = np.asarray(pickledata[1], dtype=float)
        if not _valid_numeric_array(fphi) or not _valid_numeric_array(s_ss):
            raise InstabilityError("non-finite LHB result arrays")
        return fphi, s_ss
    finally:
        close_env(env, do_render)


def run_lhb_sim(
    model_name: str,
    new_start: bool = False,
    do_render: bool = False,
) -> LhbResult:
    print(f"[{model_name}] Starting LHB test.")
    pieces: Dict[int, LhbPieceResult] = {}
    skipped: List[SkipRecord] = []

    for r_pieces in LHB_PIECES:
        print(f"[{model_name}] LHB test for {r_pieces} pieces.. ..")
        try:
            fphi, s_ss = lhb_indivtest(
                model_name,
                r_pieces=r_pieces,
                new_start=new_start,
                do_render=do_render,
            )
            pieces[r_pieces] = LhbPieceResult(status="ok", fphi=fphi, s_ss=s_ss)
        except InstabilityError as exc:
            skipped.append(
                SkipRecord(key=str(r_pieces), reason="instability", detail=str(exc))
            )
            pieces[r_pieces] = LhbPieceResult(
                status="skipped", reason="instability", detail=str(exc)
            )
            print(f"[{model_name}] Skipping LHB r_pieces={r_pieces}: {exc}")
        except Exception as exc:
            skipped.append(SkipRecord(key=str(r_pieces), reason="error", detail=str(exc)))
            pieces[r_pieces] = LhbPieceResult(
                status="skipped", reason="error", detail=str(exc)
            )
            print(f"[{model_name}] Skipping LHB r_pieces={r_pieces}: {exc}")
            traceback.print_exc()

    return LhbResult(
        model=model_name,
        r_pieces_list=LHB_PIECES.copy(),
        pieces=pieces,
        skipped=skipped,
    )


def mbi_plot(result: MbiResult) -> None:
    import matplotlib.pyplot as plt

    valid = np.isfinite(result.theta_crit)
    if not np.any(valid):
        print(f"[{result.model}] No valid MBI points to plot.")
        return

    b_a = result.b_a[valid]
    theta_crit = result.theta_crit[valid]
    b_a_base = b_a.copy()
    theta_crit_base = 2 * np.pi * np.sqrt(3) / b_a_base

    max_devi_theta_crit = np.max(np.abs(theta_crit_base - theta_crit))
    avg_deviation = np.linalg.norm(theta_crit_base - theta_crit) / len(theta_crit)
    print(f"[{result.model}] max_devi_theta_crit = {max_devi_theta_crit}")
    print(f"[{result.model}] avg_deviation = {avg_deviation}")
    if result.skipped:
        print(f"[{result.model}] skipped {len(result.skipped)} point(s)")

    plt.figure(f"Michell's Buckling Instability for {result.model}")
    plt.xlabel(r"$\beta/\alpha$")
    plt.ylabel(r"$\theta^n$")
    plt.plot(b_a_base, theta_crit_base, label="Analytical")
    plt.plot(b_a, theta_crit, label="Simulation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def _plot_lhb_piece(ax, fphi: np.ndarray, s_ss: np.ndarray, r_pieces: int, model_name: str) -> None:
    s_ss_base2 = s_ss.copy()
    fphi_base2 = (np.tanh(s_ss_base2)) ** 2.0
    id_counter = 0

    while id_counter < len(s_ss_base2):
        if s_ss_base2[id_counter] <= -6 or s_ss_base2[id_counter] > 6:
            s_ss_base2 = np.delete(s_ss_base2, id_counter, axis=0)
            fphi_base2 = np.delete(fphi_base2, id_counter, axis=0)
            fphi = np.delete(fphi, id_counter, axis=0)
        else:
            id_counter += 1

    print(f"[{model_name}] r_pieces = {r_pieces}")
    print(f"[{model_name}] fphi = {fphi}")
    print(f"[{model_name}] s_ss = {s_ss_base2}")

    avg_devi_lhb = np.linalg.norm(fphi - fphi_base2) / len(fphi_base2)
    print(f"[{model_name}] Average deviation for {r_pieces} pieces = {avg_devi_lhb}")

    s_ss_base2 = np.insert(s_ss_base2, 0, -6.0)
    fphi_base2 = np.insert(fphi_base2, 0, 1.0)
    fphi = np.insert(fphi, 0, 1.0)
    s_ss_base2 = np.insert(s_ss_base2, len(s_ss_base2), 6.0)
    fphi_base2 = np.insert(fphi_base2, len(fphi_base2), 1.0)
    fphi = np.insert(fphi, len(fphi), 1.0)
    ax.plot(s_ss_base2, fphi, alpha=0.5)


def lhb_plot(result: LhbResult) -> None:
    import matplotlib.pyplot as plt

    ok_pieces = [
        r for r in result.r_pieces_list
        if result.pieces[r].status == "ok"
    ]
    if not ok_pieces:
        print(f"[{result.model}] No valid LHB piece counts to plot.")
        return

    s_ss_base = np.arange(-6.0, 6.0, 0.01)
    fphi_base = (np.tanh(s_ss_base)) ** 2.0
    plt.rcParams.update({"pdf.fonttype": 42})
    plt.style.use("seaborn-v0_8")
    fig_dir = os.path.join(
        DATA_ROOT,
        "figs",
        data_subpath(result.model),
        f"lhb_{result.model}.pdf",
    )
    os.makedirs(os.path.dirname(fig_dir), exist_ok=True)

    fig = plt.figure(f"Localized Helical Buckling for {result.model}", figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.set_xlabel("s/s*")
    ax.set_ylabel(r"$f(\varphi)$")
    ax.plot(s_ss_base, fphi_base, color="k", linewidth="2", alpha=0.5)

    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.labelpad = 0.0

    legend_str = ["Analytical"]
    for r_pieces in ok_pieces:
        piece = result.pieces[r_pieces]
        _plot_lhb_piece(ax, piece.fphi, piece.s_ss, r_pieces, result.model)
        legend_str.append(str(r_pieces))

    if result.skipped:
        print(f"[{result.model}] skipped {len(result.skipped)} LHB piece count(s)")

    ax.legend(legend_str)
    plt.tight_layout()
    plt.savefig(fig_dir, bbox_inches="tight")
    plt.show()


def _run_model_worker(job: Tuple[str, str, bool, bool]) -> Tuple[str, str, Optional[str]]:
    model_name, test_type, new_start, do_render = job
    with _noninteractive_sim():
        try:
            if test_type == "mbi":
                payload = run_mbi_sim(model_name, new_start=new_start, do_render=do_render)
            elif test_type == "lhb":
                payload = run_lhb_sim(model_name, new_start=new_start, do_render=do_render)
            else:
                raise ValueError(f"Unknown test type: {test_type!r}")
            save_result(test_type, payload)
            return model_name, payload.status, None
        except Exception as exc:
            traceback.print_exc()
            return model_name, "failed", str(exc)


def run_simulations(
    model_names: List[str],
    test_type: str,
    *,
    new_start: bool,
    do_render: bool,
    parallel_workers: int,
) -> List[Tuple[str, str, Optional[str]]]:
    if do_render and parallel_workers > 1:
        print("Disabling render for parallel execution.")
        do_render = False

    jobs = [(name, test_type, new_start, do_render) for name in model_names]
    outcomes: List[Tuple[str, str, Optional[str]]] = []

    if parallel_workers <= 1:
        with _noninteractive_sim():
            for job in jobs:
                outcomes.append(_run_model_worker(job))
        return outcomes

    max_workers = min(parallel_workers, len(model_names))
    print(f"Running {len(model_names)} model(s) with {max_workers} worker(s)...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_model_worker, job): job[0] for job in jobs}
        for future in as_completed(futures):
            outcomes.append(future.result())
    return outcomes


def load_and_plot(model_names: List[str], test_type: str) -> None:
    for model_name in model_names:
        print(f"\n=== Plot {test_type.upper()}: {model_name} ===")
        try:
            payload = load_result(test_type, model_name)
        except FileNotFoundError as exc:
            print(f"[{model_name}] {exc}")
            continue
        if test_type == "mbi":
            mbi_plot(payload)
        else:
            lhb_plot(payload)


def main() -> None:
    parser = dtd_parse()
    _MODELS_HELP = ",".join(MODEL_REGISTRY.keys())
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=f"Comma-separated model names to run. Available: {_MODELS_HELP}. "
        f"Default: {','.join(DEFAULT_MODELS)}. Overrides --stiff when set.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="If >1, run one model per worker in parallel. Use 0/1 for sequential.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip simulation and only load saved results for plotting/metrics.",
    )
    args = parser.parse_args()

    if args.models is not None:
        model_names = parse_models_arg(args.models, DEFAULT_MODELS)
    elif args.stiff is not None:
        model_names = parse_models_arg(args.stiff, DEFAULT_MODELS)
    else:
        model_names = parse_models_arg(None, DEFAULT_MODELS)
    get_model_names(model_names)

    test_type = args.test
    do_render = bool(args.render)
    new_start = bool(args.newstart)
    plot_only = bool(args.plot_only) or bool(args.loadresults)
    parallel_workers = max(0, int(args.parallel))

    if test_type == "mbi" and args.newstart == 2:
        new_start = False

    if test_type not in ("mbi", "lhb"):
        raise ValueError(f"Unknown test type: {test_type!r}. Use 'mbi' or 'lhb'.")

    if not plot_only:
        outcomes = run_simulations(
            model_names,
            test_type,
            new_start=new_start,
            do_render=do_render,
            parallel_workers=parallel_workers,
        )
        for model_name, status, error in sorted(outcomes, key=lambda item: item[0]):
            if error:
                print(f"[{model_name}] simulation failed: {error}")
            else:
                print(f"[{model_name}] simulation status: {status}")

    load_and_plot(model_names, test_type)


if __name__ == "__main__":
    main()
