import os
import pickle
from typing import List, Optional, Sequence, Tuple

import numpy as np

from adapteddlo_muj.envs.real2sim_paramiden.base import (
    ROPE_LEN,
    bendstiff_path,
    realdata_path,
    search_limits,
    stiff_path,
    twisting_params,
    wire_params,
)
from adapteddlo_muj.utils.optimize_utils import golden_section_search, midpoint_rootfind, mbi_stiff

DEFAULT_TEST_IDS = ["0", "1", "2", "3", "4"]


def load_bending_alpha(wire_color: str, model_name: str, test_ids: Sequence[str]) -> float:
    alpha_all = []
    for test_id in test_ids:
        path = bendstiff_path(wire_color, model_name, test_id)
        with open(path, "rb") as f:
            alpha_all.append(pickle.load(f))
    alpha_all = np.array(alpha_all)
    alpha_glob = float(np.mean(alpha_all))
    alpha_cv = np.sum(np.linalg.norm(alpha_all - alpha_glob)) / len(alpha_all) / alpha_glob
    print(f"[{wire_color}/{model_name}] alpha_all = {alpha_all}")
    print(f"[{wire_color}/{model_name}] alpha_cv = {alpha_cv * 100.0}%")
    print(f"[{wire_color}/{model_name}] alpha_mean = {alpha_glob}")
    return alpha_glob


def run_bending(
    wire_color: str,
    model_name: str,
    test_id: str,
    *,
    do_render: bool = False,
    new_start: bool = False,
    loadresults: bool = False,
    grav_on: bool = True,
    rope_len: float = ROPE_LEN,
    stiff_lim: Optional[np.ndarray] = None,
    prompt: bool = True,
) -> Tuple[float, float]:
    if stiff_lim is None:
        stiff_lim, _ = search_limits(model_name, wire_color)
    os.makedirs(os.path.dirname(stiff_path(wire_color, model_name)), exist_ok=True)
    massperlen, rgba_vals = wire_params(wire_color)
    bendstiff_picklename = bendstiff_path(wire_color, model_name, test_id)
    stiff_picklename = stiff_path(wire_color, model_name)

    with open(realdata_path(wire_color, test_id), "rb") as f:
        real_pos, _ = pickle.load(f)

    mbi_stiff1 = mbi_stiff(
        model_name=model_name,
        rgba_vals=rgba_vals,
        real_pos=real_pos,
        massperlen=massperlen,
        wire_color=wire_color,
        overall_rot=0.0,
        r_len=rope_len,
        do_render=do_render,
        new_start=new_start,
        grav_on=grav_on,
    )

    if loadresults:
        with open(stiff_picklename, "rb") as f:
            alpha_glob, b_a_glob = pickle.load(f)
        print(f"[{wire_color}/{model_name}] alpha = {alpha_glob}")
        print(f"[{wire_color}/{model_name}] beta = {alpha_glob * b_a_glob}")
        print(f"[{wire_color}/{model_name}] b/a = {b_a_glob}")
        if prompt:
            input("Press 'Enter' to run experiment.. ..")
        min_diff = mbi_stiff1.opt_func(alpha_glob * (2 * np.pi) ** 3)
        print(f"[{wire_color}/{model_name}] diff = {min_diff}")
        if do_render and mbi_stiff1.env is not None:
            mbi_stiff1.env.viewer._paused = True
            for _ in range(100):
                mbi_stiff1.env.step()
        return alpha_glob, min_diff

    best_stiffval = golden_section_search(
        mbi_stiff1.opt_func,
        stiff_lim[0],
        stiff_lim[1],
        tol=1e-3,
    )
    min_diff = mbi_stiff1.opt_func(best_stiffval)
    alpha_stiff = best_stiffval / (2 * np.pi) ** 3
    print(f"[{wire_color}/{model_name}] Minimum Found!")
    print(f"[{wire_color}/{model_name}] final alpha = {alpha_stiff}")
    print(f"[{wire_color}/{model_name}] stiff_scale = {best_stiffval}")
    print(f"[{wire_color}/{model_name}] min_diff = {min_diff}")

    with open(bendstiff_picklename, "wb") as f:
        pickle.dump(alpha_stiff, f)
    print(f"[{wire_color}/{model_name}] Saved: {bendstiff_picklename}")
    return alpha_stiff, min_diff


def run_twisting(
    wire_color: str,
    model_name: str,
    *,
    test_ids: Sequence[str] = DEFAULT_TEST_IDS,
    do_render: bool = False,
    new_start: bool = False,
    loadresults: bool = False,
    grav_on: bool = True,
    rope_len: float = ROPE_LEN,
    b_a_lim: Optional[np.ndarray] = None,
    prompt: bool = True,
) -> Tuple[float, float]:
    if b_a_lim is None:
        _, b_a_lim = search_limits(model_name, wire_color)
    os.makedirs(os.path.dirname(stiff_path(wire_color, model_name)), exist_ok=True)
    massperlen, rgba_vals = wire_params(wire_color)
    ord_glob, b_a_arr = twisting_params(wire_color)
    stiff_picklename = stiff_path(wire_color, model_name)
    deg2rad = np.pi / 180.0

    if loadresults:
        alpha_glob = load_bending_alpha(wire_color, model_name, test_ids)
        b_a_avg = float(np.mean(b_a_arr))
        b_a_cv = np.sum(np.linalg.norm(b_a_arr - b_a_avg)) / len(b_a_arr) / b_a_avg
        print(f"[{wire_color}/{model_name}] beta_all = {b_a_arr}")
        print(f"[{wire_color}/{model_name}] beta_cv = {b_a_cv * 100.0}%")
        print(f"[{wire_color}/{model_name}] beta_mean = {b_a_avg}")
        if prompt:
            input()

        with open(stiff_picklename, "rb") as f:
            alpha_glob, b_a_glob = pickle.load(f)
        print(f"[{wire_color}/{model_name}] alpha = {alpha_glob}")
        print(f"[{wire_color}/{model_name}] beta = {alpha_glob * b_a_glob}")
        print(f"[{wire_color}/{model_name}] b/a = {b_a_glob}")
        if prompt:
            input()
    else:
        alpha_glob = load_bending_alpha(wire_color, model_name, test_ids)
        if prompt:
            input('Press "Enter" to run twisting experiment.. ..')

    mbi_stiff2 = mbi_stiff(
        model_name=model_name,
        rgba_vals=rgba_vals,
        massperlen=massperlen,
        wire_color=wire_color,
        overall_rot=ord_glob * deg2rad,
        r_len=rope_len,
        do_render=do_render,
        new_start=new_start,
        grav_on=grav_on,
    )
    mbi_stiff2.alpha_bar = alpha_glob

    if loadresults:
        print(f"[{wire_color}/{model_name}] diff = {mbi_stiff2.opt_func2(b_a_glob)}")
        if do_render and mbi_stiff2.env is not None:
            mbi_stiff2.env.viewer._paused = True
            for _ in range(100):
                mbi_stiff2.env.step()
        return alpha_glob, b_a_glob

    best_b_a_val = midpoint_rootfind(
        mbi_stiff2.opt_func2,
        b_a_lim[0],
        b_a_lim[1],
        tol=1e-2,
    )
    print(f"[{wire_color}/{model_name}] Beta/Alpha Found!")
    print(f"[{wire_color}/{model_name}] b/a = {best_b_a_val}")
    stiff_pickle = [alpha_glob, best_b_a_val]
    with open(stiff_picklename, "wb") as f:
        pickle.dump(stiff_pickle, f)
    print(f"[{wire_color}/{model_name}] Saved: {stiff_picklename}")
    return alpha_glob, best_b_a_val


def run_full_paramiden(
    wire_colors: Sequence[str],
    model_names: Sequence[str],
    *,
    test_ids: Sequence[str] = DEFAULT_TEST_IDS,
    do_render: bool = False,
    new_start: bool = False,
    loadresults: bool = False,
    skip_bending: bool = False,
    skip_twisting: bool = False,
    prompt: bool = False,
    stiff_lim: Optional[np.ndarray] = None,
    b_a_lim: Optional[np.ndarray] = None,
) -> None:
    for wire_color in wire_colors:
        for model_name in model_names:
            model_stiff_lim = stiff_lim
            model_b_a_lim = b_a_lim
            if model_stiff_lim is None or model_b_a_lim is None:
                default_stiff_lim, default_b_a_lim = search_limits(
                    model_name, wire_color
                )
                if model_stiff_lim is None:
                    model_stiff_lim = default_stiff_lim
                if model_b_a_lim is None:
                    model_b_a_lim = default_b_a_lim
            print(f"\n=== Parameter identification: {wire_color} / {model_name} ===")
            print(f"[{wire_color}/{model_name}] stiff_lim = {model_stiff_lim}")
            print(f"[{wire_color}/{model_name}] b_a_lim = {model_b_a_lim}")
            if not skip_bending:
                for test_id in test_ids:
                    print(f"\n--- Bending testid {test_id} ---")
                    run_bending(
                        wire_color,
                        model_name,
                        test_id,
                        do_render=do_render,
                        new_start=new_start,
                        loadresults=loadresults,
                        prompt=prompt,
                        stiff_lim=model_stiff_lim,
                    )
            if not skip_twisting:
                print("\n--- Twisting ---")
                run_twisting(
                    wire_color,
                    model_name,
                    test_ids=test_ids,
                    do_render=do_render,
                    new_start=new_start,
                    loadresults=loadresults,
                    prompt=prompt,
                    b_a_lim=model_b_a_lim,
                )
