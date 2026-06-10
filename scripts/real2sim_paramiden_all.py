import argparse

from adapteddlo_muj.envs.real2sim_paramiden.base import parse_lim_arg, search_limits
from adapteddlo_muj.envs.real2sim_paramiden.registry import (
    DEFAULT_MODELS,
    MODEL_REGISTRY,
    get_model_specs,
    parse_models_arg,
)
from adapteddlo_muj.envs.real2sim_paramiden.runner import DEFAULT_TEST_IDS, run_full_paramiden

WIRE_COLORS = ["white", "black", "red"]


def parse_wirecolors_arg(wirecolors_arg):
    if wirecolors_arg is None or wirecolors_arg.strip() == "":
        return ["white"]
    return [c.strip() for c in wirecolors_arg.split(",") if c.strip()]


def parse_testids_arg(testids_arg):
    if testids_arg is None or testids_arg.strip() == "":
        return DEFAULT_TEST_IDS
    return [t.strip() for t in testids_arg.split(",") if t.strip()]


def main():
    models_help = ",".join(MODEL_REGISTRY.keys())
    parser = argparse.ArgumentParser(
        description="Run full real-to-sim parameter identification: "
        "bending (testid 0-4) then twisting for each model."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=f"Comma-separated model names. Available: {models_help}. "
        f"Default: {','.join(DEFAULT_MODELS)}.",
    )
    parser.add_argument(
        "--stiff",
        type=str,
        default=None,
        help="Legacy alias for --models.",
    )
    parser.add_argument(
        "--wirecolor",
        type=str,
        default=None,
        help="Single wire color: white, black, or red. Default: white.",
    )
    parser.add_argument(
        "--wirecolors",
        type=str,
        default=None,
        help="Comma-separated wire colors. Overrides --wirecolor when set.",
    )
    parser.add_argument(
        "--testids",
        type=str,
        default=None,
        help="Comma-separated bending test ids. Default: 0,1,2,3,4.",
    )
    parser.add_argument("--render", type=int, default=0, help="Render mode: 0 (off) or 1 (on).")
    parser.add_argument(
        "--newstart",
        type=int,
        default=0,
        help="Re-init MBI env pickle per model: 0 (off) or 1 (on).",
    )
    parser.add_argument(
        "--loadresults",
        type=int,
        default=0,
        help="Load saved stiffness and visualize instead of optimizing.",
    )
    parser.add_argument(
        "--skip-bending",
        action="store_true",
        help="Skip bending phase (use existing bendstiff pickles).",
    )
    parser.add_argument(
        "--skip-twisting",
        action="store_true",
        help="Skip twisting phase (only run bending).",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Pause for Enter between steps (default: run unattended).",
    )
    parser.add_argument(
        "--stiff-lim",
        type=str,
        default=None,
        help="Bending search range as lo,hi on stiff_scale for all models. "
        "Default: [0,2] for all models.",
    )
    parser.add_argument(
        "--b-a-lim",
        type=str,
        default=None,
        help="Twisting search range as lo,hi on beta/alpha ratio. "
        "Default matches --stiff-lim per model.",
    )
    args = parser.parse_args()

    if args.models is not None:
        model_names = parse_models_arg(args.models, DEFAULT_MODELS)
    elif args.stiff is not None:
        model_names = parse_models_arg(args.stiff, DEFAULT_MODELS)
    else:
        model_names = parse_models_arg(None, DEFAULT_MODELS)

    get_model_specs(model_names)

    if args.wirecolors is not None:
        wire_colors = parse_wirecolors_arg(args.wirecolors)
    elif args.wirecolor is not None:
        wire_colors = [args.wirecolor]
    else:
        wire_colors = ["white"]

    unknown_colors = [c for c in wire_colors if c not in WIRE_COLORS]
    if unknown_colors:
        raise ValueError(f"Unknown wire color(s): {unknown_colors}. Use: {WIRE_COLORS}")

    test_ids = parse_testids_arg(args.testids)

    stiff_lim = (
        parse_lim_arg(args.stiff_lim, search_limits(model_names[0])[0])
        if args.stiff_lim is not None
        else None
    )
    b_a_lim = (
        parse_lim_arg(args.b_a_lim, search_limits(model_names[0])[1])
        if args.b_a_lim is not None
        else None
    )

    run_full_paramiden(
        wire_colors,
        model_names,
        test_ids=test_ids,
        do_render=bool(args.render),
        new_start=bool(args.newstart),
        loadresults=bool(args.loadresults),
        skip_bending=args.skip_bending,
        skip_twisting=args.skip_twisting,
        prompt=args.prompt,
        stiff_lim=stiff_lim,
        b_a_lim=b_a_lim,
    )
    print("\nFull parameter identification complete.")


if __name__ == "__main__":
    main()
