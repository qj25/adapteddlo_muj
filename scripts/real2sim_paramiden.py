from adapteddlo_muj.envs.real2sim_paramiden.base import parse_lim_arg, search_limits
from adapteddlo_muj.envs.real2sim_paramiden.registry import (
    DEFAULT_MODELS,
    MODEL_REGISTRY,
    get_model_specs,
    parse_models_arg,
)
from adapteddlo_muj.envs.real2sim_paramiden.runner import run_bending, run_twisting
from adapteddlo_muj.utils.argparse_utils import r2spi_parse

parser = r2spi_parse()
_MODELS_HELP = ",".join(MODEL_REGISTRY.keys())
parser.add_argument(
    "--models",
    type=str,
    default=None,
    help=f"Comma-separated model names to run. Available: {_MODELS_HELP}. "
    f"Default: {','.join(DEFAULT_MODELS)}. "
    "Overrides --stiff when set.",
)
parser.add_argument(
    "--stiff-lim",
    type=str,
    default=None,
    help="Bending search range as lo,hi on stiff_scale. "
    "Default: [0,2] for adapt/native/xfrc, [0,20] for massspring/cosserat/xpbd/geds.",
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
test_id = args.testid
wire_color = args.wirecolor
test_type_g = args.testtype
do_render_g = bool(args.render)
new_start_g = bool(args.newstart)
lfp_g = bool(args.loadresults)

for model_name in model_names:
    default_stiff_lim, default_b_a_lim = search_limits(model_name)
    stiff_lim = parse_lim_arg(args.stiff_lim, default_stiff_lim)
    b_a_lim = parse_lim_arg(args.b_a_lim, default_b_a_lim)
    print(f"\n=== Parameter identification: {wire_color} / {model_name} ===")
    print(f"[{wire_color}/{model_name}] stiff_lim = {stiff_lim}")
    print(f"[{wire_color}/{model_name}] b_a_lim = {b_a_lim}")
    if test_type_g == "twisting":
        run_twisting(
            wire_color,
            model_name,
            do_render=do_render_g,
            new_start=new_start_g,
            loadresults=lfp_g,
            prompt=True,
            b_a_lim=b_a_lim,
        )
    elif test_type_g == "bending":
        run_bending(
            wire_color,
            model_name,
            test_id,
            do_render=do_render_g,
            new_start=new_start_g,
            loadresults=lfp_g,
            prompt=True,
            stiff_lim=stiff_lim,
        )
    else:
        raise ValueError(f"Unknown testtype: {test_type_g}. Use 'bending' or 'twisting'.")

print("\nParameter identification complete.")
