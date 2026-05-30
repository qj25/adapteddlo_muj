import numpy as np

import adapteddlo_muj.utils.transform_utils as T
from adapteddlo_muj.envs.test_shape_w_arm.base import R_PIECES, run_manipulation
from adapteddlo_muj.envs.test_shape_w_arm.registry import (
    DEFAULT_MODELS,
    MODEL_REGISTRY,
    get_model_specs,
    parse_models_arg,
)
from adapteddlo_muj.utils.argparse_utils import tswa_parse

# from adapteddlo_muj.utils.transform_utils import IDENTITY_QUATERNION

# init_pos = np.array([0.35, 0., 0.3])
# init_quat = np.array([0.,1.,0.,0.])
# init_qpos = np.array([
    # 1.23530872e-08,  5.05769873e-01,
    # 1.92988983e+00, -1.04436514e-07,
    # 7.05933183e-01,  3.14159125e+00
# ])
"""
DATA:   (A-adapted, X-direct, N-native)
white (last number is alpha row is the diff_pos)
-A- alpha = 6.047162706224905e-05 (stiff_scale = 0.015) - 0.007771766311904944
-A- beta = 7.438010128656632e-05 (b_a = 1.23)
-X- alpha = 6.047162706224905e-05 (stiff_scale = 0.015) - 0.007679568747663055
-X- beta = 7.438010128656632e-05 (b_a = 1.23)
-N- alpha = 7.619425009843381e-05 (stiff_scale = 0.0189) - 0.0068583062598774405
-N- beta = 7.771813510040248e-05 (b_a = 1.02)

black 
-A- alpha = 0.0009997975674291843 (stiff_scale = 0.248) - 0.01577049623151956
-A- beta = 0.001099777324172103 (b_a = 1.10)
-X- alpha = 0.0009997975674291843 (stiff_scale = 0.248) - 0.015712488291255093
-X- beta = 0.001099777324172103 (b_a = 1.10)
-N- alpha = 0.0010643006362955833 (stiff_scale = 0.264) - 0.015673797398902685
-N- beta = 0.0011175156681103625 (b_a = 1.05)

red
-A- alpha = 0.00063696780505569 (stiff_scale = 0.158) - 0.013824481117311245
-A- beta = 0.001203869151555254 (b_a = 1.89)
-X- alpha = 0.00063696780505569 (stiff_scale = 0.158) - 0.013749439033478634
-X- beta = 0.001203869151555254 (b_a = 1.89)
-N- alpha = 0.0006732507812930395 (stiff_scale = 0.167) - 0.013383626116227867
-N- beta = 0.0012118514063274711 (b_a = 1.80)
"""

getting_jointpos = False

parser = tswa_parse()
_MODELS_HELP = ",".join(MODEL_REGISTRY.keys())
parser.add_argument(
    "--models",
    type=str,
    default=None,
    help=f"Comma-separated model names to run. Available: {_MODELS_HELP}. "
    f"Default: {','.join(DEFAULT_MODELS)}. "
    "Overrides --stiff when set.",
)
args = parser.parse_args()

if args.models is not None:
    model_names = parse_models_arg(args.models, DEFAULT_MODELS)
elif args.stiff is not None:
    model_names = parse_models_arg(args.stiff, DEFAULT_MODELS)
else:
    model_names = parse_models_arg(None, DEFAULT_MODELS)

model_specs = get_model_specs(model_names)
wire_colors = [args.wirecolor] if args.wirecolor is not None else ["black", "red", "white"]
move_id = args.moveid
do_render = bool(args.render)

n_models = len(model_names)
n_wirecolors = len(wire_colors)

move_pos = np.array([
    [0.20, -0.1, -0.15],
    [0.095, 0.0, 0.0],
    [0.2, 0.1, 0.1],
    [0.2, 0.1, 0.1],
])
move_pos[:, 0] -= 0.035
move_aa = np.array([
    [30.0, -40.0, 0.0],
    [0.0, 0.0, 0.0],
    [50.0, 10.0, 0.0],
    [50.0, 20.0, -10.0],
])
z_rot = np.array([360.0, -720.0, 360.0, -720.0])

if move_id is not None:
    move_pos = np.array([move_pos[move_id]])
    move_aa = np.array([move_aa[move_id]])
    z_rot = np.array([z_rot[move_id]])

z_rot *= np.pi / 180
move_aa *= np.pi / 180
move_quat = np.zeros((len(move_aa), 4))
for i in range(len(move_aa)):
    move_quat[i] = T.axisangle2quat(move_aa[i])
n_pos = len(move_pos)

node_pos_arr = np.zeros((n_models, n_wirecolors, n_pos, R_PIECES + 1, 3))
joint_pos_arr = np.zeros((n_models, n_wirecolors, n_pos, 6))

for i, model_name in enumerate(model_names):
    model_spec = model_specs[model_name]
    for j, wire_color in enumerate(wire_colors):
        for pos_id in range(n_pos):
            i_move = pos_id if move_id is None else move_id
            velreset = wire_color == "red" and i_move == 2

            print(f"Now computing: {wire_color}{i_move}_{model_name}")
            overall_rot = None
            env = model_spec["create_env"](wire_color, overall_rot, do_render)
            env.velreset = velreset
            if getting_jointpos:
                env.max_action = 0.02

            joint_pos, nodes_pos = run_manipulation(
                env,
                move_pos,
                move_quat,
                z_rot,
                pos_id,
                getting_jointpos=getting_jointpos,
            )
            if do_render:
                env.viewer._paused = True
                env.hold_pos(1.0)

            joint_pos_arr[i, j, pos_id] = joint_pos
            node_pos_arr[i, j, pos_id] = nodes_pos

print("sim test csvdata saved!")
