# `dlo_cpp`

C++/SWIG backend for the adapted Discrete Elastic Rod (DER) model used by the rope controllers.

## What it computes

Given rope node positions and boundary frame information, `DLO_iso` computes:

1. centerline elastic forces on nodes (`calculateCenterlineF2`),
2. local body torques (`calculateCenterlineTorq`) used by the adapted controller path.

The implementation follows DER-style bending/twist mechanics in a discretized rod.

## Core discrete geometry

For nodes `x_i` and edges `e_i = x_{i+1} - x_i`:

- edge length: `|e_i|`
- turning angle at interior node: `phi_i = angle(e_{i-1}, e_i)`
- scalar curvature-like term: `k_i = 2 * tan(phi_i / 2)`
- binormal curvature vector:
  - `kb_i = 2 * (e_{i-1} x e_i) / (|e_{i-1}||e_i| + e_{i-1} . e_i)`

These quantities are updated in:

- `updateX2E`
- `updateE2K`
- `updateE2Kb`

## Frames and twist state

- A Bishop frame is propagated along the rod (`transfBF`).
- End twist is tracked via `theta_n` / `overall_rot`.
- `updateTheta` unwraps incremental twist while handling `2*pi` periodicity.

## Elastic model (high level)

At each update, the code accumulates energy-gradient terms (bending and twisting couplings) and converts them into nodal forces.

Parameter roles:

- `alpha_bar`: bending modulus scaling
- `beta_bar`: twisting modulus scaling

These are set in constructor and can be changed via `changeAlphaBeta`.

## Force to torque mapping

`calculateCenterlineTorq` computes torque from distributed nodal forces by:

1. building pairwise node displacement data (`distmat`),
2. accumulating moment contributions `r x F`,
3. rotating global torques into each body's local frame using body quaternions.

The resulting torques are exposed back to Python as flattened `[tx, ty, tz, ...]`.

## Python-facing API

SWIG module: `Dlo_iso`

Main calls used by controllers:

- `updateVars(node_pos, bf0, bf_end_out)`
- `updateTheta(theta_n)`
- `calculateCenterlineF2(force_out)`
- `calculateCenterlineTorq(torque_out, body_quat, excl_joints)`
- `resetTheta(...)`
- `changeAlphaBeta(...)`

## Integration in this repo

Used by:

- `adapteddlo_muj/controllers/ropekin_controller_adapt.py`
- `adapteddlo_muj/controllers/ropekin_controller_xfrc.py`

where outputs are applied to either:

- `qfrc_passive` (torque/adapt path), or
- `xfrc_applied` / generalized conversion (force/xfrc path).

## Build

From this directory:

```bash
bash swigbuild.sh
```

This generates/builds:

- `Dlo_iso_wrap.cpp`
- `Dlo_iso.py`
- `_Dlo_iso.so`

