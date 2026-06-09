# `cosserat3_cpp`

C++/SWIG backend for **JTill2017 Kirchhoff rod** elastic mechanics, integrated with MuJoCo like `dlo_cpp`.

## What it computes

Given rope site positions and Bishop-frame boundary data, `RodCosserat3` evaluates quasi-static elastic wrenches in one pass (no shadow solve):

1. Material-frame curvature `u` from propagated Bishop frames (`R_s = R û`, inextensible `v = e_3`).
2. Constitutive moment `m = K(u - u*)` (Kelvin stiffness without damping).
3. Nodal forces from energy gradients of `½ (u - u*)ᵀ K (u - u*)`.
4. Body torques via `r × F` and quaternion rotation (adapt path).

Stretch and shear are omitted; segment length is fixed by the MuJoCo chain.

## Python-facing API

SWIG module: `RodCosserat3`

Main calls (mirror `DLO_iso`):

- `RodCosserat3(node_pos, bf0, theta_n, overall_rot, alpha_bar, beta_bar, radius)`
- `updateVars(node_pos, bf0, bf_end_out)`
- `updateTheta(theta_n)` / `resetTheta(...)` / `changeAlphaBeta(...)`
- `captureRestCurvature()` — store rest `u*` after reset
- `calculateCenterlineF2(force_out)`
- `calculateCenterlineTorq(torque_out, body_quat, excl_joints)`
- `calculateEnergy()`

## Integration

- Controller: `adapteddlo_muj/controllers/ropekin_controller_cosserat3.py`
- Env: `TestRopeEnv(..., model_name="cosserat3")`

## Build

```bash
bash swigbuild.sh
```

Or from repo root:

```bash
bash scripts/build_rope_backends.sh cosserat3_cpp
```
