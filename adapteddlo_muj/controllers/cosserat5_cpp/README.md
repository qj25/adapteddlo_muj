# `cosserat5_cpp`

C++/SWIG backend for **JTill2017 Kirchhoff rod** elastic mechanics (inextensible case), paired with **cable-style twist** in Python.

This is the **manual-rotation** Kirchhoff variant used alongside `massspring`: overall twist is imposed by rotating rope ends (`TestRopeManRotEnv`), while bending is evaluated quasi-statically from propagated Bishop frames.

## What it computes

Given rope site positions and boundary Bishop-frame data from MuJoCo, `RodCosserat5` evaluates:

1. **Edge geometry** and **Bishop frames** along the centerline (`updateX2E`, `updateE2K`, `updateE2Kb`, `transfBF`).
2. **Material curvature** `u` in the material frame (bend components from discrete Frenet/Bishop geometry; twist rate from distributed `theta` along edges).
3. **Bending nodal forces** from the gradient of `½ k_bend |u_bend − u*_bend|²` (twist component excluded via `kbEffectiveAtNode`).
4. **Bending torques** via `r × F` lever arms, rotated into each body frame (`calculateCenterlineTorq`).

**Twist stiffness is not applied in this C++ module.** The Python controller adds MuJoCo `cable.cc`-style twist torques (`mju_quat2Vel` + `mj_applyFT`) using `beta_bar`, matching `ropekin_controller_massspring.py`.

Stretch and shear are omitted; segment lengths follow the articulated MuJoCo chain.

## Model idea

- **Inextensible Kirchhoff rod** (JTill2017): constitutive stiffness `K = diag(k_bend, k_bend, k_twist)` with `k_bend = alpha_bar`, `k_twist = beta_bar` (same tuning convention as `dlo_cpp` / `adapt`).
- **Rest strains** `u*` and reference twist distribution are captured at init/reset via `captureRestCurvature()`.
- **End rotation**: `overall_rot` and measured end angle `theta_n` drive the interior twist-angle field; `updateTheta` redistributes `theta` uniformly along the rod before curvature is recomputed.
- **Bending energy** (C++ only):

  - `U_b = Σ ½ k_bend (Δu_x² + Δu_y²) l_bar`

  where `Δu = u − u*` and only the first two material-frame components contribute.

## Parameter roles (`alpha_bar`, `beta_bar`)

- `alpha_bar` → `k_bend_` in C++ (bending nodal forces / torques).
- `beta_bar` → `k_twist` in Python (cable twist stiffness via `J * G` with `J = π r⁴ / 2`).

Mapped in `ropekin_controller_cosserat5.py` from env `alpha_bar` / `beta_bar` (typically `alpha_val` / `beta_val` from test scripts).

## Force to torque mapping

Same pattern as `cosserat3` / `adapt`:

1. `calculateCenterlineTorq(...)` → bending torques from Kirchhoff nodal forces.
2. Write torques to `qfrc_passive` (interior joints; endpoint handling via `bothweld`).
3. `_apply_cable_twist_torques()` adds twist contribution separately on `qfrc_passive`.

## Python-facing API

SWIG module: `RodCosserat5`

Main calls:

- `RodCosserat5(node_pos, bf0, theta_n, overall_rot, alpha_bar, beta_bar, radius)`
- `updateVars(node_pos, bf0, bf_end_out)` — refresh geometry and end Bishop frame
- `updateTheta(theta_n)` / `resetTheta(theta_n, overall_rot)` / `changeAlphaBeta(alpha_bar, beta_bar)`
- `captureRestCurvature()` — store rest `u*` after env reset
- `calculateCenterlineTorq(torque_out, body_quat, excl_joints)`
- `initQe_o2m_loc(q_o2m)`, `calculateOf2Mf(mat_o, mat_res)`, `angBtwn3(...)`
- `calculateEnergy()` — bending + twisting energy (diagnostics)

Array layout (NumPy, C-contiguous):

- positions: flat `[x0,y0,z0, …]`, length `3 * n_nodes`
- `bf0` / `bf_end`: `3×3` Bishop frame at start / end, flattened row-major
- quaternions: flat `[w,x,y,z, …]`, length `4 * n_nodes`
- torques: flat `[tx,ty,tz, …]`, length `3 * n_nodes`

## Integration in this repo

Used by:

- `adapteddlo_muj/controllers/ropekin_controller_cosserat5.py`
- `TestRopeManRotEnv(..., model_name="cosserat5")` in `our_rope_manrot_valid_test.py`

Registered for modular runners:

- `adapteddlo_muj/envs/speed_test/cosserat5.py`
- `adapteddlo_muj/envs/simvreal_test/` (via shared registry)
- `adapteddlo_muj/envs/test_shape_w_arm/cosserat5.py`
- `adapteddlo_muj/envs/validation_test/registry.py`

In speed/sim-vs-real plots this model is labeled **`cosserat`** (alias for `cosserat5`).

Supports `--adapt_pickle` on speed/validation runs to reuse adapt LHB init pickles (see `argparse_utils.py`).

## Build

From this directory (SWIG, Eigen, Python headers):

```bash
bash swigbuild.sh
```

Or from repo root:

```bash
bash scripts/build_rope_backends.sh cosserat5_cpp
```

Optional: `EIGEN_INCLUDE=/path/to/eigen` if Eigen is not at `$HOME/eigen`.

Generates and builds:

- `RodCosserat5_wrap.cpp`
- `RodCosserat5.py`
- `_RodCosserat5.so`
