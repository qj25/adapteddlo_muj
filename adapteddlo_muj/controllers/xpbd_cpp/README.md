# `xpbd_cpp`

C++/SWIG backend for stretch–bend–twist (SBT) stiff-rod constraints from
[PositionBasedDynamics](https://github.com/InteractiveComputerGraphics/PositionBasedDynamics)
(`DirectPositionBasedSolverForStiffRods`), used by the XPBD rope controller.

## What it computes

Given capsule-segment poses and inertias from MuJoCo, `RodXpbd` runs a **shadow**
XPBD projection (constraints solved on a copy of the chain, not written back into
`data`) and returns per-segment:

1. world-frame nodal forces (`force_out`) that push toward the projected center-of-mass positions,
2. world-frame torque proxies (`torque_out`) from the projected quaternion change.

The Python controller converts forces to joint torques (`force2torq`), adds the
direct torque terms in each body frame, and applies the result on `qfrc_passive`
before `mj_step`.

MuJoCo still integrates the articulated chain; this module is a **predictor +
wrench bridge**, not a standalone simulator.

## Model idea

For `n` segments there are `n - 1` SBT joints between consecutive bodies.
Each joint is initialized at rest from segment centers `x_i`, orientations `q_i`
(MuJoCo order `w, x, y, z`), and a joint anchor at the midpoint
`(x_i + x_{i+1}) / 2`.

Each simulation step:

1. Copy `x_i`, `q_i` into shadow state; set `inv_mass_i = 0` on welded ends when `bothweld`.
2. For each joint, call `initBeforeProjection_StretchBendingTwistingConstraint` (resets XPBD `lambdaSum`, updates compliances from `dt`).
3. Repeat `num_iterations` times: `update_...` then `solve_StretchBendingTwistingConstraint`, applying `corr_x`, `corr_q` on the shadow chain only.
4. Map shadow deltas to wrenches:
   - `F_i = k_force * (x_shadow_i - x_i) / dt`
   - `tau_i = k_torque * rotvec(q_shadow_i * q_i^{-1}) / dt`

Compliance and rest Darboux vectors come from the vendored PBD rod solver; see
`PositionBasedElasticRods.h` (eq. 23–24 in the stiff-rod paper referenced there).

## Parameter roles (`alpha_bar`, `beta_bar`)

Mapped in `ropekin_controller_xpbd.py` for a circular cross-section
(`r = r_thickness / 2`, segment length `L = r_len / r_pieces`):

- `Ix = π r⁴ / 4`, `J1 = π r⁴ / 2`
- `E = (alpha_bar / Ix) * stiff_scale` → `youngs_modulus` in `RodXpbd`
- `G = (beta_bar / J1) * stiff_scale` → `torsion_modulus` in `RodXpbd`
- `stiff_scale = dt² * 1e2` (softens moduli for stable co-integration with MuJoCo)

Also passed into C++: `segment_length` (`L`), `radius` (`r`), and `bothweld`
(static ends via zero inverse mass).

Tune stability with `k_force`, `k_torque`, `num_iterations` on `DLORopeXpbd`, and
`f_limit` (nodal force clamp) in the controller.

## Force to torque mapping

Same pattern as `adapt`:

1. `computeWrenches` fills `force_node`, `tau_direct` (world frame).
2. `force2torq(force_node, adjacent site spacing, ...)` builds torques from nodal forces.
3. `tau_direct` is rotated into each body frame (`mju_rotVecQuat` + inverse body quat) and added.
4. Torques are scattered into `qfrc_passive` (with welded-end indexing when `bothweld`).

## API surface

Defined in `RodXpbd.h` (SWIG module `RodXpbd`):

- `RodXpbd(n_segments, rest_x, rest_quat, segment_length, radius, youngs, torsion, bothweld)`
- `setMaterial(youngs_modulus, torsion_modulus)`
- `setForceGain(k_force)`, `setTorqueGain(k_torque)`, `setNumIterations(num_iters)`
- `reinitRestPose(rest_x, rest_quat)` — after env reset / neutral capture
- `computeWrenches(x, quat, inv_mass, inv_inertia_w, dt, force_out, torque_out)`

Array layout from Python (NumPy, C-contiguous):

- positions: flat `[x0, y0, z0, x1, y1, z1, ...]`, length `3 * n_segments`
- quaternions: flat `[w, x, y, z, ...]`, length `4 * n_segments`
- `inv_mass`: length `n_segments` (`0` = kinematic / welded)
- `inv_inertia_w`: flat `3×3` row-major per segment, length `9 * n_segments`
- `force_out` / `torque_out`: flat `[fx, fy, fz, ...]` / `[tx, ty, tz, ...]`

## Vendored sources

Under `vendor/pbd/` (minimal extract from `tmp/PositionBasedDynamics`):

- `Common/Common.h` — Eigen `Real`, `Vector3r`, `Quaternionr`, …
- `PositionBasedDynamics/MathFunctions.{h,cpp}`
- `PositionBasedDynamics/PositionBasedElasticRods.{h,cpp}`
- `PositionBasedDynamics/DirectPositionBasedSolverForStiffRodsInterface.h`
- `Utils/Logger.h` — empty stub (logger unused in this build)

Not included: full `TimeStepController`, particle/tet constraints, collision, GLFW/pyPBD.

## Integration in this repo

Used by:

- `adapteddlo_muj/controllers/ropekin_controller_xpbd.py`
- `TestRopeEnv(..., model_name="xpbd")` in `our_rope_valid_test.py`

Registered for modular runners:

- `adapteddlo_muj/envs/speed_test/xpbd.py`
- `adapteddlo_muj/envs/simvreal_test/xpbd.py` (falls back to adapt pickles until `simdata_*_xpbd.pickle` exist)

Outputs are applied on `qfrc_passive` (family **E + T**), same hook as `adapt` and `massspring`.

## Build

From this directory (with `mujenv` or any env that has SWIG, Eigen, Python headers):

```bash
bash swigbuild.sh
```

Optional: `EIGEN_INCLUDE=/path/to/eigen` if Eigen is not at `$HOME/eigen`.

This generates and builds:

- `RodXpbd_wrap.cpp`
- `RodXpbd.py`
- `_RodXpbd.so`
