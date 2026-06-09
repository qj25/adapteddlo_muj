# `cosserat2_cpp`

C++/SWIG backend for the **Stable Cosserat** rod solver from
[StableCosseratRods](https://github.com/jerryhsu/StableCosseratRods) (SIGGRAPH 2025),
ported for MuJoCo coupling.

Unlike [`cosserat_cpp`](../cosserat_cpp/) (simplified node-quaternion shadow springs),
this module uses the reference **vertex + segment orientation** model:

- `iterateVertVBD()` — stretch constraints on vertex positions
- `iterateSegLambda()` — coupled stretch + bend/twist on segment orientations

## MuJoCo coupling

Each step runs a **shadow projection** (constraints solved on a copy of the chain):

1. Sync MuJoCo `xpos` / `xquat` into internal `Sim` (vertices + segments).
2. Run `num_iters` × (`iterateVertVBD` + `iterateSegLambda`) with zero inertia.
3. Map shadow deltas to wrenches:
   - `F_i = k_force * (x_shadow_i - x_i) / dt`
   - segment orientation delta → bend/twist torque on child bodies
4. Python controller (`ropekin_controller_cosserat2.py`) converts forces to joint
   torques (`force2torq`) and injects on `qfrc_passive`.

## Parameter mapping (`alpha_bar`, `beta_bar`)

For circular cross-section (`r = radius`, `dt = model.opt.timestep`):

- `stiff_scale = dt² * 1e2`
- `k_stretch = (alpha_bar / Ix) * stiff_scale` with `Ix = π r⁴/4`
- `k_bend = alpha_bar * stiff_scale` (rest-angle stiffness input)
- `k_twist = beta_bar * stiff_scale` (twist weight at wrench extraction)

Tune co-integration with `k_force`, `k_torque`, `num_iterations` on `DLORopeCosserat2`.

## Build

```bash
cd adapteddlo_muj/controllers/cosserat2_cpp
bash swigbuild.sh
```

Requires: g++, SWIG, Python dev headers, NumPy. GLM is vendored under `vendor/glm/`
(cloned automatically on first build if missing).

## Validation

```bash
conda activate mujenv
python cosserat2_validate.py
```

## Usage in scripts

```bash
python scripts/speed_test.py --models cosserat2
python scripts/test_shape_w_arm.py --models cosserat2
python scripts/real2sim_paramiden.py --models cosserat2
```
