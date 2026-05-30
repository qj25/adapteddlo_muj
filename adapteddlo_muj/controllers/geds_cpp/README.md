# `geds_cpp`

C++/SWIG backend for **Geometrically Exact Dynamic Splines** (GEDS, Theetten et al., CAD 2008) used by the rope controllers.

This is a **Tier A** implementation: a uniform cubic **Catmull–Rom** centerline with GEDS-style **bending** (Frenet curvature) and **twisting** (material roll rate plus geometric Frenet twist). **Stretch is excluded** (no Eq. 17). MuJoCo still integrates the articulated chain; this module supplies elastic wrenches only.

## What it computes

Given rope site positions and body orientations from MuJoCo, `RodGeds` computes per-node:

1. **Elastic position forces** from the variational derivative of bend energy (Frenet κ vs rest κ) sampled along Catmull–Rom spans.
2. **Elastic roll torques** from twist energy (θ′ + τ vs rest), applied about the local tangent from a propagated minimal frame.

Rest strains (κ₀, twist₀) are captured at init/reset via `reinitRest`.

Forces are obtained with central finite differences on the total GEDS elastic energy (Catmull–Rom samples, Riemann `ds`). This keeps the implementation aligned with the paper’s energy (Sec. 5.3) without duplicating the full Appendix A chain rule in closed form.

## Model idea

- **Control points** = rope site positions (1:1 with `S_first … S_last`).
- **Open Catmull–Rom** endpoint neighbors: `P_{−1} = P_0`, `P_n = P_{n−1}`.
- At each sample on segment `[P_i, P_{i+1}]`:
  - `κ = ‖r′×r″‖ / ‖r′‖³` (bend strain, Eq. 22)
  - `τ_geom = (r′×r″)·r‴ / ‖r′×r″‖²` (Eq. 19)
  - `θ` = material **twist about tangent** from body quaternion (not in-plane roll)
  - `twist = θ′ + τ_geom` (Eq. 19)
- Energy density (no stretch):
  - `U_b = ½ k_bend (κ − κ₀)² ds`
  - `U_t = ½ k_twist (twist − twist₀)² ds`
- Stiffness (circular section, paper Eq. 7):
  - `k_bend = E π D⁴ / 64`
  - `k_twist = G π D⁴ / 32`

## Parameter roles (`alpha_bar`, `beta_bar`)

Mapped in `ropekin_controller_geds.py` (same pattern as XPBD):

- `D = r_thickness`, `r = D/2`
- `Ix = π r⁴ / 4`, polar term via `J1 = π r⁴ / 2`
- `E = (alpha_bar / Ix) * stiff_scale`
- `G = (beta_bar / J1) * stiff_scale`
- `stiff_scale = dt² * 1e2` (softens moduli for stable co-integration with MuJoCo)

Tune with `f_limit`, `k_p`/`k_d` endpoint tracking, and `setNumSamples(k)`.

## Force to torque mapping

Same pattern as `xpbd` / `adapt`:

1. `computeElasticWrenches` → `force_node`, `tau_roll` (world frame, about tangent).
2. `force2torq(force_node, adjacent site spacing, …)` builds torques from nodal forces.
3. Add roll torques; rotate into each body frame; write `qfrc_passive`.

Optional **endpoint position targets** (first/last chain bodies) are handled in Python via stiff PD on `xfrc_applied`, not in this C++ module.

## Python-facing API

SWIG module: `RodGeds`

Main calls:

- `RodGeds(n_nodes, segment_length, diameter, youngs_modulus, torsion_modulus)`
- `setMaterial(youngs, torsion)`, `setNumSamples(k_per_span)`
- `reinitRest(rest_x, rest_quat)` — after env reset / neutral capture
- `computeElasticWrenches(x, quat, force_out, torque_out)`

Array layout (NumPy, C-contiguous):

- positions: flat `[x0,y0,z0, …]`, length `3 * n_nodes`
- quaternions: flat `[w,x,y,z, …]`, length `4 * n_nodes`
- `force_out` / `torque_out`: flat `[fx,fy,fz, …]`, length `3 * n_nodes`

## Integration in this repo

Used by:

- `adapteddlo_muj/controllers/ropekin_controller_geds.py`
- `TestRopeEnv(..., model_name="geds")` in `our_rope_valid_test.py`

Registered for modular runners:

- `adapteddlo_muj/envs/speed_test/geds.py`
- `adapteddlo_muj/envs/simvreal_test/geds.py`

Outputs are applied on `qfrc_passive` (interior elastic) plus optional `xfrc_applied` endpoint PD in the controller.

## Validation

From the repo root (after building this module):

```bash
python adapteddlo_muj/controllers/geds_cpp/geds_validate.py
```

Runs Catmull–Rom, elastic force, twist, restoring-direction, MuJoCo stepping, and endpoint-hold checks.

## Build

From this directory (SWIG, Eigen, Python headers):

```bash
bash swigbuild.sh
```

Optional: `EIGEN_INCLUDE=/path/to/eigen` if Eigen is not at `$HOME/eigen`.

Generates and builds:

- `RodGeds_wrap.cpp`
- `RodGeds.py`
- `_RodGeds.so`
