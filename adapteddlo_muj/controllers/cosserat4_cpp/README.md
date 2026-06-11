# `cosserat4_cpp`

C++/SWIG backend for centerline elastic mechanics with **fullDyn twist** (MuJoCo `wire.cc` `fullDyn=true`).

## Twist model

Unlike `dlo_cpp` (quasistatic centerline twist), `RodCosserat4`:

- Stores twist angle `theta[i]` per link
- Updates interior links `1..nv-1` from body quaternions each step
- Uses adjacent twist difference `(theta[i] - theta[i-1]) / l_bar[i]` for twist stiffness

## Python API

SWIG module: `RodCosserat4`

- `RodCosserat4(node_pos, bf0, theta_n, overall_rot, alpha_bar, beta_bar)`
- `updateVars(...)`, `updateThetasFullDyn(body_quats)`, `initO2MLocAll(body_quats)`
- `calculateCenterlineTorq(torque_out, body_quat, excl_joints)`
- `resetTheta(...)`, `changeAlphaBeta(...)`

## Controller

`adapteddlo_muj/controllers/ropekin_controller_cosserat4.py`

## Build

```bash
bash swigbuild.sh
```

Or:

```bash
bash scripts/build_rope_backends.sh cosserat4_cpp
```
