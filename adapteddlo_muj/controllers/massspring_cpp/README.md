# `massspring_cpp`

Lightweight C++/SWIG rope model that applies per-piece restoring torques from rotational deviation relative to a neutral pose.

## Model idea

For each rope piece `i`, we store a neutral orientation quaternion `q_neutral_i` (captured at init/reset).  
At runtime we read the current orientation `q_i` and compute the relative rotation:

- `q_rel_i = q_neutral_i^{-1} * q_i`

Then convert `q_rel_i` to a rotation-vector deviation:

- `dev_i = [dev_x, dev_y, dev_z]`

Interpretation:

- `dev_x`, `dev_y`: bending deviations around local `x`/`y`
- `dev_z`: twist deviation around local `z`

Restoring torque is linear in deviation:

- `tau_i = -diag(k_bx, k_by, k_twist) * dev_i`

or component-wise:

- `tau_x = -k_bx * dev_x`
- `tau_y = -k_by * dev_y`
- `tau_z = -k_twist * dev_z`

This is the rotational analog of a linear mass-spring law.

## Quaternion math used

- Quaternions are normalized; sign is canonicalized (`w >= 0`).
- Inverse uses conjugate (unit quaternion assumption).
- Relative quaternion is converted to axis-angle/rotation-vector via:
  - `angle = 2 * atan2(||v||, w)`
  - `rotvec = angle * v / ||v||` (with small-angle fallback `rotvec ~= 2v`)

## API surface

Defined in `MassSpring.h`:

- `MassSpring(neutral_quat, k_bend_x, k_bend_y, k_twist)`
- `setNeutralQuat(neutral_quat)`
- `setStiffness(k_bend_x, k_bend_y, k_twist)`
- `computeTorque(current_quat, node_torque_out)`

Array layout from Python:

- quaternions: flat `[w, x, y, z, ...]`
- torques: flat `[tx, ty, tz, ...]`

## Integration in this repo

The Python controller `adapteddlo_muj/controllers/ropekin_controller_massspring.py`:

1. captures neutral rope-body quaternions,
2. calls `computeTorque(...)` each step,
3. writes resulting torques to MuJoCo generalized passive forces (`qfrc_passive`).

## Build

From this directory:

```bash
bash swigbuild.sh
```

This generates and builds:

- `MassSpring_wrap.cpp`
- `MassSpring.py`
- `_MassSpring.so`

