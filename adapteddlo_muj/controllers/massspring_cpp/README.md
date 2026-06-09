# `massspring_cpp`

Lightweight C++/SWIG rope model that applies per-piece restoring torques from rotational deviation relative to a neutral pose.

## Model idea

For each joint between nodes `i-1` and `i`, we store neutral orientations and compute the
relative rotation deviation from the neutral relative rotation:

- `q_rel = q_{i-1}^{-1} * q_i`
- `q_rel0 = q_neutral_{i-1}^{-1} * q_neutral_i`
- `dev = rotvec(q_rel0^{-1} * q_rel)`

The deviation is split into bend and twist using the segment tangent `t`:

- `dev_twist = (dev · t) t`
- `dev_bend = dev - dev_twist`

Restoring torque in the parent frame:

- `tau_parent = -(k_bend * dev_bend + k_twist * dev_twist)`

where `k_bend = (k_bx + k_by) / 2`. The torque is then rotated into the child body frame.

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
- `computeTorque(current_x, current_quat, node_torque_out)`

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

