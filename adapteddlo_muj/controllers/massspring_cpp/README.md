# `massspring_cpp`

Lightweight C++/SWIG rope model that applies per-piece restoring torques from rotational deviation relative to a neutral pose.

## Model idea

For each joint between nodes `i-1` and `i`, we store neutral orientations and compute the
relative rotation deviation from the neutral relative rotation:

- `q_rel = q_{i-1}^{-1} * q_i`
- `q_rel0 = q_neutral_{i-1}^{-1} * q_neutral_i`
- `dev = rotvec(q_rel0^{-1} * q_rel)`

The tangent component (twist about the segment axis) is removed using the segment tangent `t`:

- `dev_bend = dev - (dev · t) t`

Restoring torque in the parent frame (bend only):

- `tau_parent = -k_bend * dev_bend`

where `k_bend = (k_bx + k_by) / 2`. The torque is then rotated into the child body frame.

Twist stiffness is applied separately in Python via the cable-style layer in
`adapteddlo_muj/controllers/ropekin_controller_massspring.py` (matching `cable.cc`).

## Quaternion math used

- Quaternions are normalized; sign is canonicalized (`w >= 0`).
- Inverse uses conjugate (unit quaternion assumption).
- Relative quaternion is converted to axis-angle/rotation-vector via:
  - `angle = 2 * atan2(||v||, w)`
  - `rotvec = angle * v / ||v||` (with small-angle fallback `rotvec ~= 2v`)

## API surface

Defined in `MassSpring.h`:

- `MassSpring(neutral_quat, k_bend_x, k_bend_y)`
- `setNeutralQuat(neutral_quat)`
- `setStiffness(k_bend_x, k_bend_y)`
- `computeTorque(current_x, current_quat, node_torque_out)`

Array layout from Python:

- quaternions: flat `[w, x, y, z, ...]`
- torques: flat `[tx, ty, tz, ...]`

## Integration in this repo

The Python controller `adapteddlo_muj/controllers/ropekin_controller_massspring.py`:

1. captures neutral rope-body quaternions,
2. calls `computeTorque(...)` each step for bending,
3. applies cable-style twist torques via `mj_applyFT`,
4. writes resulting torques to MuJoCo generalized passive forces (`qfrc_passive`).

## Build

From this directory:

```bash
bash swigbuild.sh
```

This generates and builds:

- `MassSpring_wrap.cpp`
- `MassSpring.py`
- `_MassSpring.so`
