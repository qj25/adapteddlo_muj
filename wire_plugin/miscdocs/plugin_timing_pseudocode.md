# MuJoCo Passive Plugin Pseudocode

## Table 1: Passive Plugin Call Flow per Step

```
mj_step(m,d)
  → mj_forward(m,d)
    → mj_fwdVelocity(m,d)
      → mj_passive(m,d)
        FOR i = 0 TO nplugin-1:
          IF plugin[i].capabilityflags & mjPLUGIN_PASSIVE:
            plugin[i].compute(m, d, i, mjPLUGIN_PASSIVE)
```

## Table 2: Plugin Compute Pseudocode

### Wire (mujoco.elasticity.wire)

| Step | Pseudocode |
|------|------------|
| 1 | `updateVars(d); updateBishopFrame(d);` |
| 2 | `IF fullDyn: FOR bwi: updateTheta(θ_loc,bwi)` |
| 3 | `ELIF pqsActive: detectInteractions; updateTheta/splitTheta per segment` |
| 4 | `ELSE: updateTheta(θ_loc,nv); splitTheta(0,nv)` |
| 5 | `updateMatFrame();` |
| 6 | `FOR nodes: compute curvature ω, material frame;` |
| 7 | `compute bend/twist torques (dE/dθ, dE/dγ);` |
| 8 | `mj_applyFT(m,d, 0, lfrc, xpos, body, qfrc_passive)` |

### WireQST (mujoco.elasticity.wire_qst)

| Step | Pseudocode |
|------|------------|
| 1 | `updateVars(d); UpdateBishopFrame(d); θ_n = get_thetan(d); updateTheta(θ_n);` |
| 2 | `populate distmat (distance matrix);` |
| 3 | `FOR nodes: compute curvature k, kb; nabkb, nabpsi;` |
| 4 | `force[j] += -α∇kb·kb/lbar + βΔθ·∇ψ/Lbar;` |
| 5 | `torqvec = distmat × force; torq = rot(torqvec, quat_inv);` |
| 6 | `apply torques to d->qfrc_passive` |

### Cable (mujoco.elasticity.cable)

| Step | Pseudocode |
|------|------------|
| 1 | `FOR each body b:` |
| 2 | `  IF stiffness[b]==0: continue` |
| 3 | `  QuatDiff(quat, body_quat, joint_quat);` |
| 4 | `  LocalStress(stress, stiffness, quat, omega0) → elastic torque` |
| 5 | `  lfrc += stress (from prev/next neighbors);` |
| 6 | `  mj_applyFT(m,d, 0, lfrc, xpos, body, qfrc_passive)` |

---

## Summary: Key Helper Functions

### QuatDiff

Computes the quaternion difference between two frames in joint coordinates.

- **Inputs**: `body_quat` (body orientation), `joint_quat` (joint orientation)
- **Output**: `quat` = relative orientation between the two frames
- **pullback=false**: `quat = body_quat * joint_quat` — orientation in local/body frame
- **pullback=true**: Same product, then negated — orientation pulled back into the neighboring body’s frame
- **Purpose**: Gives the relative rotation between adjacent bodies for computing elastic deformation (deviation from reference curvature).

### LocalStress

Computes the local elastic stress (restoring torque) from material properties and orientation.

- **Inputs**: `stiffness` (twist G, bend Iy·E, Iz·E, length), `quat` (orientation from QuatDiff), `omega0` (reference curvature)
- **Output**: `stress[3]` — elastic torque in local coordinates
- **Steps**: (1) Convert `quat` to curvature `omega` via `quat2Vel`; (2) Compute `stress = -stiffness[i]·(omega[i] - omega0[i]) / length` for each axis; (3) Optionally pull-back into the other body frame
- **Purpose**: Produces the elastic restoring torque that resists deviation from the reference curvature.

### mj_applyFT vs Direct Addition to qfrc_passive

| Approach | When to use | What it does |
|----------|-------------|--------------|
| **Direct addition** | Forces/torques already in joint space | Add `qfrc[dof_addr] += torque` for each joint DOF. Requires knowing the mapping from bodies to DOFs. |
| **mj_applyFT** | Forces/torques in Cartesian space (F, τ at a 3D point) | Computes `qfrc += J'·[F; τ]` where J is the Jacobian from joint velocities to linear/angular velocity at the point. Maps a Cartesian wrench at a body point to generalized forces. |

**Why mj_applyFT**: Elasticity models (e.g. Cable) produce forces and torques in Cartesian or body-local space. To affect the simulation, these must be converted to joint-space generalized forces. `mj_applyFT` performs this conversion via the transpose Jacobian. Direct addition would only apply if the model already produced `qfrc` values.