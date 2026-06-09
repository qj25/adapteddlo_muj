#include "RodCosserat.h"

#include <Eigen/Geometry>
#include <cmath>
#include <cstring>
#include <vector>

using Quaterniond = Eigen::Quaterniond;
using Vector3d = Eigen::Vector3d;

static Quaterniond quatFromArray(const double* q) {
    return Quaterniond(q[0], q[1], q[2], q[3]);
}

static void quatToArray(const Quaterniond& q, double* out) {
    out[0] = q.w();
    out[1] = q.x();
    out[2] = q.y();
    out[3] = q.z();
}

static Vector3d vec3FromArray(const double* v) {
    return Vector3d(v[0], v[1], v[2]);
}

static Vector3d quatToRotvec(const Quaterniond& dq_in) {
    Quaterniond dq = dq_in;
    if (dq.w() < 0.0) {
        dq.coeffs() = -dq.coeffs();
    }
    const double w = dq.w();
    const Vector3d v(dq.x(), dq.y(), dq.z());
    const double vn = v.norm();
    if (vn < 1e-12) {
        return 2.0 * v;
    }
    const double angle = 2.0 * std::atan2(vn, w);
    return (angle / vn) * v;
}

static Quaterniond rotvecToQuat(const Vector3d& rv) {
    const double angle = rv.norm();
    if (angle < 1e-12) {
        return Quaterniond::Identity();
    }
    return Quaterniond(Eigen::AngleAxisd(angle, rv / angle));
}

static Vector3d safeNormalize(const Vector3d& v, const Vector3d& fallback) {
    const double n = v.norm();
    if (n < 1e-12) {
        return fallback;
    }
    return v / n;
}

RodCosserat::RodCosserat(
    int n_nodes,
    double segment_length,
    double k_bend,
    double k_twist)
    : n_nodes_(n_nodes),
      n_joints_(n_nodes > 1 ? n_nodes - 1 : 0),
      segment_length_(segment_length),
      k_bend_(k_bend),
      k_twist_(k_twist),
      k_stretch_(10.0 * k_bend),
      k_torque_(0.01),
      num_iters_(4),
      use_velocity_drive_(false),
      joint_child_torques_(true),
      rest_rel_quat_(nullptr) {
    if (n_joints_ > 0) {
        rest_rel_quat_ = new double[4 * n_joints_];
        std::memset(rest_rel_quat_, 0, sizeof(double) * 4 * n_joints_);
        for (int j = 0; j < n_joints_; ++j) {
            rest_rel_quat_[4 * j + 0] = 1.0;
        }
    }
}

RodCosserat::~RodCosserat() {
    delete[] rest_rel_quat_;
}

void RodCosserat::setMaterial(double k_bend, double k_twist) {
    k_bend_ = k_bend;
    k_twist_ = k_twist;
}

void RodCosserat::setStretchStiffness(double k_stretch) {
    k_stretch_ = k_stretch;
}

void RodCosserat::setNumIterations(int num_iters) {
    num_iters_ = num_iters < 1 ? 1 : num_iters;
}

void RodCosserat::setTorqueGain(double k_torque) {
    k_torque_ = k_torque;
}

void RodCosserat::setVelocityDrive(bool use_velocity_drive) {
    use_velocity_drive_ = use_velocity_drive;
}

void RodCosserat::setJointChildTorques(bool joint_child_torques) {
    joint_child_torques_ = joint_child_torques;
}

void RodCosserat::buildRestAngles(int dim_q, const double* rest_quat) {
    (void)dim_q;
    if (n_joints_ < 1) {
        return;
    }
    for (int j = 0; j < n_joints_; ++j) {
        const Quaterniond q0 = quatFromArray(rest_quat + 4 * j);
        const Quaterniond q1 = quatFromArray(rest_quat + 4 * (j + 1));
        const Quaterniond q_rest = q0.inverse() * q1;
        quatToArray(q_rest, rest_rel_quat_ + 4 * j);
    }
}

void RodCosserat::reinitRest(
    int dim_x,
    const double* rest_x,
    int dim_q,
    const double* rest_quat) {
    (void)dim_x;
    (void)rest_x;
    buildRestAngles(dim_q, rest_quat);
}

static void splitBendTwist(
    const Vector3d& rotvec,
    const Vector3d& tangent,
    Vector3d& bend_vec,
    Vector3d& twist_vec) {
    const double twist_scalar = rotvec.dot(tangent);
    twist_vec = twist_scalar * tangent;
    bend_vec = rotvec - twist_vec;
}

// Bend/twist-only shadow orientation solve (stretch skipped; kss=0 in reference).
// Uses unit constraint weights here; physical k_bend/k_twist are applied only when
// converting the shadow correction to output torques (XPBD-style decoupling).
static void shadowOrientationSolve(
    std::vector<Quaterniond>& q_shadow,
    const double* rest_rel_quat,
    int n_joints) {
    // q0^{-1}*q1 rotvecs are in the parent joint frame; split with local +X.
    const Vector3d tangent_parent(1.0, 0.0, 0.0);
    const double step_gain = 0.25;

    for (int j = 0; j < n_joints; ++j) {
        const Quaterniond q0 = q_shadow[static_cast<size_t>(j)];
        const Quaterniond q1 = q_shadow[static_cast<size_t>(j + 1)];
        const Quaterniond q_rest = quatFromArray(rest_rel_quat + 4 * j);

        Quaterniond qq = q0.inverse() * q1;
        Quaterniond delta = qq * q_rest.inverse();
        if (delta.w() < 0.0) {
            delta.coeffs() = -delta.coeffs();
        }

        Vector3d bend_vec;
        Vector3d twist_vec;
        splitBendTwist(
            quatToRotvec(delta),
            tangent_parent,
            bend_vec,
            twist_vec);
        const Vector3d corr = step_gain * (bend_vec + twist_vec);

        const Quaterniond dq0 = rotvecToQuat(corr);
        const Quaterniond dq1 = rotvecToQuat(-corr);
        q_shadow[static_cast<size_t>(j)] = (dq0 * q_shadow[static_cast<size_t>(j)]).normalized();
        q_shadow[static_cast<size_t>(j + 1)] =
            (q_shadow[static_cast<size_t>(j + 1)] * dq1).normalized();
    }
}

static Vector3d nodeTangent(
    int i,
    int n,
    const std::vector<Vector3d>& x_anchor,
    const Vector3d& fallback) {
    if (n < 2) {
        return fallback;
    }
    if (i <= 0) {
        return safeNormalize(x_anchor[1] - x_anchor[0], fallback);
    }
    if (i >= n - 1) {
        return safeNormalize(x_anchor[n - 1] - x_anchor[n - 2], fallback);
    }
    return safeNormalize(
        x_anchor[static_cast<size_t>(i + 1)] - x_anchor[static_cast<size_t>(i - 1)],
        fallback);
}

void RodCosserat::computeWrenches(
    int dim_x,
    const double* x,
    int dim_q,
    const double* quat,
    double dt,
    int dim_f,
    double* force_out,
    int dim_t,
    double* torque_out) {
    (void)dim_x;
    (void)dim_q;
    (void)dim_f;
    (void)dim_t;

    const int n = n_nodes_;
    std::memset(force_out, 0, sizeof(double) * 3 * n);
    std::memset(torque_out, 0, sizeof(double) * 3 * n);
    if (n < 2 || n_joints_ < 1 || dt <= 0.0) {
        return;
    }

    std::vector<Quaterniond> q0(static_cast<size_t>(n));
    std::vector<Quaterniond> q_shadow(static_cast<size_t>(n));
    std::vector<Vector3d> x_anchor(static_cast<size_t>(n));

    for (int i = 0; i < n; ++i) {
        q0[static_cast<size_t>(i)] = quatFromArray(quat + 4 * i).normalized();
        q_shadow[static_cast<size_t>(i)] = q0[static_cast<size_t>(i)];
        x_anchor[static_cast<size_t>(i)] = vec3FromArray(x + 3 * i);
    }

    for (int iter = 0; iter < num_iters_; ++iter) {
        shadowOrientationSolve(
            q_shadow,
            rest_rel_quat_,
            n_joints_);
    }

    double scale = k_torque_;
    if (use_velocity_drive_) {
        scale *= 1.0 / dt;
    }
    const Vector3d fallback_tangent(1.0, 0.0, 0.0);

    // Pairwise distance springs (world frame, symmetric nodal forces).
    if (k_stretch_ > 0.0) for (int j = 0; j < n_joints_; ++j) {
        const Vector3d edge =
            x_anchor[static_cast<size_t>(j + 1)] - x_anchor[static_cast<size_t>(j)];
        const double dist = edge.norm();
        if (dist < 1e-12) {
            continue;
        }
        const double err = dist - segment_length_;
        const Vector3d f = k_stretch_ * err * (edge / dist);
        force_out[3 * j + 0] += f[0];
        force_out[3 * j + 1] += f[1];
        force_out[3 * j + 2] += f[2];
        force_out[3 * (j + 1) + 0] -= f[0];
        force_out[3 * (j + 1) + 1] -= f[1];
        force_out[3 * (j + 1) + 2] -= f[2];
    }

    const Vector3d tangent_parent(1.0, 0.0, 0.0);

    if (joint_child_torques_) {
        // Shadow-corrected relative rotation per edge; torque on child in child frame.
        for (int i = 1; i < n; ++i) {
            const int j = i - 1;
            const Quaterniond q_prev = q0[static_cast<size_t>(j)];
            const Quaterniond q_cur = q0[static_cast<size_t>(i)];
            const Quaterniond qs_prev = q_shadow[static_cast<size_t>(j)];
            const Quaterniond qs_cur = q_shadow[static_cast<size_t>(i)];

            const Quaterniond q_rel = q_prev.inverse() * q_cur;
            const Quaterniond q_rel_shadow = qs_prev.inverse() * qs_cur;
            Quaterniond delta = q_rel_shadow * q_rel.inverse();
            if (delta.w() < 0.0) {
                delta.coeffs() = -delta.coeffs();
            }

            Vector3d bend_vec;
            Vector3d twist_vec;
            splitBendTwist(
                quatToRotvec(delta),
                tangent_parent,
                bend_vec,
                twist_vec);
            const Vector3d weighted = k_bend_ * bend_vec + k_twist_ * twist_vec;
            const Vector3d tau_child = q_rel.inverse() * weighted;

            torque_out[3 * i + 0] = scale * tau_child[0];
            torque_out[3 * i + 1] = scale * tau_child[1];
            torque_out[3 * i + 2] = scale * tau_child[2];
        }
    } else {
        // Per-node shadow correction; torque in world frame.
        for (int i = 0; i < n; ++i) {
            Quaterniond dq = q_shadow[static_cast<size_t>(i)] * q0[static_cast<size_t>(i)].inverse();
            if (dq.w() < 0.0) {
                dq.coeffs() = -dq.coeffs();
            }
            Vector3d bend_vec;
            Vector3d twist_vec;
            splitBendTwist(
                quatToRotvec(dq),
                nodeTangent(i, n, x_anchor, fallback_tangent),
                bend_vec,
                twist_vec);
            const Vector3d weighted = k_bend_ * bend_vec + k_twist_ * twist_vec;
            torque_out[3 * i + 0] = scale * weighted[0];
            torque_out[3 * i + 1] = scale * weighted[1];
            torque_out[3 * i + 2] = scale * weighted[2];
        }
    }
}
