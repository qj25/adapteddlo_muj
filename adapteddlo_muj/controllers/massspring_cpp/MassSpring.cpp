#include "MassSpring.h"

#include <algorithm>
#include <cmath>

namespace {
Eigen::Vector3d cross3(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return Eigen::Vector3d(
        a(1) * b(2) - a(2) * b(1),
        a(2) * b(0) - a(0) * b(2),
        a(0) * b(1) - a(1) * b(0)
    );
}
}  // namespace

MassSpring::MassSpring(
    int dim_nq,
    double* neutral_quat,
    double k_bend_x,
    double k_bend_y,
    double k_twist
) : n_nodes_(dim_nq / 4),
    neutral_quat_(new Eigen::Vector4d[n_nodes_]),
    k_bend_x_(k_bend_x),
    k_bend_y_(k_bend_y),
    k_twist_(k_twist) {
    setNeutralQuat(dim_nq, neutral_quat);
}

MassSpring::~MassSpring() {
    delete[] neutral_quat_;
}

void MassSpring::setNeutralQuat(int dim_nq, double* neutral_quat) {
    const int n_new = dim_nq / 4;
    if (n_new != n_nodes_) {
        delete[] neutral_quat_;
        n_nodes_ = n_new;
        neutral_quat_ = new Eigen::Vector4d[n_nodes_];
    }
    for (int i = 0; i < n_nodes_; ++i) {
        Eigen::Vector4d q;
        q << neutral_quat[i * 4 + 0],
            neutral_quat[i * 4 + 1],
            neutral_quat[i * 4 + 2],
            neutral_quat[i * 4 + 3];
        neutral_quat_[i] = normalizeQuat(q);
    }
}

void MassSpring::setStiffness(double k_bend_x, double k_bend_y, double k_twist) {
    k_bend_x_ = k_bend_x;
    k_bend_y_ = k_bend_y;
    k_twist_ = k_twist;
}

void MassSpring::computeTorque(
    int dim_x,
    double* current_x,
    int dim_cq,
    double* current_quat,
    int dim_nt,
    double* node_torque
) {
    const int n_pos = dim_x / 3;
    const int n_cur = dim_cq / 4;
    const int n_torq = dim_nt / 3;
    const int n = std::min(n_nodes_, std::min(n_cur, std::min(n_torq, n_pos)));

    for (int i = 0; i < n_torq * 3; ++i) {
        node_torque[i] = 0.0;
    }

    auto readQuat = [&](int idx, double* src) {
        Eigen::Vector4d q;
        q << src[idx * 4 + 0],
            src[idx * 4 + 1],
            src[idx * 4 + 2],
            src[idx * 4 + 3];
        return normalizeQuat(q);
    };

    auto readPos = [&](int idx, double* src) {
        return Eigen::Vector3d(
            src[idx * 3 + 0],
            src[idx * 3 + 1],
            src[idx * 3 + 2]
        );
    };

    const Eigen::Vector3d fallback_tangent(1.0, 0.0, 0.0);
    const double k_bend = 0.5 * (k_bend_x_ + k_bend_y_);

    // Joint spring: deviation of relative rotation from neutral relative rotation.
    // Bend/twist split uses the segment tangent; torque is on the child in child frame.
    for (int i = 1; i < n; ++i) {
        const Eigen::Vector4d q_prev = readQuat(i - 1, current_quat);
        const Eigen::Vector4d q_cur = readQuat(i, current_quat);
        const Eigen::Vector4d q_rel = multiplyQuat(invertQuat(q_prev), q_cur);
        const Eigen::Vector4d q_rel0 = multiplyQuat(invertQuat(neutral_quat_[i - 1]), neutral_quat_[i]);
        const Eigen::Vector4d q_dev = multiplyQuat(invertQuat(q_rel0), q_rel);
        const Eigen::Vector3d dev = quatToRotvec(q_dev);

        const Eigen::Vector3d tangent = safeNormalize(
            readPos(i, current_x) - readPos(i - 1, current_x),
            fallback_tangent
        );
        const double twist_scalar = dev.dot(tangent);
        const Eigen::Vector3d twist_vec = twist_scalar * tangent;
        const Eigen::Vector3d bend_vec = dev - twist_vec;
        const Eigen::Vector3d weighted = k_bend * bend_vec + k_twist_ * twist_vec;
        const Eigen::Vector3d tau_parent = -weighted;

        const Eigen::Vector3d tau_child = rotVecQuat(tau_parent, invertQuat(q_rel));
        node_torque[i * 3 + 0] = tau_child(0);
        node_torque[i * 3 + 1] = tau_child(1);
        node_torque[i * 3 + 2] = tau_child(2);
    }
}

Eigen::Vector4d MassSpring::normalizeQuat(const Eigen::Vector4d& q) const {
    double norm = q.norm();
    if (norm < 1e-12) {
        return Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
    }
    Eigen::Vector4d qn = q / norm;
    if (qn(0) < 0.0) {
        qn = -qn;
    }
    return qn;
}

Eigen::Vector4d MassSpring::invertQuat(const Eigen::Vector4d& q) const {
    Eigen::Vector4d qi;
    qi << q(0), -q(1), -q(2), -q(3);
    return qi;
}

Eigen::Vector4d MassSpring::multiplyQuat(const Eigen::Vector4d& qa, const Eigen::Vector4d& qb) const {
    Eigen::Vector4d out;
    const double wa = qa(0);
    const double xa = qa(1);
    const double ya = qa(2);
    const double za = qa(3);
    const double wb = qb(0);
    const double xb = qb(1);
    const double yb = qb(2);
    const double zb = qb(3);

    out(0) = wa * wb - xa * xb - ya * yb - za * zb;
    out(1) = wa * xb + xa * wb + ya * zb - za * yb;
    out(2) = wa * yb - xa * zb + ya * wb + za * xb;
    out(3) = wa * zb + xa * yb - ya * xb + za * wb;
    return normalizeQuat(out);
}

Eigen::Vector3d MassSpring::quatToRotvec(const Eigen::Vector4d& q) const {
    const Eigen::Vector4d qn = normalizeQuat(q);
    const double w = std::max(-1.0, std::min(1.0, qn(0)));
    const Eigen::Vector3d v = qn.segment<3>(1);
    const double v_norm = v.norm();

    if (v_norm < 1e-12) {
        return 2.0 * v;
    }

    const double angle = 2.0 * std::atan2(v_norm, w);
    return (angle / v_norm) * v;
}

Eigen::Vector3d MassSpring::safeNormalize(
    const Eigen::Vector3d& v,
    const Eigen::Vector3d& fallback
) const {
    const double n = v.norm();
    if (n < 1e-12) {
        return fallback;
    }
    return v / n;
}

Eigen::Vector3d MassSpring::rotVecQuat(const Eigen::Vector3d& vec, const Eigen::Vector4d& quat) const {
    if (vec.squaredNorm() < 1e-24) {
        return Eigen::Vector3d::Zero();
    }
    if (std::abs(quat(0) - 1.0) < 1e-12 && quat.segment<3>(1).squaredNorm() < 1e-24) {
        return vec;
    }

    const Eigen::Vector3d q_xyz = quat.segment<3>(1);
    const Eigen::Vector3d tmp(
        quat(0) * vec(0) + quat(2) * vec(2) - quat(3) * vec(1),
        quat(0) * vec(1) + quat(3) * vec(0) - quat(1) * vec(2),
        quat(0) * vec(2) + quat(1) * vec(1) - quat(2) * vec(0)
    );
    return vec + 2.0 * cross3(q_xyz, tmp);
}
