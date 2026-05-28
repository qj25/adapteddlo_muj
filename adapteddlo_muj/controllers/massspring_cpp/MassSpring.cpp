#include "MassSpring.h"

#include <algorithm>
#include <cmath>

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

void MassSpring::computeTorque(int dim_cq, double* current_quat, int dim_nt, double* node_torque) {
    const int n_cur = dim_cq / 4;
    const int n_torq = dim_nt / 3;
    const int n = std::min(n_nodes_, std::min(n_cur, n_torq));

    for (int i = 0; i < n_torq * 3; ++i) {
        node_torque[i] = 0.0;
    }

    for (int i = 0; i < n; ++i) {
        Eigen::Vector4d q_cur;
        q_cur << current_quat[i * 4 + 0],
            current_quat[i * 4 + 1],
            current_quat[i * 4 + 2],
            current_quat[i * 4 + 3];
        q_cur = normalizeQuat(q_cur);

        const Eigen::Vector4d q_rel = multiplyQuat(invertQuat(neutral_quat_[i]), q_cur);
        const Eigen::Vector3d dev = quatToRotvec(q_rel);

        node_torque[i * 3 + 0] = -k_bend_x_ * dev(0);
        node_torque[i * 3 + 1] = -k_bend_y_ * dev(1);
        node_torque[i * 3 + 2] = -k_twist_ * dev(2);
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
