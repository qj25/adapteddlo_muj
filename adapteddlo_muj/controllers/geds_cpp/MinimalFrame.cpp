#include "MinimalFrame.h"

#include <cmath>

#include "Eigen/Geometry"

namespace geds {

namespace {

Eigen::Vector3d safeNormalize(const Eigen::Vector3d& v, const Eigen::Vector3d& fallback) {
    const double n = v.norm();
    if (n < 1e-12) {
        return fallback;
    }
    return v / n;
}

}  // namespace

void MinimalFrame::propagate(
    const std::vector<Eigen::Vector3d>& positions,
    std::vector<Frame>& frames) {
    const int n = static_cast<int>(positions.size());
    frames.resize(static_cast<size_t>(n));

    if (n == 0) {
        return;
    }
    if (n == 1) {
        frames[0].tangent = Eigen::Vector3d(1.0, 0.0, 0.0);
        frames[0].normal = Eigen::Vector3d(0.0, 0.0, 1.0);
        frames[0].binormal = Eigen::Vector3d(0.0, 1.0, 0.0);
        return;
    }

    Eigen::Vector3d t0 = safeNormalize(
        positions[1] - positions[0], Eigen::Vector3d(1.0, 0.0, 0.0));
    Eigen::Vector3d n0 = Eigen::Vector3d(0.0, 0.0, 1.0);
    if (std::abs(t0.dot(n0)) > 0.95) {
        n0 = Eigen::Vector3d(0.0, 1.0, 0.0);
    }
    n0 = safeNormalize(n0 - t0 * t0.dot(n0), Eigen::Vector3d(0.0, 0.0, 1.0));
    Eigen::Vector3d b0 = t0.cross(n0);

    frames[0].tangent = t0;
    frames[0].normal = n0;
    frames[0].binormal = b0;

    for (int i = 1; i < n; ++i) {
        const Eigen::Vector3d ti = safeNormalize(
            positions[i] - positions[i - 1], frames[i - 1].tangent);
        const Eigen::Vector3d bi = ti.cross(frames[i - 1].normal);
        const Eigen::Vector3d ni = bi.cross(ti);
        frames[i].tangent = ti;
        frames[i].normal = safeNormalize(ni, frames[i - 1].normal);
        frames[i].binormal = frames[i].tangent.cross(frames[i].normal);
    }
}

Eigen::Vector3d MinimalFrame::rotateVectorByQuat(
    const Eigen::Vector4d& quat_wxyz,
    const Eigen::Vector3d& v) {
    const double w = quat_wxyz(0);
    const Eigen::Vector3d qv(quat_wxyz(1), quat_wxyz(2), quat_wxyz(3));
    return v + 2.0 * qv.cross(qv.cross(v) + w * v);
}

double MinimalFrame::twistAboutTangent(
    const Eigen::Vector4d& quat_wxyz,
    const Eigen::Vector3d& tangent) {
    const Eigen::Vector3d t = safeNormalize(tangent, Eigen::Vector3d(1.0, 0.0, 0.0));
    double w = quat_wxyz(0);
    Eigen::Vector3d v(quat_wxyz(1), quat_wxyz(2), quat_wxyz(3));
    if (w < 0.0) {
        w = -w;
        v = -v;
    }
    const double v_n = v.norm();
    if (v_n < 1e-12) {
        return 0.0;
    }
    const Eigen::Vector3d axis = v / v_n;
    const double angle = 2.0 * std::atan2(v_n, w);
    return angle * axis.dot(t);
}

double MinimalFrame::rollFromQuat(
    const Eigen::Vector4d& quat_wxyz,
    const Frame& frame) {
    return twistAboutTangent(quat_wxyz, frame.tangent);
}

}  // namespace geds
