#include "Cosserat3_utils.h"

#include <cmath>
#include "Eigen/Dense"
#include "Eigen/Geometry"

const Eigen::Vector3d Cosserat3Utils::rotateVector3(
    const Eigen::Vector3d& v, const Eigen::Vector3d& u, const double a)
{
    Eigen::Matrix3d R;
    Eigen::Vector3d v_res;
    const Eigen::Vector3d u_norm = u / u.norm();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == j) {
                R(i, j) = cos(a) + (pow(u_norm(i), 2.) * (1. - cos(a)));
            } else {
                double ss = 1.;
                if (i < j) { ss *= -1.; }
                if (((i + 1) * (j + 1)) % 2 != 0) { ss *= -1.; }
                R(i, j) = u_norm(i) * u_norm(j) * (1. - cos(a)) + ss * u(3 - (i + j)) * sin(a);
            }
        }
    }
    v_res = R * v;
    return v_res;
}

const double Cosserat3Utils::calculateAngleBetween(
    const Eigen::Vector3d& v1, const Eigen::Vector3d& v2)
{
    double cos_ab = v1.dot(v2) / (v1.norm() * v2.norm());
    if (cos_ab > 1) { return 0; }
    return acos(cos_ab);
}

const Eigen::Matrix3d Cosserat3Utils::createSkewSym(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d skew_sym_v;
    skew_sym_v << 0., -v[2], v[1],
        v[2], 0., -v[0],
        -v[1], v[0], 0.;
    return skew_sym_v;
}

const Eigen::Vector4d Cosserat3Utils::inverseQuat(const Eigen::Vector4d& quat)
{
    Eigen::Vector4d quat_inv = -quat;
    quat_inv(0) = -quat_inv(0);
    quat_inv = quat_inv / quat_inv.dot(quat_inv);
    return quat_inv;
}

const Eigen::Vector3d Cosserat3Utils::rotVecQuat(
    const Eigen::Vector3d& vec, const Eigen::Vector4d& quat)
{
    Eigen::Vector3d res;
    if (vec[0] == 0 && vec[1] == 0 && vec[2] == 0) {
        res << 0., 0., 0.;
    } else if (quat[0] == 1 && quat[1] == 0 && quat[2] == 0 && quat[3] == 0) {
        res = vec;
    } else {
        Eigen::Vector3d tmp;
        tmp << quat[0] * vec[0] + quat[2] * vec[2] - quat[3] * vec[1],
            quat[0] * vec[1] + quat[3] * vec[0] - quat[1] * vec[2],
            quat[0] * vec[2] + quat[1] * vec[1] - quat[2] * vec[0];
        res[0] = vec[0] + 2 * (quat[2] * tmp[2] - quat[3] * tmp[1]);
        res[1] = vec[1] + 2 * (quat[3] * tmp[0] - quat[1] * tmp[2]);
        res[2] = vec[2] + 2 * (quat[1] * tmp[1] - quat[2] * tmp[0]);
    }
    return res;
}

Eigen::Vector3d Cosserat3Utils::rotvecFromRotationMatrix(const Eigen::Matrix3d& R)
{
    Eigen::AngleAxisd aa(R);
    return aa.angle() * aa.axis();
}
