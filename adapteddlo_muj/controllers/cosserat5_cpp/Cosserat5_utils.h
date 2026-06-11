#ifndef COSSERAT5_UTILS_H
#define COSSERAT5_UTILS_H

#include "Eigen/Core"

class Cosserat5Utils
{
public:
    static const Eigen::Vector3d rotateVector3(
        const Eigen::Vector3d& v, const Eigen::Vector3d& u, const double a);
    static const double calculateAngleBetween(
        const Eigen::Vector3d& v1, const Eigen::Vector3d& v2);
    static const Eigen::Matrix3d createSkewSym(const Eigen::Vector3d& v);
    static const Eigen::Vector4d inverseQuat(const Eigen::Vector4d& quat);
    static const Eigen::Vector3d rotVecQuat(
        const Eigen::Vector3d& vec, const Eigen::Vector4d& quat);
    static Eigen::Vector3d rotvecFromRotationMatrix(const Eigen::Matrix3d& R);
};

#endif
