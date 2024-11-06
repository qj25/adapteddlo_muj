#ifndef DLOUTILS_H
#define DLOUTILS_H

#include <vector>
#include "Eigen/Core"

class DloUtils
{
public:
    static const Eigen::Vector3d rotateVector3(const Eigen::Vector3d &v, const Eigen::Vector3d &u, const double a);
    /* 
    for rotation of 3d vector 
        - v: vector
        - u: anchor vector (rotate about this vector)
        - a: angle in radians
    */
    static const double calculateAngleBetween(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2);
    // vectors point away from the point where angle is taken
    static const double calculateAngleBetween2(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2, const Eigen::Vector3d &v_anchor);
    // use sin and cos to find angle diff from -np.pi to np.pi
    // rotation angle of v1 to v2 wrt to axis v_anchor
    static const Eigen::Matrix3d createSkewSym(const Eigen::Vector3d &v);
    static const Eigen::Vector4d inverseQuat(const Eigen::Vector4d &quat);
    static const Eigen::Vector3d rotVecQuat(const Eigen::Vector3d &vec, const Eigen::Vector4d &quat);
};

#endif
