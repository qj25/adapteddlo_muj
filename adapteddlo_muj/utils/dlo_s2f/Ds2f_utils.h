#ifndef DS2FUTILS_H
#define DS2FUTILS_H

#include <vector>
#include "Eigen/Core"
#include "Ds2f_obj.h"

class Ds2fUtils
{
public:
    static const Eigen::VectorXd solveUpperTriangular(const Eigen::MatrixXd& A, const Eigen::VectorXd& B);
    static const Eigen::VectorXd solveUTBig(const Eigen::MatrixXd& A, const Eigen::VectorXd& B);
    static const Eigen::VectorXd solveUTXBig(const Eigen::MatrixXd& A, const Eigen::VectorXd& B);

    bool checkLinearCorrelation(const Eigen::Vector3d& vec1, const Eigen::Vector3d& vec2);

    static bool remove_from_back(Section& section);
    static bool remove_from_front(Section& section);

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
    static const Eigen::Vector3d getVecfromSkewSym(const Eigen::Matrix3d  &v_ss);

    static const Eigen::Vector4d inverseQuat(const Eigen::Vector4d &quat);
    static const Eigen::Vector3d rotVecQuat(const Eigen::Vector3d &vec, const Eigen::Vector4d &quat);

    static std::pair<Eigen::Vector3d, double> computeLeastSquares(
        const Eigen::MatrixXd& A,
        const Eigen::VectorXd& c
    );
};

#endif