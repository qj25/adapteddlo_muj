#include "CatmullRom.h"

#include "Eigen/Geometry"

namespace geds {

Eigen::Vector3d CatmullRom::position(
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2,
    const Eigen::Vector3d& p3,
    double t) {
    const double t2 = t * t;
    const double t3 = t2 * t;
    return 0.5 * ((2.0 * p1) + (-p0 + p2) * t +
                  (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
                  (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
}

Eigen::Vector3d CatmullRom::derivative1(
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2,
    const Eigen::Vector3d& p3,
    double t) {
    const double t2 = t * t;
    return 0.5 * ((-p0 + p2) + 2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t +
                  3.0 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t2);
}

Eigen::Vector3d CatmullRom::derivative2(
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2,
    const Eigen::Vector3d& p3,
    double t) {
    return 0.5 * (2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) +
                  6.0 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t);
}

Eigen::Vector3d CatmullRom::derivative3(
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2,
    const Eigen::Vector3d& p3,
    double t) {
    (void)t;
    return 3.0 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3);
}

double CatmullRom::scalarPosition(double s0, double s1, double s2, double s3, double t) {
    const double t2 = t * t;
    const double t3 = t2 * t;
    return 0.5 * ((2.0 * s1) + (-s0 + s2) * t +
                  (2.0 * s0 - 5.0 * s1 + 4.0 * s2 - s3) * t2 +
                  (-s0 + 3.0 * s1 - 3.0 * s2 + s3) * t3);
}

double CatmullRom::scalarDerivative1(double s0, double s1, double s2, double s3, double t) {
    const double t2 = t * t;
    return 0.5 * ((-s0 + s2) + 2.0 * (2.0 * s0 - 5.0 * s1 + 4.0 * s2 - s3) * t +
                  3.0 * (-s0 + 3.0 * s1 - 3.0 * s2 + s3) * t2);
}

}  // namespace geds
