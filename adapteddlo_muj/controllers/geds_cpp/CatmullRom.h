#ifndef CATMULL_ROM_H
#define CATMULL_ROM_H

#include "Eigen/Core"

namespace geds {

// Uniform cubic Catmull-Rom segment between P1 and P2 using neighbors P0, P3.
// Parameter t in [0, 1].
struct CatmullRom {
    static Eigen::Vector3d position(
        const Eigen::Vector3d& p0,
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& p2,
        const Eigen::Vector3d& p3,
        double t);

    static Eigen::Vector3d derivative1(
        const Eigen::Vector3d& p0,
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& p2,
        const Eigen::Vector3d& p3,
        double t);

    static Eigen::Vector3d derivative2(
        const Eigen::Vector3d& p0,
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& p2,
        const Eigen::Vector3d& p3,
        double t);

    static Eigen::Vector3d derivative3(
        const Eigen::Vector3d& p0,
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& p2,
        const Eigen::Vector3d& p3,
        double t);

    static double scalarPosition(double s0, double s1, double s2, double s3, double t);
    static double scalarDerivative1(double s0, double s1, double s2, double s3, double t);
};

}  // namespace geds

#endif
