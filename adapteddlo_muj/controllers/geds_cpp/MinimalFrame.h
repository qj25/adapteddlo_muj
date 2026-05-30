#ifndef MINIMAL_FRAME_H
#define MINIMAL_FRAME_H

#include <vector>

#include "Eigen/Core"

namespace geds {

struct Frame {
    Eigen::Vector3d tangent;
    Eigen::Vector3d normal;
    Eigen::Vector3d binormal;
};

// Propagate a rotation-minimizing frame along discrete positions (Sloan/Bishop style).
class MinimalFrame {
public:
    static void propagate(
        const std::vector<Eigen::Vector3d>& positions,
        std::vector<Frame>& frames);

    // Material twist angle (rad) about the propagated tangent axis.
    static double rollFromQuat(
        const Eigen::Vector4d& quat_wxyz,
        const Frame& frame);

    static double twistAboutTangent(
        const Eigen::Vector4d& quat_wxyz,
        const Eigen::Vector3d& tangent);

    static Eigen::Vector3d rotateVectorByQuat(
        const Eigen::Vector4d& quat_wxyz,
        const Eigen::Vector3d& v);
};

}  // namespace geds

#endif
