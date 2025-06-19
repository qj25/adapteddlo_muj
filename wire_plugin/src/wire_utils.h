#ifndef MUJOCO_SRC_PLUGIN_ELASTICITY_WIRE_UTILS_H_
#define MUJOCO_SRC_PLUGIN_ELASTICITY_WIRE_UTILS_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace mujoco::plugin::elasticity {

class WireUtils {
 public:
  // Rotate a vector around an axis
  static const Eigen::Vector3d rotateVector3(
      const Eigen::Vector3d &v,
      const Eigen::Vector3d &u,
      const double a);

  // Calculate angle between two vectors
  static const double calculateAngleBetween(
      const Eigen::Vector3d &v1,
      const Eigen::Vector3d &v2);

  // Calculate angle between two vectors with anchor
  static const double calculateAngleBetween2(
      const Eigen::Vector3d &v1,
      const Eigen::Vector3d &v2,
      const Eigen::Vector3d &v_anchor);

  // Calculate angle between two vectors with anchor (another method)
  static const double calculateAngleBetween2b(
    const Eigen::Vector3d &v1,
    const Eigen::Vector3d &v2,
    const Eigen::Vector3d &va);

  // Create skew-symmetric matrix from vector
  static const Eigen::Matrix3d createSkewSym(
      const Eigen::Vector3d &v);

  // Calculate inverse of quaternion
  static const Eigen::Vector4d inverseQuat(
      const Eigen::Vector4d &quat);

  // Rotate vector by quaternion
  static const Eigen::Vector3d rotVecQuat(
      const Eigen::Vector3d &vec,
      const Eigen::Vector4d &quat);
};

}  // namespace mujoco::plugin::elasticity

#endif  // MUJOCO_SRC_PLUGIN_ELASTICITY_WIRE_UTILS_H_ 