#include "wire_utils.h"
#include <iostream>

namespace mujoco::plugin::elasticity {

const Eigen::Vector3d WireUtils::rotateVector3(
    const Eigen::Vector3d &v,
    const Eigen::Vector3d &u,
    const double a) {
  Eigen::Matrix3d R;
  Eigen::Vector3d v_res;
  Eigen::Vector3d u_norm = u / u.norm();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j) {
        R(i,j) = (
          cos(a)
          + (pow(u_norm(i),2.) * (1. - cos(a)))
        );
      }
      else {
        double ss = 1.;
        if (i < j) {ss *= -1.;}
        if (((i+1) * (j+1)) % 2 != 0) {ss *= -1.;}
        R(i,j) = (
          u_norm(i) * u_norm(j) * (1. - cos(a))
          + ss * u(3-(i+j)) * sin(a)
        );
      }
    }
  }
  v_res = R*v;
  return v_res;
}

const double WireUtils::calculateAngleBetween(
    const Eigen::Vector3d &v1,
    const Eigen::Vector3d &v2) {
  double cos_ab, ab;
  cos_ab = v1.dot(v2) / (v1.norm() * v2.norm());
  if (cos_ab > 1) {return 0;}
  ab = acos(cos_ab);
  if (std::isnan(ab)) {
    std::string str_error = "Error in WireUtils::calculateAngleBetween";
    std::cout << str_error << std::endl;
    std::cout << v1 << std::endl;
    std::cout << v2 << std::endl;
  }
  return ab;
}

const double WireUtils::calculateAngleBetween2(
    const Eigen::Vector3d &v1,
    const Eigen::Vector3d &v2,
    const Eigen::Vector3d &v_anchor) {
  double e_tol, sab, n_div, cab, ab;
  e_tol = 1e-3;
  sab = 0.;
  n_div = 0.;

  Eigen::Vector3d cross12;
  if ((v1-v2).norm() < e_tol) {return 0;}
  cross12 = v1.cross(v2);
  for (int i = 0; i < 3; i++) {
    if (std::abs(v_anchor(i)) < e_tol) {continue;}
    sab += (
      cross12(i)
      / (v1.norm() * v2.norm())
      / v_anchor(i)
    );
    n_div += 1.;
  }
  sab /= n_div;
  cab = (
    v1.dot(v2)
    / (v1.norm() * v2.norm())
  );
  ab = atan2(sab, cab);
  return ab;
}

const double WireUtils::calculateAngleBetween2b(
  const Eigen::Vector3d &v1,
  const Eigen::Vector3d &v2,
  const Eigen::Vector3d &va) {
double dot_norm_val = v1.dot(v2) / (v1.norm() * v2.norm());
if (dot_norm_val > 1.0) dot_norm_val = 1.0;
double theta_diff = std::acos(dot_norm_val);
if ((v1.cross(v2)).dot(va) < 0) {
  theta_diff *= -1.0;
}
return theta_diff;
}

const Eigen::Matrix3d WireUtils::createSkewSym(
    const Eigen::Vector3d &v) {
  Eigen::Matrix3d skew_sym_v;
  skew_sym_v << 0., -v[2], v[1],
      v[2], 0., -v[0],
      -v[1], v[0], 0.;
  return skew_sym_v;
}

const Eigen::Vector4d WireUtils::inverseQuat(
    const Eigen::Vector4d &quat) {
  Eigen::Vector4d quat_inv;
  quat_inv = - quat;
  quat_inv(0) = - quat_inv(0);
  quat_inv = quat_inv / quat_inv.dot(quat_inv);
  return quat_inv;
}

const Eigen::Vector3d WireUtils::rotVecQuat(
    const Eigen::Vector3d &vec,
    const Eigen::Vector4d &quat) {
  Eigen::Vector3d res;
  Eigen::Vector3d tmp;
  // zero vec: zero res
  if (vec[0] == 0 && vec[1] == 0 && vec[2] == 0) {
    res << 0., 0., 0.;
  }
  // null quat: copy vec
  else if (quat[0] == 1 && quat[1] == 0 && quat[2] == 0 && quat[3] == 0) {
    res = vec;
  }
  // regular processing
  else {
    // tmp = q_w * v + cross(q_xyz, v)
    tmp <<  quat[0]*vec[0] + quat[2]*vec[2] - quat[3]*vec[1],
        quat[0]*vec[1] + quat[3]*vec[0] - quat[1]*vec[2],
        quat[0]*vec[2] + quat[1]*vec[1] - quat[2]*vec[0];

    // res = v + 2 * cross(q_xyz, t)
    res[0] = vec[0] + 2 * (quat[2]*tmp[2] - quat[3]*tmp[1]);
    res[1] = vec[1] + 2 * (quat[3]*tmp[0] - quat[1]*tmp[2]);
    res[2] = vec[2] + 2 * (quat[1]*tmp[1] - quat[2]*tmp[0]);
  }
  return res;
}

}  // namespace mujoco::plugin::elasticity 