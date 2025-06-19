#include <algorithm>
#include <cstddef>
#include <sstream>
#include <optional>
#include <iostream>

#include <mujoco/mjplugin.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include "wire_qst.h"
#include "wire_utils.h"

/*
note:
*B* Distance matrix defined:
    final distmat[a].row(b) = distmat[b].row(a) 
    is the distance from a to b (b-a)
    such that b < a,
    JUST USE negative to get a to b!
    this distance.cross(force) will give the torque direction in *A*.
*/

namespace mujoco::plugin::elasticity {
namespace {

// Jet color palette for visualization
void scalar2rgba(float rgba[4], mjtNum stress[3], mjtNum vmin, mjtNum vmax) {
  // L2 norm of the stress
  mjtNum v = mju_norm3(stress);
  v = v < vmin ? vmin : v;
  v = v > vmax ? vmax : v;
  mjtNum dv = vmax - vmin;

  if (v < (vmin + 0.25 * dv)) {
    rgba[0] = 0;
    rgba[1] = 4 * (v - vmin) / dv;
    rgba[2] = 1;
  } else if (v < (vmin + 0.5 * dv)) {
    rgba[0] = 0;
    rgba[1] = 1;
    rgba[2] = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
  } else if (v < (vmin + 0.75 * dv)) {
    rgba[0] = 4 * (v - vmin - 0.5 * dv) / dv;
    rgba[1] = 1;
    rgba[2] = 0;
  } else {
    rgba[0] = 1;
    rgba[1] = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
    rgba[2] = 0;
  }
}

// Helper function to read numeric attributes
bool CheckAttr(const char* name, const mjModel* m, int instance) {
  char *end;
  std::string value = mj_getPluginConfig(m, instance, name);
  value.erase(std::remove_if(value.begin(), value.end(), isspace), value.end());
  strtod(value.c_str(), &end);
  return end == value.data() + value.size();
}

}  // namespace

// Factory function
std::optional<WireQST> WireQST::Create(const mjModel* m, mjData* d, int instance) {
  if (CheckAttr("twist", m, instance) && CheckAttr("bend", m, instance) && CheckAttr("twist_displace", m, instance)) {
    return WireQST(m, d, instance);
  } else {
    mju_warning("Invalid parameter specification in wire_qst plugin");
    return std::nullopt;
  }
}

// Plugin constructor
WireQST::WireQST(const mjModel* m, mjData* d, int instance) {
  // parameters were validated by the factor function
  std::string flat = mj_getPluginConfig(m, instance, "flat");
  mjtNum G = strtod(mj_getPluginConfig(m, instance, "twist"), nullptr);
  mjtNum E = strtod(mj_getPluginConfig(m, instance, "bend"), nullptr);
  vmax = strtod(mj_getPluginConfig(m, instance, "vmax"), nullptr);
  p_thetan = strtod(mj_getPluginConfig(m, instance, "twist_displace"), nullptr);

  // count plugin bodies
  n = 0;
  for (int i = 1; i < m->nbody; i++) {
    if (m->body_plugin[i] == instance) {
      if (!n++) {
        i0 = i;
      }
    }
  }
  n--;

  // run forward kinematics to populate xquat (mjData not yet initialized)
  mju_zero(d->mocap_quat, 4*m->nmocap);
  mju_copy(d->qpos, m->qpos0, m->nq);
  mj_kinematics(m, d);

  // Initialize DER variables
  nv = n - 1;
  bigL_bar = 0.;
  j_rot << 0., -1., 1., 0.;

  // Initialize nodes and edges
  nodes.resize(nv + 2);
  edges.resize(nv + 1);
  distmat.resize(nv + 2, Eigen::Matrix<double, Eigen::Dynamic, 3>(nv + 2, 3));

  // Initialize twist displacement
  edges[nv].theta = p_thetan;
  p_thetan = std::fmod(p_thetan, (2. * M_PI));
  if (p_thetan > M_PI) {
    p_thetan -= 2 * M_PI;
  }
  theta_displace = p_thetan;

  // compute initial curvature and material properties
  for (int b = 0; b < n; b++) {
    int i = i0 + b;
    if (m->body_plugin[i] != instance) {
      mju_error("This body does not have the requested plugin instance");
    }

    // compute physical parameters
    int geom_i = m->body_geomadr[i];
    mjtNum J = 0, Iy = 0, Iz = 0;
    if (m->geom_type[geom_i] == mjGEOM_CYLINDER ||
        m->geom_type[geom_i] == mjGEOM_CAPSULE) {
      // https://en.wikipedia.org/wiki/Torsion_constant#Circle
      // https://en.wikipedia.org/wiki/List_of_second_moments_of_area
      J = mjPI * pow(m->geom_size[3*geom_i+0], 4) / 2;
      Iy = Iz = mjPI * pow(m->geom_size[3*geom_i+0], 4) / 4.;
    } else if (m->geom_type[geom_i] == mjGEOM_BOX) {
      // https://en.wikipedia.org/wiki/Torsion_constant#Rectangle
      // https://en.wikipedia.org/wiki/List_of_second_moments_of_area
      mjtNum h = m->geom_size[3*geom_i+1];
      mjtNum w = m->geom_size[3*geom_i+2];
      mjtNum a = std::max(h, w);
      mjtNum b = std::min(h, w);
      J = a*pow(b, 3)*(16./3.-3.36*b/a*(1-pow(b, 4)/pow(a, 4)/12));
      Iy = pow(2 * w, 3) * 2 * h / 12.;
      Iz = pow(2 * h, 3) * 2 * w / 12.;
    }
    alpha_bar = Iy * E;
    beta_bar = J * G;
  }
  // Get joint velocity addresses for first and lastwire bodies
  int body_id = i0 + 1; // +1 because 0th body has no joint
  int joint_id = m->body_jntadr[body_id];
  qvel0_addr = m->jnt_dofadr[joint_id];
  body_id = i0 + nv;
  joint_id = m->body_jntadr[body_id];
  qvellast_addr = m->jnt_dofadr[joint_id];

  updateVars(d);  // Call updateVars before InitBishopFrame
  InitBishopFrame();
  InitO2M(d);
  transfBF();
}

void WireQST::updateVars(mjData* d) {
  // Update node positions and orientations
  for (int i = 0; i < nv + 2; i++) {
    int body_id = i0 + i;
    nodes[i].pos << d->xpos[3*body_id], d->xpos[3*body_id+1], d->xpos[3*body_id+2];
    nodes[i].quat << d->xquat[4*body_id], d->xquat[4*body_id+1], 
                     d->xquat[4*body_id+2], d->xquat[4*body_id+3];
    // Initialize distance matrix
    distmat[i].setZero();
    // Initialize nabkb with 3 zero matrices
    nodes[i].nabkb.clear();
    for (int j = 0; j < 3; j++) {
      nodes[i].nabkb.push_back(Eigen::Matrix3d::Zero());
    }
  }

  bigL_bar = 0;
  // for i = 0
  edges[0].e = nodes[1].pos - nodes[0].pos;
  edges[0].e_bar = edges[0].e.norm();

  distmat[1].row(0) = nodes[1].pos - nodes[0].pos;

  for (int i = 1; i < nv+1; i++) {
    edges[i].e = nodes[i+1].pos - nodes[i].pos;
    edges[i].e_bar = edges[i].e.norm();
    edges[i].l_bar = edges[i].e_bar + edges[i-1].e_bar;
    bigL_bar += edges[i].l_bar;
    distmat[i+1].row(i) = nodes[i+1].pos - nodes[i].pos;
  }
  bigL_bar /= 2.;
}

void WireQST::Compute(const mjModel* m, mjData* d, int instance) {
  updateVars(d);  // Call updateVars at the start of Compute
  // Update bishop frame
  UpdateBishopFrame(d);
  // Update theta_n
  theta_n = get_thetan(d);
  updateTheta(theta_n);

  // populate rest of distance matrix
  for (int i = 2; i < (nv+2); i++) {
    distmat[i].block(0,0,i-1,3) = \
      distmat[i-1].block(0,0,i-1,3).array().rowwise() \
      + distmat[i].row(i-1).array();
    // distmat[i-1] = - distmat[i-1];
    for (int j = 0; j < (i); j++) {
      distmat[j].row(i) = distmat[i].row(j);
    }
  }
  // extra for id 1:
  distmat[0].row(1) = distmat[1].row(0); // not negative - alr done

  // Calculate forces and torques
  for (int i = 0; i < nv + 2; i++) {
    nodes[i].force.setZero();
    nodes[i].torq.setZero();
  }

  // Calculate forces using DER logic
  for (int i = 1; i < nv + 1; i++) {
    // Calculate curvature
    nodes[i].phi_i = WireUtils::calculateAngleBetween(edges[i-1].e, edges[i].e);
    nodes[i].k = 2. * std::tan(nodes[i].phi_i / 2.);
    nodes[i].kb = 2. * edges[i-1].e.cross(edges[i].e) / 
                  (edges[i-1].e_bar * edges[i].e_bar + edges[i-1].e.dot(edges[i].e));

    nodes[i].nabkb[0] = (
      (
        2 * WireUtils::createSkewSym(edges[i].e)
        + (nodes[i].kb * edges[i].e.transpose())
      )
      / (
        edges[i-1].e_bar * (edges[i].e_bar)
        + edges[i-1].e.dot(edges[i].e)
      )
    );
    nodes[i].nabkb[2] = (
      (
        2 * WireUtils::createSkewSym(edges[i-1].e)
        - (nodes[i].kb * edges[i-1].e.transpose())
      )
      / (
        edges[i-1].e_bar * (edges[i].e_bar)
        + edges[i-1].e.dot(edges[i].e)
      )
    );
    nodes[i].nabkb[1] = (- (nodes[i].nabkb[0] + nodes[i].nabkb[2]));
    nodes[i].nabpsi.row(0) = nodes[i].kb / (2 * edges[i-1].e_bar);
    nodes[i].nabpsi.row(2) = - nodes[i].kb / (2 * edges[i].e_bar);
    nodes[i].nabpsi.row(1) = - (nodes[i].nabpsi.row(0) + nodes[i].nabpsi.row(2));
    
    
    // nodes[i].force << 0., 0., 0.;
    for (int j = (i-1); j < (i+2); j++) {
      // if ((j > nv) || (j<1)) {continue;}
      nodes[j].force += - (
        2. * alpha_bar
        * nodes[i].nabkb[j-i+1].transpose() * nodes[i].kb
      ) / edges[i].l_bar;
      nodes[j].force += (
        beta_bar * (edges[nv].theta - edges[0].theta)
        * nodes[i].nabpsi.row(j-i+1)
      ) / bigL_bar;
      // std::cout << j << std::endl;
      // if (j == 0) {
      //     std::cout << nodes[j].force << std::endl;
      // }
    }
  }

  // Calculate torques using distance matrix
  Eigen::Matrix<double, Eigen::Dynamic, 3> torqvec(nv + 2, 3);
  Eigen::Matrix<double, Eigen::Dynamic, 3> torqvec_indiv(nv+2, 3);
  torqvec.setZero();
  for (int i = 0; i < nv + 2; i++) {
    torqvec_indiv = distmat[i].array().rowwise().cross(nodes[i].force);
    torqvec += torqvec_indiv;
  }
  torqvec /= 2.0;
  // mjtNum quat_inv[4];
  for (int i = 0; i < (nv+2); i++) {
    nodes[i].torq = WireUtils::rotVecQuat(
      torqvec.row(i),
      WireUtils::inverseQuat(nodes[i].quat)
    );
  }

  // Apply forces and torques to MuJoCo bodies
  // Apply all torques at once to qfrc_passive
  int num_dofs = qvellast_addr - qvel0_addr + 3;  // +3 because we include the full id for the last piece
  for (int i = 0; i < num_dofs; i++) {
    d->qfrc_passive[qvel0_addr + i] += nodes[i/3 + 1].torq[i%3];
  }
}

void WireQST::Visualize(const mjModel* m, mjData* d, mjvScene* scn, int instance) {
  if (!vmax) {
    return;
  }

  for (int b = 0; b < n; b++) {
    int i = i0 + b;
    // set geometry color based on stress norm
    mjtNum stress_m[3] = {0};
    scalar2rgba(m->geom_rgba + 4*m->body_geomadr[i], stress_m, 0, vmax);
  }
}

void WireQST::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name = "mujoco.elasticity.wire_qst";
  plugin.capabilityflags |= mjPLUGIN_PASSIVE;

  const char* attributes[] = {"twist", "bend", "flat", "vmax", "twist_displace"};
  plugin.nattribute = sizeof(attributes) / sizeof(attributes[0]);
  plugin.attributes = attributes;
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    auto elasticity_or_null = WireQST::Create(m, d, instance);
    if (!elasticity_or_null.has_value()) {
      return -1;
    }
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(
        new WireQST(std::move(*elasticity_or_null)));
    return 0;
  };

  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<WireQST*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };

  plugin.compute = +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
    auto* elasticity = reinterpret_cast<WireQST*>(d->plugin_data[instance]);
    elasticity->Compute(m, d, instance);
  };

  plugin.visualize = +[](const mjModel* m, mjData* d, const mjvOption* opt, mjvScene* scn,
                         int instance) {
    auto* elasticity = reinterpret_cast<WireQST*>(d->plugin_data[instance]);
    elasticity->Visualize(m, d, scn, instance);
  };

  mjp_registerPlugin(&plugin);
}

void WireQST::InitBishopFrame() {
  const double parll_tol = 1e-6;
  
  // Initialize first column of bishop frame
  bf0_bar.col(0) = edges[0].e / edges[0].e_bar;
  
  // Initialize second column with cross product with z-axis
  bf0_bar.col(1) = bf0_bar.col(0).cross(Eigen::Vector3d(0, 0, 1));
  
  // If the cross product is too small (vectors are nearly parallel), use y-axis instead
  if (bf0_bar.col(1).norm() < parll_tol) {
    bf0_bar.col(1) = bf0_bar.col(0).cross(Eigen::Vector3d(0, 1, 0));
  }
  
  // Normalize the second column
  bf0_bar.col(1) /= bf0_bar.col(1).norm();
  
  // Initialize third column as cross product of first two columns
  bf0_bar.col(2) = bf0_bar.col(0).cross(bf0_bar.col(1));
}

void WireQST::InitO2M(mjData* d) {
  // Get the initial orientation from the first body's quaternion
  Eigen::Quaterniond q_o0(
    d->xquat[4*i0],     // w
    d->xquat[4*i0+1],   // x
    d->xquat[4*i0+2],   // y
    d->xquat[4*i0+3]    // z
  );
  q_o0.normalize();

  // Convert bishop frame to quaternion
  Eigen::Quaterniond q_b0(bf0_bar);
  q_b0.normalize();

  // Calculate quaternion error
  Eigen::Quaterniond q_error = q_b0 * q_o0.inverse();
  q_error.normalize();

  // Transform error into local frame of q_o0
  qe_o2m_loc = q_o0.inverse() * q_error * q_o0;
  qe_o2m_loc.normalize();
}

void WireQST::UpdateBishopFrame(mjData* d) {
  // Get the quaternion directly from the first body's orientation
  Eigen::Quaterniond q_body(
    d->xquat[4*i0],     // w
    d->xquat[4*i0+1],   // x
    d->xquat[4*i0+2],   // y
    d->xquat[4*i0+3]    // z
  );
  q_body.normalize();
  
  // Apply the local rotation to get the bishop frame
  Eigen::Quaterniond q_bf = q_body * qe_o2m_loc;
  q_bf.normalize();
  
  // Convert back to matrix (no need to transpose)
  bf0_bar = q_bf.toRotationMatrix();

  // Transfer bishop frame along the wire
  transfBF();

  // Set the bishop frame at the end
  bfe = edges[nv].bf;
}

bool WireQST::transfBF() {
  bool bf_align = true;

  edges[0].bf = bf0_bar;
  
  for (int i = 1; i < nv+1; i++) {
    edges[i].bf.col(0) = edges[i].e / edges[i].e.norm();
    if (nodes[i].kb.norm() == 0) {
      edges[i].bf.col(1) = edges[i-1].bf.col(1);
    } else {
      edges[i].bf.col(1) = WireUtils::rotateVector3(
        edges[i-1].bf.col(1),
        nodes[i].kb / nodes[i].kb.norm(),
        nodes[i].phi_i
      );
      if (std::abs(edges[i].bf.col(1).dot(edges[i].bf.col(0))) > 1e-1) {
        bf_align = false;
      }
    }
    edges[i].bf.col(2) = edges[i].bf.col(0).cross(edges[i].bf.col(1));
  }
  return bf_align;
}

double WireQST::get_thetan(mjData* d) {
  // Get the quaternion of the end body
  Eigen::Quaterniond quat_on(
    d->xquat[4*(i0+nv)],     // w
    d->xquat[4*(i0+nv)+1],   // x
    d->xquat[4*(i0+nv)+2],   // y
    d->xquat[4*(i0+nv)+3]    // z
  );
  quat_on.normalize();

  // Get the bishop frame at the end (no need to transpose)
  Eigen::Matrix3d mat_bn = bfe;

  // Calculate the orientation in the material frame
  Eigen::Matrix3d mat_mn = (quat_on * qe_o2m_loc).normalized().toRotationMatrix();

  // Calculate the angle between the second columns of mat_bn and mat_mn around mat_bn's first column
  Eigen::Vector3d v1 = mat_bn.col(1);
  Eigen::Vector3d v2 = mat_mn.col(1);
  Eigen::Vector3d va = mat_bn.col(0);

  double dot_norm_val = v1.dot(v2) / (v1.norm() * v2.norm());
  if (dot_norm_val > 1.0) dot_norm_val = 1.0;
  double theta_diff = std::acos(dot_norm_val);
  if ((v1.cross(v2)).dot(va) < 0) {
    theta_diff *= -1.0;
  }

  return theta_diff + theta_displace;
}

void WireQST::updateThetaN(double theta_n) {
  double diff_theta = theta_n - p_thetan;

  // Account for 2pi rotation
  if (std::abs(diff_theta) < M_PI) {
    edges[nv].theta += diff_theta;
  } else if (diff_theta > 0.) {
    edges[nv].theta += diff_theta - (2 * M_PI);
  } else {
    edges[nv].theta += diff_theta + (2 * M_PI);
  }
  p_thetan = theta_n;
}

double WireQST::updateTheta(double theta_n) {
  double d_theta;

  updateThetaN(theta_n);
  
  d_theta = (edges[nv].theta - edges[0].theta) / nv;
  for (int i = 0; i < (nv+1); i++) {
    edges[i].theta = d_theta * i;
  }
  return edges[nv].theta;
}

}  // namespace mujoco::plugin::elasticity
