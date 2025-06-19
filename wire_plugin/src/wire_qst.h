#ifndef MUJOCO_SRC_PLUGIN_ELASTICITY_WIREQST_H_
#define MUJOCO_SRC_PLUGIN_ELASTICITY_WIREQST_H_

#include <optional>
#include <vector>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mjvisualize.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace mujoco::plugin::elasticity {

// Node structure for wire simulation
struct NodeQST {
  Eigen::Vector3d pos;
  Eigen::Vector3d force;
  Eigen::Vector3d torq;
  Eigen::Vector4d quat;
  double phi_i;
  double k;
  Eigen::Vector3d kb;
  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> nabkb;
  Eigen::Matrix3d nabpsi;
};

// Edge structure for wire simulation
struct EdgeQST {
  Eigen::Vector3d e;
  Eigen::Matrix3d bf;
  double theta;
  double e_bar;
  double l_bar;
};

class WireQST {
 public:
  // Creates a new WireQST instance (allocated with `new`) or
  // returns null on failure.
  static std::optional<WireQST> Create(const mjModel* m, mjData* d, int instance);
  WireQST(WireQST&&) = default;
  ~WireQST() = default;

  void Compute(const mjModel* m, mjData* d, int instance);
  void Visualize(const mjModel* m, mjData* d, mjvScene* scn, int instance);

  static void RegisterPlugin();

  // Core simulation parameters
  int i0;                         // index of first body
  int n;                          // number of bodies in the wire
  mjtNum vmax;                    // max value in colormap

  // DER cpp variables
  int qvel0_addr = -1;
  int qvellast_addr = -1;
  std::vector<NodeQST, Eigen::aligned_allocator<NodeQST>> nodes;
  std::vector<EdgeQST, Eigen::aligned_allocator<EdgeQST>> edges;
  int nv;
  double alpha_bar;
  double beta_bar;
  double bigL_bar;
  Eigen::Matrix3d bf0_bar;
  Eigen::Matrix2d j_rot;
  double theta_n;
  double p_thetan;
  double theta_displace;
  Eigen::Matrix3d bf0mat;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>, 
              Eigen::aligned_allocator<Eigen::Matrix<double, Eigen::Dynamic, 3>>> distmat;
  Eigen::Quaterniond qe_o2m_loc;
  Eigen::Matrix3d bfe;  // Bishop frame at the end of the wire

  // Bishop frame initialization
  void InitBishopFrame();
  void InitO2M(mjData* d);
  void UpdateBishopFrame(mjData* d);
  bool transfBF();
  double get_thetan(mjData* d);
  void updateThetaN(double theta_n);
  double updateTheta(double theta_n);
  void updateVars(mjData* d);

 private:
  WireQST(const mjModel* m, mjData* d, int instance);
};

}  // namespace mujoco::plugin::elasticity

#endif  // MUJOCO_SRC_PLUGIN_ELASTICITY_WIREQST_H_
