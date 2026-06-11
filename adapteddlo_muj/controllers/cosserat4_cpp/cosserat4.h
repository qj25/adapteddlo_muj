#ifndef COSSERAT4_H
#define COSSERAT4_H

#include "Cosserat4_obj.h"
#include "Cosserat4_utils.h"
#include <vector>
#include "Eigen/Core"
#include "Eigen/Geometry"

/*
RodCosserat4:
Centerline elastic forces/torques with fullDyn twist (wire.cc fullDyn):
per-link stored twist and adjacent-segment twist stiffness.
*/

class RodCosserat4
{
public:
    RodCosserat4(
        int dim_np,
        double *node_pos,
        int dim_bf0,
        double *bf0sim,
        const double theta_n,
        const double overall_rot,
        const double a_bar,
        const double b_bar
    );

    std::vector <Vecnodes, Eigen::aligned_allocator<Vecnodes> > nodes;
    double overall_rot;

    int d_vec;
    int nv;
    std::vector <SegEdges, Eigen::aligned_allocator<SegEdges> > edges;

    double bigL_bar;
    Eigen::Matrix3d bf0_bar;

    double alpha_bar;
    double beta_bar;
    Eigen::Matrix2d j_rot;

    double p_thetan;

    Eigen::Matrix3d bf0mat;

    Eigen::Quaterniond qe_o2m_loc;
    Eigen::Quaterniond qe_m2o_loc;

    int excl_joints;
    int nintgsteps;
    double step_const;
    double step_gain;
    std::vector <Eigen::Matrix<double, Eigen::Dynamic, 3>, Eigen::aligned_allocator<Eigen::Matrix<double, Eigen::Dynamic, 3>> > distmat;

    bool updateVars(
        int dim_np,
        double *node_pos,
        int dim_bf0,
        double *bf0sim,
        int dim_bfe,
        double *bfesim
    );

    void calculateCenterlineF2(int dim_nf, double *node_force);

    void calculateCenterlineTorq(
        int dim_nt, double *node_torq,
        int dim_nq, double *node_quat,
        int excl_jnts
    );

    void calculateF2LocalTorq();

    void updateThetasFullDyn(int dim_nq, double *node_quat);

    void resetTheta(double theta_n, double overall_rot);

    void changeAlphaBeta(double a_bar, double b_bar);

    void initQe_o2m_loc(int dim_qo2m, double *q_o2m);

    void initO2MLocAll(int dim_nq, double *node_quat);

    void calculateOf2Mf(
        int dim_mato, double *mat_o,
        int dim_matres, double *mat_res
    );

    void calculateOf2MfAtEdge(
        int edge_idx,
        int dim_mato, double *mat_o,
        int dim_matres, double *mat_res
    );

    double angBtwn3(
        int dim_v1, double *v1,
        int dim_v2, double *v2,
        int dim_va, double *va
    );

    double calculateEnergy();

private:
    void initVars(
        int dim_np,
        double *node_pos,
        int dim_bf0,
        double *bf0sim
    );

    void initEdgeTheta(const double overall_rot);

    void update_XVecs(const double *node_pos);

    void updateX2E();
    void updateE2K();
    void updateE2Kb();
    bool transfBF(const Eigen::Matrix3d &bf_0);

    double getThetaLoc(int idx, const Eigen::Vector4d &body_quat);
    void updateThetaLoc(double theta_loc, int idx);
    void initO2MLoc(int idx, const Eigen::Vector4d &body_quat);

    void calculateNabKbandNabPsi_sub2(const int start_i, const int end_i);
    void addThetaTwistTorq();

    double calculateBendingEnergy();
    double calculateTwistingEnergy();
};

#endif
