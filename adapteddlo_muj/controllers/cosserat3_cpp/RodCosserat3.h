#ifndef RODCOSSERAT3_H
#define RODCOSSERAT3_H

#include "Cosserat3_obj.h"
#include "Cosserat3_utils.h"
#include <vector>
#include "Eigen/Core"
#include "Eigen/Geometry"

/*
 * Kirchhoff rod elastic evaluator (JTill2017, inextensible case).
 * Quasi-static nodal forces / body torques from m = K(u - u*).
 * MuJoCo integrates dynamics; no shadow solve.
 */
class RodCosserat3
{
public:
    RodCosserat3(
        int dim_np,
        double* node_pos,
        int dim_bf0,
        double* bf0sim,
        const double theta_n,
        const double overall_rot,
        const double a_bar,
        const double b_bar,
        const double radius
    );

    std::vector<Vecnodes, Eigen::aligned_allocator<Vecnodes>> nodes;
    double overall_rot;

    int d_vec;
    int nv;
    std::vector<SegEdges, Eigen::aligned_allocator<SegEdges>> edges;

    double bigL_bar;
    Eigen::Matrix3d bf0_bar;
    double alpha_bar;
    double beta_bar;
    double radius_;
    double k_bend_;
    double k_twist_;
    double p_thetan;
    double theta_star_total_;

    Eigen::Matrix3d bf0mat;
    Eigen::Quaterniond qe_o2m_loc;

    int excl_joints;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>, Eigen::aligned_allocator<Eigen::Matrix<double, Eigen::Dynamic, 3>>> distmat;

    bool updateVars(
        int dim_np,
        double* node_pos,
        int dim_bf0,
        double* bf0sim,
        int dim_bfe,
        double* bfesim
    );

    void calculateCenterlineF2(int dim_nf, double* node_force);
    void calculateCenterlineTorq(
        int dim_nt, double* node_torq,
        int dim_nq, double* node_quat,
        int excl_jnts
    );

    double updateTheta(double theta_n);
    void resetTheta(double theta_n, double overall_rot);
    void changeAlphaBeta(double a_bar, double b_bar);
    void captureRestCurvature();

    void initQe_o2m_loc(int dim_qo2m, double* q_o2m);
    void calculateOf2Mf(int dim_mato, double* mat_o, int dim_matres, double* mat_res);
    double angBtwn3(int dim_v1, double* v1, int dim_v2, double* v2, int dim_va, double* va);
    double calculateEnergy();

private:
    void initVars(int dim_np, double* node_pos, int dim_bf0, double* bf0sim);
    void updateStiffnessFromMaterial();
    void update_XVecs(const double* node_pos);
    void updateX2E();
    void updateE2K();
    void updateE2Kb();
    bool transfBF(const Eigen::Matrix3d& bf_0);
    void updateMaterialCurvature();
    void updateThetaN(double theta_n);

    void calculateKirchhoffForces_sub(const int start_i, const int end_i);
    void calculateF2LocalTorq();

    double calculateBendingEnergy();
    double calculateTwistingEnergy();

    Eigen::Vector3d materialCurvatureAtNode(int i) const;
    Eigen::Vector3d kbEffectiveAtNode(int i) const;
};

#endif
