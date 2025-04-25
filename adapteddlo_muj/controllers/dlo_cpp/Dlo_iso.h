#ifndef DLOISO_H
#define DLOISO_H

#include "Dlo_obj.h"
#include "Dlo_utils.h"
#include <vector>
#include "Eigen/Core"
#include "Eigen/Geometry"
// #include <thread>

/*
DLO_iso:
1. calculates forces on each node in an arrangement of equality constrained 
discrete cylinders.
2. refer to paper: Simulation and Manipulation of a Deformable Linear Object
*/

class DLO_iso
{
public:
    DLO_iso(
        int dim_np,
        double *node_pos,
        int dim_bf0,
        double *bf0sim,
        const double theta_n,
        const double overall_rot,
        const double a_bar,
        const double b_bar
    );

    // ~DLO_iso();
    
    std::vector <Vecnodes, Eigen::aligned_allocator<Vecnodes> > nodes;
    double overall_rot;

    int d_vec;
    int nv;
    std::vector <SegEdges, Eigen::aligned_allocator<SegEdges> > edges;
    
    // init _variables
    double bigL_bar;
    Eigen::Matrix3d bf0_bar;

    // define variable constants
    double alpha_bar;
    double beta_bar;
    Eigen::Matrix2d j_rot;
    
    double p_thetan;

    // Cpp2Py vars
    Eigen::Matrix3d bf0mat;

    // misc calc vars
    Eigen::Quaterniond qe_o2m_loc;
    Eigen::Quaterniond qe_m2o_loc;

    // new DLO_iso things
    int excl_joints;
    int nintgsteps;
    double step_const;
    double step_gain;
    std::vector <Eigen::Matrix<double, Eigen::Dynamic, 3>, Eigen::aligned_allocator<Eigen::Matrix<double, Eigen::Dynamic, 3>> > distmat;

    // // threading variables
    // int s_i, e_i, d_i, n_threads, over_i;
    // std::vector<std::thread> nkbpsiThreads;
    // // std::vector <Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > force_sub;
    
    // Functions:
    // void initVars(
    //     const std::vector <Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &node_pos, 
    //     const Eigen::Matrix3d &bf0sim
    // );  //
    bool updateVars(
        int dim_np,
        double *node_pos,
        int dim_bf0,
        double *bf0sim,
        int dim_bfe,
        double *bfesim
    );  //

    void calculateCenterlineF2(int dim_nf, double *node_force);  // multithreading w/ force-nab loop share

    void calculateCenterlineTorq(
        int dim_nt, double *node_torq,
        // int dim_ntg, double *node_torq_global,
        int dim_nq, double *node_quat,
        int excl_jnts
    );

    void calculateF2LocalTorq();

    double updateTheta(double theta_n);   //

    void resetTheta(double theta_n, double overall_rot); //

    void changeAlphaBeta(double a_bar, double b_bar); //

    void initQe_o2m_loc(int dim_qo2m, double *q_o2m);

    void calculateOf2Mf(
        int dim_mato, double *mat_o,
        int dim_matres, double *mat_res
    );

    double angBtwn3(
        int dim_v1, double *v1,
        int dim_v2, double *v2,
        int dim_va, double *va
    );

    double calculateEnergy();
    // void changeintgstep(const int new_nis);
    // void changestepgain(const double new_sg);

private:
    void initVars(
        int dim_np,
        double *node_pos,
        int dim_bf0,
        double *bf0sim
    );  //
    // void initThreads(); //

    // void updateVars(
    //     const std::vector <Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &node_pos,
    //     const Eigen::Matrix3d &bf0sim
    // );  //

    // void update_XVecs(const std::vector <Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &node_pos);
    void update_XVecs(const double *node_pos);
    
    void updateX2E();   //
    void updateE2K();   //
    void updateE2Kb();  //
    // void initBF(Eigen::Matrix3d &bf_0);  //
    bool transfBF(const Eigen::Matrix3d &bf_0);    //
    
    void updateThetaN_old(double theta_n);  //
    void updateThetaN(double theta_n);  //

    // main calculation
    // void calculateNabKbandNabPsi(); // multithreading
    // void calculateNabKbandNabPsi_sub(const int start_i, const int end_i); //
    void calculateNabKbandNabPsi_sub2(const int start_i, const int end_i); // force-nab loop share

    double calculateBendingEnergy();
    double calculateTwistingEnergy();

    // void calculateCenterlineF();  // multithreading

    // // bring this to python (maybe)
    // void dampForce();
    // void limitForce();
    // void limitTotalForce();

    // void updateForce(
    //     const std::vector <Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &node_pos,
    //     double overall_rot,
    //     double theta_n
    // );

    // double calculateEBend();
    // double calculateETwist();

    // // Function to help integration w/ Python
};

#endif
