#ifndef DLOS2F_H
#define DLOS2F_H

#include "Ds2f_obj.h"
#include "Ds2f_utils.h"
#include <vector>
#include "Eigen/Core"
#include "Eigen/Geometry"
// #include <thread>

/*
DLO_s2f:
1. calculates forces on each node in an arrangement of equality constrained 
discrete cylinders.
2. refer to paper: Simulation and Manipulation of a Deformable Linear Object
*/

class DLO_s2f
{
public:
    DLO_s2f(
        const double r_length,
        const int r_pieces,
        const double r_weight = 0.0,
        const bool boolErrs = false
    );

    // ~DLO_s2f();
    
    int nv;
    std::vector <Vecnodes, Eigen::aligned_allocator<Vecnodes> > nodes;
    std::vector <Extforces, Eigen::aligned_allocator<Extforces> > extforces;
    double length;
    Eigen::Vector3d w_perpiece;

    int n_force;

    bool boolUnreliable = false;
    bool raiseErrs = false;

    std::vector <Eigen::Matrix<double, Eigen::Dynamic, 3>, Eigen::aligned_allocator<Eigen::Matrix<double, Eigen::Dynamic, 3>> > distmat;
    Eigen::Matrix<double, Eigen::Dynamic, 3> nodeposMat;
    // Eigen::Matrix<double, Eigen::Dynamic, 3> forceMat;
    // Eigen::Matrix<double, Eigen::Dynamic, 3> torq_glob;
    // Eigen::Matrix<double, Eigen::Dynamic, 3> force_;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matA;
    Eigen::VectorXd vecB;
    Eigen::VectorXd fSol;
    Eigen::VectorXd tSol;

    std::vector<int> secMid1;
    std::vector<int> secMid2;

    bool calculateExternalForces(
        int dim_efp, double *ext_forcepos,
        int dim_ef, double *ext_force,    
        int dim_et, double *ext_torq,
        int dim_nt, double *node_torq,
        int dim_np, double *node_pos,
        int dim_nq, double *node_quat
    );

    void updateState(
        int dim_nt, double *node_torq,
        int dim_np, double *node_pos,
        int dim_nq, double *node_quat
    );
    
private:
    void classifyExtForces(
        int dim_efp, double *ext_forcepos
    );
    bool solvable_checks();
    void calculateLocal2GlobalTorq();

    void getMidSections();
    void formSolvingMatrices();

    void solveTorques();

    void allocateFT(
        int dim_ef, double *ext_force,
        int dim_et, double *ext_torq    
    );
};

#endif
