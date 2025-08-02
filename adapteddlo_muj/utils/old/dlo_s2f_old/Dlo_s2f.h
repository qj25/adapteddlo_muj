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

Notes:
    - if first few sections line up, then they are from undisturbed end.
    - torques are calculated with simple torq balance for DS,
        use force in midpoint wire s section
    - alternative is assuming no torque, find the force location which 
        balances the eqn.
*/

class DLO_s2f
{
public:
    DLO_s2f(
        const double r_length,
        const int r_pieces,
        const double r_weight = 0.0,
        const bool boolErrs = false,
        const bool boolSolveTorq = false
    );

    // ~DLO_s2f();
    
    int nv;
    double length;
    Eigen::Vector3d w_perpiece;

    int n_force;
    std::vector<ForceSection> force_sections;  // store the sections

    bool raiseErrs = false;
    bool bSolveTorq = false;

    Eigen::Matrix<double, Eigen::Dynamic, 3> nodeposMat;
    Eigen::Matrix<double, Eigen::Dynamic, 3> nodetorqMat;

    bool calculateExternalForces(
        int dim_nt, double *node_torq,
        int dim_np, double *node_pos,
        int dim_nq, double *node_quat
    );

    void updateState(
        int dim_nt, double *node_torq,
        int dim_np, double *node_pos,
        int dim_nq, double *node_quat
    );
    
    void findParallelEndSections();
    void checkUDEndSections();
    bool hasParallelVectors(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const Eigen::Vector3d& v3, double threshold = 0.999);
    Eigen::Vector3d findExactMidpoint(int startId, int endId);
    Eigen::Vector3d findMiddleNodePosition(int startId, int endId);
    Eigen::Vector3d findTorqueBalancePoint(int startId, int endId, const Eigen::Vector3d& force, const Eigen::Vector3d& targetTorque);

private:
    bool checkConsistency();

    void solveTorques();
    void solveForcePos();

    std::vector<Section> undisturbed_sections_;
    std::vector<Section> disturbed_sections_;
    int undistStart = -1;    // End index of parallel section at start
    int undistEnd = -1;    // Start index of parallel section at end
};

#endif
