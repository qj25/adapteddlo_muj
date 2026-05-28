#ifndef MASSSPRING_H
#define MASSSPRING_H

#include "Eigen/Core"

class MassSpring {
public:
    MassSpring(
        int dim_nq,
        double* neutral_quat,
        double k_bend_x,
        double k_bend_y,
        double k_twist
    );

    ~MassSpring();

    void setNeutralQuat(int dim_nq, double* neutral_quat);
    void setStiffness(double k_bend_x, double k_bend_y, double k_twist);
    void computeTorque(int dim_cq, double* current_quat, int dim_nt, double* node_torque);

private:
    Eigen::Vector4d normalizeQuat(const Eigen::Vector4d& q) const;
    Eigen::Vector4d invertQuat(const Eigen::Vector4d& q) const;
    Eigen::Vector4d multiplyQuat(const Eigen::Vector4d& qa, const Eigen::Vector4d& qb) const;
    Eigen::Vector3d quatToRotvec(const Eigen::Vector4d& q) const;

    int n_nodes_;
    Eigen::Vector4d* neutral_quat_;

    double k_bend_x_;
    double k_bend_y_;
    double k_twist_;
};

#endif
