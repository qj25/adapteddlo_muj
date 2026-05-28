#ifndef RODXPBD_H
#define RODXPBD_H

#include <vector>

class RodXpbd {
public:
    RodXpbd(
        int n_segments,
        int dim_x,
        const double* rest_x,
        int dim_q,
        const double* rest_quat,
        double segment_length,
        double radius,
        double youngs_modulus,
        double torsion_modulus,
        bool bothweld
    );

    ~RodXpbd();

    void setMaterial(double youngs_modulus, double torsion_modulus);
    void setForceGain(double k_force);
    void setTorqueGain(double k_torque);
    void setNumIterations(int num_iters);

    void reinitRestPose(int dim_x, const double* rest_x, int dim_q, const double* rest_quat);

    void computeWrenches(
        int dim_x,
        const double* x,
        int dim_q,
        const double* quat,
        int dim_m,
        const double* inv_mass,
        int dim_i,
        const double* inv_inertia_w,
        double dt,
        int dim_f,
        double* force_out,
        int dim_t,
        double* torque_out
    );

private:
    struct JointData;

    int n_segments_;
    int n_joints_;
    bool bothweld_;
    double segment_length_;
    double radius_;
    double youngs_modulus_;
    double torsion_modulus_;
    double k_force_;
    double k_torque_;
    int num_iters_;

    std::vector<JointData*> joints_;

    void initJoints(const double* rest_x, const double* rest_quat);
    void clearJoints();
};

#endif
