#ifndef ROD_COSSERAT2_H
#define ROD_COSSERAT2_H

class RodCosserat2 {
public:
    RodCosserat2(
        int n_nodes,
        int dim_x,
        const double* rest_x,
        int dim_q,
        const double* rest_quat,
        double segment_length,
        double radius,
        double k_stretch,
        double k_bend,
        double k_twist,
        bool bothweld);

    ~RodCosserat2();

    void setMaterial(double k_stretch, double k_bend, double k_twist);
    void setForceGain(double k_force);
    void setTorqueGain(double k_torque);
    void setNumIterations(int num_iters);

    void reinitRest(int dim_x, const double* rest_x, int dim_q, const double* rest_quat);

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
        double* torque_out);

private:
    int n_nodes_;
    int n_segments_;
    bool bothweld_;
    double segment_length_;
    double radius_;
    double k_stretch_;
    double k_bend_;
    double k_twist_;
    double k_force_;
    double k_torque_;
    int num_iters_;

    struct SimState;
    SimState* sim_;

    void buildChain(const double* rest_x, const double* rest_quat);
    void destroySim();
};

#endif
