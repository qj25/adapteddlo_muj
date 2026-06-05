#ifndef ROD_COSSERAT_H
#define ROD_COSSERAT_H

class RodCosserat {
public:
    RodCosserat(
        int n_nodes,
        double segment_length,
        double k_bend,
        double k_twist);

    ~RodCosserat();

    void setMaterial(double k_bend, double k_twist);
    void setNumIterations(int num_iters);
    void setTorqueGain(double k_torque);

    void reinitRest(
        int dim_x,
        const double* rest_x,
        int dim_q,
        const double* rest_quat);

    void computeWrenches(
        int dim_x,
        const double* x,
        int dim_q,
        const double* quat,
        double dt,
        int dim_t,
        double* torque_out);

private:
    int n_nodes_;
    int n_joints_;
    double segment_length_;
    double k_bend_;
    double k_twist_;
    double k_torque_;
    int num_iters_;

    double* rest_rel_quat_;

    void buildRestAngles(int dim_q, const double* rest_quat);
};

#endif
