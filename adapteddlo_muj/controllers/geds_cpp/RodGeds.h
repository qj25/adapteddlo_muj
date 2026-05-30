#ifndef ROD_GEDS_H
#define ROD_GEDS_H

class RodGeds {
public:
    RodGeds(
        int n_nodes,
        double segment_length,
        double diameter,
        double youngs_modulus,
        double torsion_modulus);

    ~RodGeds();

    void setMaterial(double youngs_modulus, double torsion_modulus);
    void setNumSamples(int samples_per_span);

    void reinitRest(
        int dim_x,
        const double* rest_x,
        int dim_q,
        const double* rest_quat);

    void computeElasticWrenches(
        int dim_x,
        const double* x,
        int dim_q,
        const double* quat,
        int dim_f,
        double* force_out,
        int dim_t,
        double* torque_out);

private:
    int n_nodes_;
    double segment_length_;
    double diameter_;
    double youngs_modulus_;
    double torsion_modulus_;
    int samples_per_span_;

    double k_bend_;
    double k_twist_;

    double* rest_x_;
    double* rest_theta_;
    double* rest_kappa_;
    double* rest_twist_;
    int n_rest_samples_;

    void updateStiffness();
    void buildRestStrainSamples();
    double totalEnergy(
        const double* pos,
        const double* theta) const;
};

#endif
