#include "RodGeds.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "Eigen/Geometry"
#include "CatmullRom.h"
#include "MinimalFrame.h"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEps = 1e-12;
constexpr double kFdEps = 1e-7;

Eigen::Vector3d getCP(const double* pos, int n, int idx) {
    if (idx < 0) {
        return Eigen::Vector3d(pos[0], pos[1], pos[2]);
    }
    if (idx >= n) {
        return Eigen::Vector3d(pos[3 * (n - 1) + 0], pos[3 * (n - 1) + 1], pos[3 * (n - 1) + 2]);
    }
    return Eigen::Vector3d(pos[3 * idx + 0], pos[3 * idx + 1], pos[3 * idx + 2]);
}

double getTheta(const double* theta, int n, int idx) {
    if (idx < 0) {
        return theta[0];
    }
    if (idx >= n) {
        return theta[n - 1];
    }
    return theta[idx];
}

void sampleStrainAt(
    const double* pos,
    int n_nodes,
    const double* theta,
    int seg,
    double t,
    double& kappa,
    double& twist) {
    const Eigen::Vector3d p0 = getCP(pos, n_nodes, seg - 1);
    const Eigen::Vector3d p1 = getCP(pos, n_nodes, seg);
    const Eigen::Vector3d p2 = getCP(pos, n_nodes, seg + 1);
    const Eigen::Vector3d p3 = getCP(pos, n_nodes, seg + 2);

    const Eigen::Vector3d rp = geds::CatmullRom::derivative1(p0, p1, p2, p3, t);
    const Eigen::Vector3d rpp = geds::CatmullRom::derivative2(p0, p1, p2, p3, t);
    const Eigen::Vector3d rppp = geds::CatmullRom::derivative3(p0, p1, p2, p3, t);

    const Eigen::Vector3d C = rp.cross(rpp);
    const double rp_n = rp.norm();
    const double C_n2 = C.squaredNorm();

    if (rp_n < kEps) {
        kappa = 0.0;
        twist = 0.0;
        return;
    }

    kappa = std::sqrt(C_n2) / (rp_n * rp_n * rp_n);

    double tau_geom = 0.0;
    if (C_n2 > kEps) {
        tau_geom = C.dot(rppp) / C_n2;
    }

    const double th0 = getTheta(theta, n_nodes, seg - 1);
    const double th1 = getTheta(theta, n_nodes, seg);
    const double th2 = getTheta(theta, n_nodes, seg + 1);
    const double th3 = getTheta(theta, n_nodes, seg + 2);
    const double theta_p = geds::CatmullRom::scalarDerivative1(th0, th1, th2, th3, t);

    twist = theta_p + tau_geom;
}

}  // namespace

RodGeds::RodGeds(
    int n_nodes,
    double segment_length,
    double diameter,
    double youngs_modulus,
    double torsion_modulus)
    : n_nodes_(n_nodes),
      segment_length_(segment_length),
      diameter_(diameter),
      youngs_modulus_(youngs_modulus),
      torsion_modulus_(torsion_modulus),
      samples_per_span_(4),
      rest_x_(nullptr),
      rest_theta_(nullptr),
      rest_kappa_(nullptr),
      rest_twist_(nullptr),
      n_rest_samples_(0) {
    updateStiffness();
    rest_x_ = new double[3 * n_nodes_];
    rest_theta_ = new double[n_nodes_];
    std::memset(rest_x_, 0, sizeof(double) * 3 * n_nodes_);
    std::memset(rest_theta_, 0, sizeof(double) * n_nodes_);
}

RodGeds::~RodGeds() {
    delete[] rest_x_;
    delete[] rest_theta_;
    delete[] rest_kappa_;
    delete[] rest_twist_;
}

void RodGeds::updateStiffness() {
    const double d = diameter_;
    const double d4 = d * d * d * d;
    k_bend_ = youngs_modulus_ * kPi * d4 / 64.0;
    k_twist_ = torsion_modulus_ * kPi * d4 / 32.0;
}

void RodGeds::setMaterial(double youngs_modulus, double torsion_modulus) {
    youngs_modulus_ = youngs_modulus;
    torsion_modulus_ = torsion_modulus;
    updateStiffness();
}

void RodGeds::setNumSamples(int samples_per_span) {
    samples_per_span_ = std::max(1, samples_per_span);
}

void RodGeds::buildRestStrainSamples() {
    delete[] rest_kappa_;
    delete[] rest_twist_;
    rest_kappa_ = nullptr;
    rest_twist_ = nullptr;

    if (n_nodes_ < 2) {
        n_rest_samples_ = 0;
        return;
    }

    const int n_seg = n_nodes_ - 1;
    n_rest_samples_ = n_seg * samples_per_span_;
    rest_kappa_ = new double[n_rest_samples_];
    rest_twist_ = new double[n_rest_samples_];

    int idx = 0;
    for (int seg = 0; seg < n_seg; ++seg) {
        for (int s = 0; s < samples_per_span_; ++s) {
            const double t = (static_cast<double>(s) + 0.5) / static_cast<double>(samples_per_span_);
            sampleStrainAt(rest_x_, n_nodes_, rest_theta_, seg, t, rest_kappa_[idx], rest_twist_[idx]);
            ++idx;
        }
    }
}

void RodGeds::reinitRest(
    int dim_x,
    const double* rest_x,
    int dim_q,
    const double* rest_quat) {
    (void)dim_q;
    const int n = std::min(n_nodes_, dim_x / 3);
    for (int i = 0; i < n; ++i) {
        rest_x_[3 * i + 0] = rest_x[3 * i + 0];
        rest_x_[3 * i + 1] = rest_x[3 * i + 1];
        rest_x_[3 * i + 2] = rest_x[3 * i + 2];
    }

    std::vector<Eigen::Vector3d> positions(static_cast<size_t>(n_nodes_));
    for (int i = 0; i < n_nodes_; ++i) {
        positions[static_cast<size_t>(i)] = getCP(rest_x_, n_nodes_, i);
    }

    std::vector<geds::Frame> frames;
    geds::MinimalFrame::propagate(positions, frames);

    for (int i = 0; i < n; ++i) {
        Eigen::Vector4d q;
        q << rest_quat[4 * i + 0],
            rest_quat[4 * i + 1],
            rest_quat[4 * i + 2],
            rest_quat[4 * i + 3];
        rest_theta_[i] = geds::MinimalFrame::rollFromQuat(q, frames[static_cast<size_t>(i)]);
    }

    buildRestStrainSamples();
}

double RodGeds::totalEnergy(const double* pos, const double* theta) const {
    if (n_nodes_ < 2 || n_rest_samples_ <= 0) {
        return 0.0;
    }

    const int n_seg = n_nodes_ - 1;
    double energy = 0.0;
    int rest_idx = 0;

    for (int seg = 0; seg < n_seg; ++seg) {
        for (int s = 0; s < samples_per_span_; ++s) {
            const double t = (static_cast<double>(s) + 0.5) / static_cast<double>(samples_per_span_);

            double kappa = 0.0;
            double twist = 0.0;
            sampleStrainAt(pos, n_nodes_, theta, seg, t, kappa, twist);

            const Eigen::Vector3d p0 = getCP(pos, n_nodes_, seg - 1);
            const Eigen::Vector3d p1 = getCP(pos, n_nodes_, seg);
            const Eigen::Vector3d p2 = getCP(pos, n_nodes_, seg + 1);
            const Eigen::Vector3d p3 = getCP(pos, n_nodes_, seg + 2);
            const Eigen::Vector3d rp = geds::CatmullRom::derivative1(p0, p1, p2, p3, t);
            const double ds = std::max(rp.norm(), kEps) / static_cast<double>(samples_per_span_);

            const double dk = kappa - rest_kappa_[rest_idx];
            const double dtw = twist - rest_twist_[rest_idx];
            energy += 0.5 * k_bend_ * dk * dk * ds;
            energy += 0.5 * k_twist_ * dtw * dtw * ds;
            ++rest_idx;
        }
    }

    return energy;
}

void RodGeds::computeElasticWrenches(
    int dim_x,
    const double* x,
    int dim_q,
    const double* quat,
    int dim_f,
    double* force_out,
    int dim_t,
    double* torque_out) {
    const int n = std::min(n_nodes_, std::min(dim_x / 3, dim_q / 4));
    const int n_f = dim_f / 3;
    const int n_t = dim_t / 3;

    for (int i = 0; i < dim_f; ++i) {
        force_out[i] = 0.0;
    }
    for (int i = 0; i < dim_t; ++i) {
        torque_out[i] = 0.0;
    }

    if (n < 2) {
        return;
    }

    std::vector<double> pos(3 * n_nodes_, 0.0);
    std::vector<double> theta(n_nodes_, 0.0);
    for (int i = 0; i < n; ++i) {
        pos[3 * i + 0] = x[3 * i + 0];
        pos[3 * i + 1] = x[3 * i + 1];
        pos[3 * i + 2] = x[3 * i + 2];
    }

    std::vector<Eigen::Vector3d> positions(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        positions[static_cast<size_t>(i)] = getCP(pos.data(), n_nodes_, i);
    }

    std::vector<geds::Frame> frames;
    geds::MinimalFrame::propagate(positions, frames);

    for (int i = 0; i < n; ++i) {
        Eigen::Vector4d q;
        q << quat[4 * i + 0],
            quat[4 * i + 1],
            quat[4 * i + 2],
            quat[4 * i + 3];
        theta[static_cast<size_t>(i)] = geds::MinimalFrame::rollFromQuat(q, frames[static_cast<size_t>(i)]);
    }

    for (int i = 0; i < n && i < n_f; ++i) {
        for (int d = 0; d < 3; ++d) {
            pos[3 * i + d] += kFdEps;
            const double e_plus = totalEnergy(pos.data(), theta.data());
            pos[3 * i + d] -= 2.0 * kFdEps;
            const double e_minus = totalEnergy(pos.data(), theta.data());
            pos[3 * i + d] += kFdEps;
            force_out[3 * i + d] = -((e_plus - e_minus) / (2.0 * kFdEps));
        }
    }

    for (int i = 0; i < n && i < n_t; ++i) {
        theta[static_cast<size_t>(i)] += kFdEps;
        const double e_plus = totalEnergy(pos.data(), theta.data());
        theta[static_cast<size_t>(i)] -= 2.0 * kFdEps;
        const double e_minus = totalEnergy(pos.data(), theta.data());
        theta[static_cast<size_t>(i)] += kFdEps;

        const double roll_torque = -((e_plus - e_minus) / (2.0 * kFdEps));
        const Eigen::Vector3d t = frames[static_cast<size_t>(i)].tangent;
        torque_out[3 * i + 0] = roll_torque * t(0);
        torque_out[3 * i + 1] = roll_torque * t(1);
        torque_out[3 * i + 2] = roll_torque * t(2);
    }
}
