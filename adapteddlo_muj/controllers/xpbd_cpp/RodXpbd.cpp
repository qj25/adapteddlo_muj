#include "RodXpbd.h"

#include <cmath>
#include <cstring>

#include "Common/Common.h"
#include "PositionBasedElasticRods.h"

using namespace PBD;

struct RodXpbd::JointData {
    Eigen::Matrix<Real, 3, 4, Eigen::DontAlign> jointInfo;
    Vector3r stiffnessCoefficientK;
    Vector3r restDarbouxVector;
    Vector3r stretchCompliance;
    Vector3r bendingAndTorsionCompliance;
    Vector6r lambdaSum;
    Vector3r jointPos;
};

static Quaternionr quatFromArray(const double* q) {
    Quaternionr out(q[0], q[1], q[2], q[3]);
    return out;
}

static void quatToArray(const Quaternionr& q, double* out) {
    out[0] = q.w();
    out[1] = q.x();
    out[2] = q.y();
    out[3] = q.z();
}

static Vector3r vec3FromArray(const double* v) {
    return Vector3r(static_cast<Real>(v[0]), static_cast<Real>(v[1]), static_cast<Real>(v[2]));
}

static Matrix3r mat3FromArray(const double* m) {
    Matrix3r out;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out(r, c) = static_cast<Real>(m[r * 3 + c]);
        }
    }
    return out;
}

static Vector3r jointPosition(const Vector3r& x0, const Vector3r& x1) {
    return static_cast<Real>(0.5) * (x0 + x1);
}

RodXpbd::RodXpbd(
    int n_segments,
    int dim_x,
    const double* rest_x,
    int dim_q,
    const double* rest_quat,
    double segment_length,
    double radius,
    double youngs_modulus,
    double torsion_modulus,
    bool bothweld)
    : n_segments_(n_segments),
      n_joints_(n_segments > 1 ? n_segments - 1 : 0),
      bothweld_(bothweld),
      segment_length_(segment_length),
      radius_(radius),
      youngs_modulus_(youngs_modulus),
      torsion_modulus_(torsion_modulus),
      k_force_(1.0),
      k_torque_(1.0),
      num_iters_(3) {
    (void)dim_x;
    (void)dim_q;
    initJoints(rest_x, rest_quat);
}

RodXpbd::~RodXpbd() {
    clearJoints();
}

void RodXpbd::clearJoints() {
    for (JointData* j : joints_) {
        delete j;
    }
    joints_.clear();
}

void RodXpbd::initJoints(const double* rest_x, const double* rest_quat) {
    clearJoints();
    joints_.resize(static_cast<size_t>(n_joints_), nullptr);

    for (int j = 0; j < n_joints_; ++j) {
        JointData* jd = new JointData();
        const Vector3r x0 = vec3FromArray(rest_x + 3 * j);
        const Vector3r x1 = vec3FromArray(rest_x + 3 * (j + 1));
        const Quaternionr q0 = quatFromArray(rest_quat + 4 * j);
        const Quaternionr q1 = quatFromArray(rest_quat + 4 * (j + 1));
        jd->jointPos = jointPosition(x0, x1);

        DirectPositionBasedSolverForStiffRods::init_StretchBendingTwistingConstraint(
            x0,
            q0,
            x1,
            q1,
            jd->jointPos,
            static_cast<Real>(radius_),
            static_cast<Real>(segment_length_),
            static_cast<Real>(youngs_modulus_),
            static_cast<Real>(torsion_modulus_),
            jd->jointInfo,
            jd->stiffnessCoefficientK,
            jd->restDarbouxVector);

        joints_[static_cast<size_t>(j)] = jd;
    }
}

void RodXpbd::setMaterial(double youngs_modulus, double torsion_modulus) {
    youngs_modulus_ = youngs_modulus;
    torsion_modulus_ = torsion_modulus;
}

void RodXpbd::setForceGain(double k_force) {
    k_force_ = k_force;
}

void RodXpbd::setTorqueGain(double k_torque) {
    k_torque_ = k_torque;
}

void RodXpbd::setNumIterations(int num_iters) {
    num_iters_ = num_iters < 1 ? 1 : num_iters;
}

void RodXpbd::reinitRestPose(int dim_x, const double* rest_x, int dim_q, const double* rest_quat) {
    (void)dim_x;
    (void)dim_q;
    initJoints(rest_x, rest_quat);
}

void RodXpbd::computeWrenches(
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
    double* torque_out) {
    (void)dim_x;
    (void)dim_q;
    (void)dim_m;
    (void)dim_i;
    (void)dim_f;
    (void)dim_t;

    const int n = n_segments_;
    std::memset(force_out, 0, sizeof(double) * 3 * n);
    std::memset(torque_out, 0, sizeof(double) * 3 * n);

    if (n_joints_ < 1 || dt <= 0.0) {
        return;
    }

    std::vector<Vector3r> x_shadow(static_cast<size_t>(n));
    std::vector<Quaternionr> q_shadow(static_cast<size_t>(n));
    std::vector<Vector3r> x0(static_cast<size_t>(n));

    for (int i = 0; i < n; ++i) {
        x_shadow[static_cast<size_t>(i)] = vec3FromArray(x + 3 * i);
        x0[static_cast<size_t>(i)] = x_shadow[static_cast<size_t>(i)];
        q_shadow[static_cast<size_t>(i)] = quatFromArray(quat + 4 * i);
        q_shadow[static_cast<size_t>(i)].normalize();
    }

    const Real h = static_cast<Real>(dt);
    const Real inv_h2 = static_cast<Real>(1.0) / (h * h);

    for (int j = 0; j < n_joints_; ++j) {
        JointData* jd = joints_[static_cast<size_t>(j)];
        jd->lambdaSum.setZero();
        DirectPositionBasedSolverForStiffRods::initBeforeProjection_StretchBendingTwistingConstraint(
            jd->stiffnessCoefficientK,
            static_cast<Real>(1.0) / h,
            static_cast<Real>(segment_length_),
            jd->stretchCompliance,
            jd->bendingAndTorsionCompliance,
            jd->lambdaSum);
    }

    for (int iter = 0; iter < num_iters_; ++iter) {
        for (int j = 0; j < n_joints_; ++j) {
            JointData* jd = joints_[static_cast<size_t>(j)];
            const int i0 = j;
            const int i1 = j + 1;

            const Real invMass0 = static_cast<Real>(inv_mass[i0]);
            const Real invMass1 = static_cast<Real>(inv_mass[i1]);

            DirectPositionBasedSolverForStiffRods::update_StretchBendingTwistingConstraint(
                x_shadow[static_cast<size_t>(i0)],
                q_shadow[static_cast<size_t>(i0)],
                x_shadow[static_cast<size_t>(i1)],
                q_shadow[static_cast<size_t>(i1)],
                jd->jointInfo);

            Vector3r corr_x0, corr_x1;
            Quaternionr corr_q0, corr_q1;

            const bool res = DirectPositionBasedSolverForStiffRods::solve_StretchBendingTwistingConstraint(
                invMass0,
                x_shadow[static_cast<size_t>(i0)],
                mat3FromArray(inv_inertia_w + 9 * i0),
                q_shadow[static_cast<size_t>(i0)],
                invMass1,
                x_shadow[static_cast<size_t>(i1)],
                mat3FromArray(inv_inertia_w + 9 * i1),
                q_shadow[static_cast<size_t>(i1)],
                jd->restDarbouxVector,
                static_cast<Real>(segment_length_),
                jd->stretchCompliance,
                jd->bendingAndTorsionCompliance,
                jd->jointInfo,
                corr_x0,
                corr_q0,
                corr_x1,
                corr_q1,
                jd->lambdaSum);

            if (!res) {
                continue;
            }

            if (invMass0 > static_cast<Real>(0.0)) {
                x_shadow[static_cast<size_t>(i0)] += corr_x0;
                q_shadow[static_cast<size_t>(i0)].coeffs() += corr_q0.coeffs();
                q_shadow[static_cast<size_t>(i0)].normalize();
            }
            if (invMass1 > static_cast<Real>(0.0)) {
                x_shadow[static_cast<size_t>(i1)] += corr_x1;
                q_shadow[static_cast<size_t>(i1)].coeffs() += corr_q1.coeffs();
                q_shadow[static_cast<size_t>(i1)].normalize();
            }
        }
    }

    const Real kf = static_cast<Real>(k_force_);
    const Real kt = static_cast<Real>(k_torque_);
    const Real inv_dt = static_cast<Real>(1.0) / h;
    (void)inv_h2;

    for (int i = 0; i < n; ++i) {
        if (inv_mass[i] <= 0.0) {
            continue;
        }

        const Vector3r dx = x_shadow[static_cast<size_t>(i)] - x0[static_cast<size_t>(i)];
        // Velocity-level drive (softer than dx/dt^2) for MuJoCo coupling.
        force_out[3 * i + 0] = static_cast<double>(kf * dx[0] * inv_dt);
        force_out[3 * i + 1] = static_cast<double>(kf * dx[1] * inv_dt);
        force_out[3 * i + 2] = static_cast<double>(kf * dx[2] * inv_dt);

        Quaternionr dq = q_shadow[static_cast<size_t>(i)] * quatFromArray(quat + 4 * i).conjugate();
        if (dq.w() < static_cast<Real>(0.0)) {
            dq.coeffs() = -dq.coeffs();
        }
        Vector3r rotvec;
        const Real w = dq.w();
        const Vector3r v(dq.x(), dq.y(), dq.z());
        const Real vn = v.norm();
        if (vn > static_cast<Real>(1e-8)) {
            const Real angle = static_cast<Real>(2.0) * std::atan2(vn, w);
            rotvec = (angle / vn) * v;
        } else {
            rotvec = static_cast<Real>(2.0) * v;
        }

        torque_out[3 * i + 0] = static_cast<double>(kt * rotvec[0] * inv_dt);
        torque_out[3 * i + 1] = static_cast<double>(kt * rotvec[1] * inv_dt);
        torque_out[3 * i + 2] = static_cast<double>(kt * rotvec[2] * inv_dt);
    }
}
