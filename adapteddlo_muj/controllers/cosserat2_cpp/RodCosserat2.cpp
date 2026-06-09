#include "RodCosserat2.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "vendor/cosserat/CosseratTypes.h"

using namespace Cosserat;
using namespace glm;

struct RodCosserat2::SimState {
    Sim sim;
    std::vector<double> rest_quat;
};

static vec3 vec3FromArray(const double* v) {
    return vec3(static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2]));
}

static mat3 mujocoQuatToMat(const double* q) {
    const double w = q[0];
    const double x = q[1];
    const double y = q[2];
    const double z = q[3];
    mat3 m(1.0f);
    m[0][0] = static_cast<float>(1 - 2 * (y * y + z * z));
    m[0][1] = static_cast<float>(2 * (x * y - w * z));
    m[0][2] = static_cast<float>(2 * (x * z + w * y));
    m[1][0] = static_cast<float>(2 * (x * y + w * z));
    m[1][1] = static_cast<float>(1 - 2 * (x * x + z * z));
    m[1][2] = static_cast<float>(2 * (y * z - w * x));
    m[2][0] = static_cast<float>(2 * (x * z - w * y));
    m[2][1] = static_cast<float>(2 * (y * z + w * x));
    m[2][2] = static_cast<float>(1 - 2 * (x * x + y * y));
    return m;
}

static Rotor mujocoQuatToRotor(const double* q) {
    return Rotor::fromMatrix(mujocoQuatToMat(q));
}

static vec3 safeNormalize(vec3 v, vec3 fallback) {
    const float n = length(v);
    if (n < 1e-12f) {
        return fallback;
    }
    return v / n;
}

static vec3 rotorToRotvec(const Rotor& dq_in) {
    Rotor dq = dq_in;
    if (dq.w < 0.0f) {
        dq.v = -dq.v;
    }
    float angle = 0.0f;
    vec3 axis = dq.axis(angle);
    return axis * angle;
}

static void syncSegmentFromBodies(
    Sim& sim,
    const double* quat,
    int n_segments) {
    const vec3 ex(1.0f, 0.0f, 0.0f);
    for (int j = 0; j < n_segments; ++j) {
        const vec3 p0 = sim.verts[j].pos;
        const vec3 p1 = sim.verts[j + 1].pos;
        vec3 tangent = p1 - p0;
        const float len = length(tangent);
        if (len < 1e-12f) {
            tangent = ex;
        } else {
            tangent /= len;
            sim.segs[j].l = len;
        }

        const Rotor bodyRot = mujocoQuatToRotor(quat + 4 * j);
        const vec3 bodyTangent = safeNormalize(bodyRot * ex, tangent);
        const Rotor align = Rotor::fromTo(ex, tangent);
        const Rotor bodyFrame = Rotor::fromTo(ex, bodyTangent);
        const Rotor twist = align.inverse() * bodyFrame;
        sim.segs[j].q = normalize(align * twist);
    }
}

RodCosserat2::RodCosserat2(
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
    bool bothweld)
    : n_nodes_(n_nodes),
      n_segments_(n_nodes > 1 ? n_nodes - 1 : 0),
      bothweld_(bothweld),
      segment_length_(segment_length),
      radius_(radius),
      k_stretch_(k_stretch),
      k_bend_(k_bend),
      k_twist_(k_twist),
      k_force_(0.05),
      k_torque_(0.05),
      num_iters_(4),
      sim_(new SimState()) {
    (void)dim_x;
    (void)dim_q;
    buildChain(rest_x, rest_quat);
}

RodCosserat2::~RodCosserat2() {
    destroySim();
}

void RodCosserat2::destroySim() {
    delete sim_;
    sim_ = nullptr;
}

void RodCosserat2::setMaterial(double k_stretch, double k_bend, double k_twist) {
    k_stretch_ = k_stretch;
    k_bend_ = k_bend;
    k_twist_ = k_twist;
}

void RodCosserat2::setForceGain(double k_force) {
    k_force_ = k_force;
}

void RodCosserat2::setTorqueGain(double k_torque) {
    k_torque_ = k_torque;
}

void RodCosserat2::setNumIterations(int num_iters) {
    num_iters_ = num_iters < 1 ? 1 : num_iters;
}

void RodCosserat2::buildChain(const double* rest_x, const double* rest_quat) {
    Sim& sim = sim_->sim;
    sim.segs.clear();
    sim.verts.clear();
    sim.restAngles.clear();
    sim.vels.clear();
    sim.dxs.clear();
    sim.vertSegIndices.clear();
    sim.vertSegIds.clear();
    sim.segAngleIndices.clear();
    sim.segAngleIds.clear();

    sim_->rest_quat.assign(4 * n_nodes_, 0.0);
    for (int i = 0; i < n_nodes_; ++i) {
        sim.addVertex(vec3FromArray(rest_x + 3 * i));
        for (int k = 0; k < 4; ++k) {
            sim_->rest_quat[4 * i + k] = rest_quat[4 * i + k];
        }
    }

    for (int j = 0; j < n_segments_; ++j) {
        sim.addSegment(ivec2(j, j + 1), static_cast<float>(k_stretch_), 1.0f);
    }

    for (int j = 0; j < n_segments_ - 1; ++j) {
        sim.addRestAngle(ivec2(j, j + 1), static_cast<float>(k_bend_));
    }

    sim.init();
    syncSegmentFromBodies(sim, rest_quat, n_segments_);

    if (bothweld_ && n_nodes_ > 0) {
        sim.pinVertex(0);
        sim.pinVertex(n_nodes_ - 1);
    }

    for (size_t i = 0; i < sim.verts.size(); ++i) {
        sim.verts[i].invMass = 1.0f;
    }
    if (bothweld_ && n_nodes_ > 0) {
        sim.verts[0].invMass = 0.0f;
        sim.verts[n_nodes_ - 1].invMass = 0.0f;
    }
}

void RodCosserat2::reinitRest(
    int dim_x,
    const double* rest_x,
    int dim_q,
    const double* rest_quat) {
    (void)dim_x;
    (void)dim_q;
    buildChain(rest_x, rest_quat);
}

void RodCosserat2::computeWrenches(
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
    (void)dim_i;
    (void)dim_f;
    (void)dim_t;
    (void)dim_m;

    const int n = n_nodes_;
    std::memset(force_out, 0, sizeof(double) * 3 * n);
    std::memset(torque_out, 0, sizeof(double) * 3 * n);

    if (n < 2 || n_segments_ < 1 || dt <= 0.0) {
        return;
    }

    Sim& base = sim_->sim;
    Sim shadow;
    shadow.copyFrom(base);

    std::vector<vec3> x0(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        x0[static_cast<size_t>(i)] = vec3FromArray(x + 3 * i);
        shadow.verts[static_cast<size_t>(i)].pos = x0[static_cast<size_t>(i)];
        if (inv_mass[i] <= 0.0) {
            shadow.verts[static_cast<size_t>(i)].invMass = 0.0f;
        } else {
            shadow.verts[static_cast<size_t>(i)].invMass = 1.0f;
        }
    }

    syncSegmentFromBodies(shadow, quat, n_segments_);

    shadow.dxs.assign(n, vec3(0));
    shadow.vels.assign(n, vec3(0));
    shadow.meta.h = static_cast<float>(dt);

    for (int iter = 0; iter < num_iters_; ++iter) {
        shadow.iterateStableCosserat();
        for (int i = 0; i < n; ++i) {
            shadow.verts[static_cast<size_t>(i)].pos += shadow.dxs[static_cast<size_t>(i)];
            shadow.dxs[static_cast<size_t>(i)] = vec3(0);
        }
    }

    const double inv_dt = 1.0 / dt;
    const double kf = k_force_;
    const double kt = k_torque_;

    for (int i = 0; i < n; ++i) {
        if (inv_mass[i] <= 0.0) {
            continue;
        }
        const vec3 dx = shadow.verts[static_cast<size_t>(i)].pos - x0[static_cast<size_t>(i)];
        force_out[3 * i + 0] = kf * static_cast<double>(dx.x) * inv_dt;
        force_out[3 * i + 1] = kf * static_cast<double>(dx.y) * inv_dt;
        force_out[3 * i + 2] = kf * static_cast<double>(dx.z) * inv_dt;
    }

    const vec3 fallback_tangent(1.0f, 0.0f, 0.0f);
    Sim current;
    current.copyFrom(base);
    for (int i = 0; i < n; ++i) {
        current.verts[static_cast<size_t>(i)].pos = x0[static_cast<size_t>(i)];
    }
    syncSegmentFromBodies(current, quat, n_segments_);

    for (int j = 0; j < n_segments_; ++j) {
        const int child = j + 1;
        if (inv_mass[child] <= 0.0) {
            continue;
        }

        Rotor delta = shadow.segs[static_cast<size_t>(j)].q *
            current.segs[static_cast<size_t>(j)].q.inverse();
        const vec3 rotvec = rotorToRotvec(delta);
        const vec3 tangent = safeNormalize(
            shadow.segs[static_cast<size_t>(j)].q * vec3(1, 0, 0),
            fallback_tangent);

        const float twist_scalar = dot(rotvec, tangent);
        const vec3 twist_vec = twist_scalar * tangent;
        const vec3 bend_vec = rotvec - twist_vec;
        const vec3 weighted = static_cast<float>(k_bend_) * bend_vec +
            static_cast<float>(k_twist_) * twist_vec;

        const Rotor q_prev = mujocoQuatToRotor(quat + 4 * j);
        const Rotor q_cur = mujocoQuatToRotor(quat + 4 * child);
        const Rotor q_rel = q_prev.inverse() * q_cur;
        const vec3 tau_child = q_rel.inverse() * weighted;

        torque_out[3 * child + 0] += kt * static_cast<double>(tau_child.x) * inv_dt;
        torque_out[3 * child + 1] += kt * static_cast<double>(tau_child.y) * inv_dt;
        torque_out[3 * child + 2] += kt * static_cast<double>(tau_child.z) * inv_dt;
    }

    (void)inv_inertia_w;
}
