#include "RodCosserat5.h"

#include <cmath>
#include "Eigen/Dense"
#include "Eigen/Geometry"

RodCosserat5::RodCosserat5(
    int dim_np,
    double* node_pos,
    int dim_bf0,
    double* bf0sim,
    const double theta_n,
    const double overall_rot,
    const double a_bar,
    const double b_bar,
    const double radius)
{
    SegEdges e1;
    Vecnodes x1;

    d_vec = 0;
    nv = static_cast<int>(dim_np / 3) - 2 - d_vec * 2;
    bigL_bar = 0.;
    alpha_bar = a_bar;
    beta_bar = b_bar;
    radius_ = radius;
    updateStiffnessFromMaterial();

    Eigen::MatrixXd dist1(nv + 2, 3);
    for (int i = 0; i < (nv + 1); i++) {
        edges.push_back(e1);
        nodes.push_back(x1);
        distmat.push_back(dist1);
    }
    nodes.push_back(x1);
    distmat.push_back(dist1);

    edges[nv].theta = overall_rot;
    p_thetan = fmod(edges[nv].theta, (2. * M_PI));
    if (p_thetan > M_PI) { p_thetan -= 2 * M_PI; }

    initVars(dim_np, node_pos, dim_bf0, bf0sim);
    captureRestCurvature();
    (void)theta_n;
}

void RodCosserat5::updateStiffnessFromMaterial()
{
    // JTill2017 K = diag(EI, EI, GI); alpha_bar/beta_bar map to bending/twist scales.
    // Use alpha_bar/beta_bar directly (same tuning as adapt/dlo controllers).
    k_bend_ = alpha_bar;
    k_twist_ = beta_bar;
}

void RodCosserat5::initVars(
    int dim_np,
    double* node_pos,
    int dim_bf0,
    double* bf0sim)
{
    Eigen::Matrix3d init_mat3d = Eigen::Matrix3d::Zero();
    for (int i = 0; i < (nv + 2); i++) {
        for (int j = 0; j < 3; j++) {
            nodes[i].nabkb.push_back(init_mat3d);
        }
        nodes[i].u.setZero();
        nodes[i].u_star.setZero();
    }

    bf0mat << bf0sim[0], bf0sim[1], bf0sim[2],
        bf0sim[3], bf0sim[4], bf0sim[5],
        bf0sim[6], bf0sim[7], bf0sim[8];

    update_XVecs(node_pos);
    updateX2E();
    updateE2K();
    updateE2Kb();
    transfBF(bf0mat);
    updateMaterialCurvature();
}

bool RodCosserat5::updateVars(
    int dim_np,
    double* node_pos,
    int dim_bf0,
    double* bf0sim,
    int dim_bfe,
    double* bfesim)
{
    (void)dim_np;
    (void)dim_bf0;

    bf0mat << bf0sim[0], bf0sim[1], bf0sim[2],
        bf0sim[3], bf0sim[4], bf0sim[5],
        bf0sim[6], bf0sim[7], bf0sim[8];

    bool bf_align = true;

    update_XVecs(node_pos);
    updateX2E();
    updateE2K();
    updateE2Kb();
    bf_align = transfBF(bf0mat);
    updateMaterialCurvature();

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            bfesim[3 * i + j] = edges[nv].bf(i, j);
        }
    }
    (void)dim_bfe;
    return bf_align;
}

void RodCosserat5::update_XVecs(const double* node_pos)
{
    for (int i = 0; i < nv + 2; i++) {
        nodes[i].pos << node_pos[3 * i], node_pos[3 * i + 1], node_pos[3 * i + 2];
    }
}

void RodCosserat5::updateX2E()
{
    bigL_bar = 0;
    edges[0].e = nodes[1].pos - nodes[0].pos;
    edges[0].e_bar = edges[0].e.norm();
    for (int i = 1; i < nv + 1; i++) {
        edges[i].e = nodes[i + 1].pos - nodes[i].pos;
        edges[i].e_bar = edges[i].e.norm();
        edges[i].l_bar = edges[i].e_bar + edges[i - 1].e_bar;
        bigL_bar += edges[i].l_bar;
    }
    bigL_bar /= 2.;
}

void RodCosserat5::updateE2K()
{
    nodes[0].phi_i = M_PI;
    nodes[nv + 1].phi_i = M_PI;
    nodes[0].k = 0.0;
    nodes[nv + 1].k = 0.0;
    for (int i = 1; i < nv + 1; i++) {
        nodes[i].phi_i = Cosserat5Utils::calculateAngleBetween(edges[i - 1].e, edges[i].e);
        nodes[i].k = 2. * tan(nodes[i].phi_i / 2.);
    }
}

void RodCosserat5::updateE2Kb()
{
    nodes[0].kb << 0., 0., 0.;
    nodes[nv + 1].kb << 0., 0., 0.;
    for (int i = 1; i < nv + 1; i++) {
        nodes[i].kb = (
            2. * edges[i - 1].e.cross(edges[i].e)
            / (
                edges[i - 1].e_bar * edges[i].e_bar
                + edges[i - 1].e.dot(edges[i].e)
            )
        );
    }
}

bool RodCosserat5::transfBF(const Eigen::Matrix3d& bf_0)
{
    bool bf_align = true;

    edges[0].bf = bf_0;

    for (int i = 1; i < nv + 1; i++) {
        edges[i].bf.row(0) = edges[i].e / edges[i].e.norm();
        if (nodes[i].kb.norm() == 0) {
            edges[i].bf.row(1) = edges[i - 1].bf.row(1);
        } else {
            edges[i].bf.row(1) = Cosserat5Utils::rotateVector3(
                edges[i - 1].bf.row(1),
                nodes[i].kb / nodes[i].kb.norm(),
                nodes[i].phi_i
            );
            if (std::abs(edges[i].bf.row(1).dot(edges[i].bf.row(0))) > 1e-1) {
                bf_align = false;
            }
        }
        edges[i].bf.row(2) = edges[i].bf.row(0).cross(edges[i].bf.row(1));
    }
    return bf_align;
}

Eigen::Vector3d RodCosserat5::materialCurvatureAtNode(int i) const
{
    Eigen::Vector3d u_local;
    u_local.setZero();
    if (i < 1 || i > nv) {
        return u_local;
    }

    const Eigen::Matrix3d& R = edges[i].bf;
    u_local(0) = nodes[i].kb.dot(R.row(1));
    u_local(1) = nodes[i].kb.dot(R.row(2));

    if (i >= 1 && edges[i - 1].e_bar > 1e-12) {
        u_local(2) = (edges[i].theta - edges[i - 1].theta) / edges[i - 1].e_bar;
    }
    return u_local;
}

void RodCosserat5::updateMaterialCurvature()
{
    for (int i = 0; i < nv + 2; i++) {
        nodes[i].u = materialCurvatureAtNode(i);
    }
}

void RodCosserat5::captureRestCurvature()
{
    for (int i = 0; i < nv + 2; i++) {
        nodes[i].u_star = nodes[i].u;
    }
    for (int i = 0; i < nv + 1; i++) {
        edges[i].theta_star = edges[i].theta;
    }
    theta_star_total_ = edges[nv].theta - edges[0].theta;
}

Eigen::Vector3d RodCosserat5::kbEffectiveAtNode(int i) const
{
    const Eigen::Matrix3d& R = edges[i].bf;
    const Eigen::Vector3d du = nodes[i].u - nodes[i].u_star;
    return du(0) * R.row(1).transpose() + du(1) * R.row(2).transpose();
}

void RodCosserat5::updateThetaN(double theta_n)
{
    double diff_theta = theta_n - p_thetan;

    if (abs(diff_theta) < (M_PI)) {
        edges[nv].theta += diff_theta;
    } else if (diff_theta > 0.) {
        edges[nv].theta += diff_theta - (2 * M_PI);
    } else {
        edges[nv].theta += diff_theta + (2 * M_PI);
    }
    p_thetan = theta_n;
}

double RodCosserat5::updateTheta(const double theta_n)
{
    updateThetaN(theta_n);

    const double d_theta = (edges[nv].theta - edges[0].theta) / nv;
    for (int i = 0; i < (nv + 1); i++) {
        edges[i].theta = d_theta * i;
    }
    updateMaterialCurvature();
    return edges[nv].theta;
}

void RodCosserat5::resetTheta(const double theta_n, const double overall_rot)
{
    p_thetan = theta_n;
    edges[nv].theta = overall_rot;
}

void RodCosserat5::changeAlphaBeta(const double a_bar, const double b_bar)
{
    alpha_bar = a_bar;
    beta_bar = b_bar;
    updateStiffnessFromMaterial();
}

void RodCosserat5::calculateKirchhoffForces_sub(const int start_i, const int end_i)
{
    for (int i = start_i; i < end_i; i++) {
        nodes[i].nabkb[0] = (
            (
                2 * Cosserat5Utils::createSkewSym(edges[i].e)
                + (nodes[i].kb * edges[i].e.transpose())
            )
            / (
                edges[i - 1].e_bar * (edges[i].e_bar)
                + edges[i - 1].e.dot(edges[i].e)
            )
        );
        nodes[i].nabkb[2] = (
            (
                2 * Cosserat5Utils::createSkewSym(edges[i - 1].e)
                - (nodes[i].kb * edges[i - 1].e.transpose())
            )
            / (
                edges[i - 1].e_bar * (edges[i].e_bar)
                + edges[i - 1].e.dot(edges[i].e)
            )
        );
        nodes[i].nabkb[1] = (-(nodes[i].nabkb[0] + nodes[i].nabkb[2]));
        nodes[i].nabpsi.row(0) = nodes[i].kb / (2 * edges[i - 1].e_bar);
        nodes[i].nabpsi.row(2) = -nodes[i].kb / (2 * edges[i].e_bar);
        nodes[i].nabpsi.row(1) = -(nodes[i].nabpsi.row(0) + nodes[i].nabpsi.row(2));

        const Eigen::Vector3d kb_eff = kbEffectiveAtNode(i);

        for (int j = (i - 1); j < (i + 2); j++) {
            // Gradient of 0.5 * k_bend * |u_bend - u*_bend|^2 via kb chain rule.
            nodes[j].force += -(
                2. * k_bend_
                * nodes[i].nabkb[j - i + 1].transpose() * kb_eff
            ) / edges[i].l_bar;
        }
    }
}

void RodCosserat5::calculateCenterlineF2(int dim_nf, double* node_force)
{
    (void)dim_nf;
    for (int i = 0; i < (nv + 2); i++) {
        nodes[i].force << 0., 0., 0.;
    }

    calculateKirchhoffForces_sub(1, nv + 1);

    for (int i = 0; i < (nv + 2); i++) {
        for (int j = 0; j < 3; j++) {
            node_force[3 * i + j] = nodes[i].force(j);
        }
    }
}

void RodCosserat5::calculateCenterlineTorq(
    int dim_nt, double* node_torq,
    int dim_nq, double* node_quat,
    int excl_jnts)
{
    (void)dim_nt;
    (void)dim_nq;
    Eigen::Vector3d dist_diff;

    excl_joints = excl_jnts;

    for (int i = 0; i < (nv + 2); i++) {
        nodes[i].force << 0., 0., 0.;
        nodes[i].torq << 0., 0., 0.;
        nodes[i].quat << node_quat[4 * i], node_quat[4 * i + 1], node_quat[4 * i + 2], node_quat[4 * i + 3];
        if (i > 0) {
            dist_diff = nodes[i].pos - nodes[i - 1].pos;
            distmat[i].row(i - 1) = dist_diff;
        }
        distmat[i].row(i) << 0., 0., 0.;
    }
    for (int i = 2; i < (nv + 2); i++) {
        distmat[i].block(0, 0, i - 1, 3) =
            distmat[i - 1].block(0, 0, i - 1, 3).array().rowwise()
            + distmat[i].row(i - 1).array();
        for (int j = 0; j < (i); j++) {
            distmat[j].row(i) = distmat[i].row(j);
        }
    }
    distmat[0].row(1) = distmat[1].row(0);

    calculateKirchhoffForces_sub(1, nv + 1);
    calculateF2LocalTorq();

    for (int i = 0; i < (nv + 2); i++) {
        for (int j = 0; j < 3; j++) {
            node_torq[3 * i + j] = nodes[i].torq(j);
        }
    }
}

void RodCosserat5::calculateF2LocalTorq()
{
    Eigen::MatrixXd torqvec(nv + 2, 3);
    Eigen::MatrixXd torqvec_indiv(nv + 2, 3);
    for (int i = 0; i < (nv + 2); i++) {
        torqvec.row(i) << 0., 0., 0.;
    }
    for (int i = excl_joints; i < (nv + 2 - excl_joints); i++) {
        torqvec_indiv = distmat[i].array().rowwise().cross(nodes[i].force);
        torqvec += torqvec_indiv;
    }
    torqvec /= 2.0;
    for (int i = 0; i < (nv + 2); i++) {
        nodes[i].torq = Cosserat5Utils::rotVecQuat(
            torqvec.row(i),
            Cosserat5Utils::inverseQuat(nodes[i].quat)
        );
    }
}

double RodCosserat5::calculateBendingEnergy()
{
    double energy_bending = 0.0;
    for (int i = 1; i < (nv + 1); i++) {
        const Eigen::Vector3d du = nodes[i].u - nodes[i].u_star;
        energy_bending += 0.5 * k_bend_ * (du(0) * du(0) + du(1) * du(1)) * edges[i].l_bar;
    }
    return energy_bending;
}

double RodCosserat5::calculateTwistingEnergy()
{
    const double dtheta = (edges[nv].theta - edges[0].theta) - theta_star_total_;
    return 0.5 * k_twist_ * dtheta * dtheta / bigL_bar;
}

double RodCosserat5::calculateEnergy()
{
    return calculateBendingEnergy();
}

void RodCosserat5::initQe_o2m_loc(int dim_qo2m, double* q_o2m)
{
    (void)dim_qo2m;
    qe_o2m_loc.x() = q_o2m[0];
    qe_o2m_loc.y() = q_o2m[1];
    qe_o2m_loc.z() = q_o2m[2];
    qe_o2m_loc.w() = q_o2m[3];
    qe_o2m_loc = qe_o2m_loc.normalized();
}

void RodCosserat5::calculateOf2Mf(
    int dim_mato, double* mat_o,
    int dim_matres, double* mat_res)
{
    (void)dim_mato;
    (void)dim_matres;
    Eigen::Matrix3d mat_mato;
    Eigen::Matrix3d mat_result;
    mat_mato << mat_o[0], mat_o[1], mat_o[2],
        mat_o[3], mat_o[4], mat_o[5],
        mat_o[6], mat_o[7], mat_o[8];

    Eigen::Quaterniond q_mato(mat_mato);
    mat_result = (q_mato * qe_o2m_loc).normalized().toRotationMatrix();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat_res[j + 3 * i] = mat_result(i, j);
        }
    }
}

double RodCosserat5::angBtwn3(
    int dim_v1, double* v1,
    int dim_v2, double* v2,
    int dim_va, double* va)
{
    (void)dim_v1;
    (void)dim_v2;
    (void)dim_va;
    Eigen::Vector3d v1_c(v1[0], v1[1], v1[2]);
    Eigen::Vector3d v2_c(v2[0], v2[1], v2[2]);
    Eigen::Vector3d va_c(va[0], va[1], va[2]);
    double dot_norm_val = v1_c.dot(v2_c) / (v1_c.norm() * v2_c.norm());
    if (dot_norm_val > 1.) { dot_norm_val = 1.; }
    double theta_diff = acos(dot_norm_val);
    if ((v1_c.cross(v2_c)).dot(va_c) < 0) { theta_diff *= -1.; }
    return theta_diff;
}
