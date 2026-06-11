#include "cosserat4.h"
#include "Cosserat4_obj.h"

#include <cmath>
#include <iostream>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Geometry"

RodCosserat4::RodCosserat4(
    int dim_np,
    double *node_pos,
    int dim_bf0,
    double *bf0sim,
    const double theta_n,
    const double overall_rot,
    const double a_bar,
    const double b_bar
)
{
    SegEdges e1;
    Vecnodes x1;

    d_vec = 0;
    nv = (int)(dim_np / 3) - 2 - d_vec * 2;
    bigL_bar = 0.;
    alpha_bar = a_bar;
    beta_bar = b_bar;

    Eigen::MatrixXd dist1(nv+2, 3);
    for (int i = 0; i < (nv+1); i++) {
        edges.push_back(e1);
        nodes.push_back(x1);
        distmat.push_back(dist1);
    }
    nodes.push_back(x1);
    distmat.push_back(dist1);

    j_rot << 0., -1., 1., 0.;

    initEdgeTheta(overall_rot);
    p_thetan = theta_n;
    (void)theta_n;

    initVars(dim_np, node_pos, dim_bf0, bf0sim);
}

void RodCosserat4::initEdgeTheta(const double overall_rot_val)
{
    for (int b = 0; b < (nv+1); b++) {
        edges[b].theta = overall_rot_val / nv * b;
        edges[b].p_thetaloc = std::fmod(edges[b].theta, (2. * M_PI));
        if (edges[b].p_thetaloc > M_PI) {
            edges[b].p_thetaloc -= 2. * M_PI;
        }
        edges[b].theta_displace = edges[b].p_thetaloc;
        edges[b].qe_o2m_loc = Eigen::Quaterniond::Identity();
    }
    overall_rot = edges[nv].theta;
}

void RodCosserat4::initVars(
    int dim_np,
    double *node_pos,
    int dim_bf0,
    double *bf0sim
)
{
    Eigen::Matrix3d init_mat3d;
    init_mat3d << 0., 0., 0.,
        0., 0., 0.,
        0., 0., 0.;
    for (int i = 0; i < (nv+2); i++) {
        for (int j = 0; j < 3; j++) {
            nodes[i].nabkb.push_back(init_mat3d);
        }
    }

    bf0mat << bf0sim[0], bf0sim[1], bf0sim[2],
        bf0sim[3], bf0sim[4], bf0sim[5],
        bf0sim[6], bf0sim[7], bf0sim[8];

    update_XVecs(node_pos);
    updateX2E();
    updateE2K();
    updateE2Kb();
    transfBF(bf0mat);
}

bool RodCosserat4::updateVars(
    int dim_np,
    double *node_pos,
    int dim_bf0,
    double *bf0sim,
    int dim_bfe,
    double *bfesim
)
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

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            bfesim[3*i+j] = edges[nv].bf(i,j);
        }
    }
    (void)dim_bfe;
    return bf_align;
}

void RodCosserat4::update_XVecs(const double *node_pos)
{
    for (int i = 0; i < nv+2; i++) {
        nodes[i].pos << node_pos[3*i], node_pos[3*i+1], node_pos[3*i+2];
    }
}

void RodCosserat4::updateX2E()
{
    bigL_bar = 0;
    edges[0].e = nodes[1].pos - nodes[0].pos;
    edges[0].e_bar = edges[0].e.norm();
    for (int i = 1; i < nv+1; i++) {
        edges[i].e = nodes[i+1].pos - nodes[i].pos;
        edges[i].e_bar = edges[i].e.norm();
        edges[i].l_bar = edges[i].e_bar + edges[i-1].e_bar;
        bigL_bar += edges[i].l_bar;
    }
    bigL_bar /= 2.;
}

void RodCosserat4::updateE2K()
{
    nodes[0].phi_i = M_PI;
    nodes[nv+1].phi_i = M_PI;
    nodes[0].k = 0.0;
    nodes[nv+1].k = 0.0;
    for (int i = 1; i < nv+1; i++) {
        nodes[i].phi_i = Cosserat4Utils::calculateAngleBetween(edges[i-1].e, edges[i].e);
        nodes[i].k = 2. * tan(nodes[i].phi_i / 2.);
    }
}

void RodCosserat4::updateE2Kb()
{
    nodes[0].kb << 0., 0., 0.;
    nodes[nv+1].kb << 0., 0., 0.;
    for (int i = 1; i < nv+1; i++) {
        nodes[i].kb = (
            2. * edges[i-1].e.cross(edges[i].e)
            / (
                edges[i-1].e_bar * edges[i].e_bar
                + edges[i-1].e.dot(edges[i].e)
            )
        );
    }
}

bool RodCosserat4::transfBF(const Eigen::Matrix3d &bf_0)
{
    bool bf_align = true;

    edges[0].bf = bf_0;

    for (int i = 1; i < nv+1; i++) {
        edges[i].bf.row(0) = edges[i].e / edges[i].e.norm();
        if (nodes[i].kb.norm() == 0) {
            edges[i].bf.row(1) = edges[i-1].bf.row(1);
        } else {
            edges[i].bf.row(1) = Cosserat4Utils::rotateVector3(
                edges[i-1].bf.row(1),
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

double RodCosserat4::getThetaLoc(int idx, const Eigen::Vector4d &body_quat)
{
    Eigen::Quaterniond q_o(body_quat(0), body_quat(1), body_quat(2), body_quat(3));
    q_o.normalize();

    Eigen::Matrix3d mat_bn = edges[idx].bf;
    Eigen::Matrix3d mat_mn = (q_o * edges[idx].qe_o2m_loc).normalized().toRotationMatrix();

    Eigen::Vector3d v1 = mat_bn.col(1);
    Eigen::Vector3d v2 = mat_mn.col(1);
    Eigen::Vector3d va = mat_bn.col(0);

    double theta_diff = Cosserat4Utils::calculateAngleBetween2(v1, v2, va)
        + edges[idx].theta_displace;
    if (theta_diff > M_PI) {
        theta_diff -= 2. * M_PI;
    }
    return theta_diff;
}

void RodCosserat4::updateThetaLoc(double theta_loc, int idx)
{
    double diff_theta = theta_loc - edges[idx].p_thetaloc;

    if (std::abs(diff_theta) < M_PI) {
        edges[idx].theta += diff_theta;
    } else if (diff_theta > 0.) {
        edges[idx].theta += diff_theta - (2. * M_PI);
    } else {
        edges[idx].theta += diff_theta + (2. * M_PI);
    }
    edges[idx].p_thetaloc = theta_loc;

    std::cout << "updateThetaLoc idx=" << idx
              << " theta_loc=" << theta_loc
              << " diff_theta=" << diff_theta
              << " theta=" << edges[idx].theta
              << " | p_thetaloc:";
    for (int k = 0; k <= nv; k++) {
        std::cout << " [" << k << "]=" << edges[k].p_thetaloc;
    }
    std::cout << std::endl;
}

void RodCosserat4::initO2MLoc(int idx, const Eigen::Vector4d &body_quat)
{
    Eigen::Quaterniond q_o(body_quat(0), body_quat(1), body_quat(2), body_quat(3));
    q_o.normalize();

    Eigen::Quaterniond q_b(edges[idx].bf);
    q_b.normalize();

    Eigen::Quaterniond q_error = q_b * q_o.inverse();
    q_error.normalize();

    edges[idx].qe_o2m_loc = q_o.inverse() * q_error * q_o;
    edges[idx].qe_o2m_loc.normalize();
}

void RodCosserat4::initO2MLocAll(int dim_nq, double *node_quat)
{
    (void)dim_nq;
    for (int i = 0; i < (nv+1); i++) {
        Eigen::Vector4d quat(
            node_quat[4*i],
            node_quat[4*i+1],
            node_quat[4*i+2],
            node_quat[4*i+3]
        );
        initO2MLoc(i, quat);
    }
}

void RodCosserat4::updateThetasFullDyn(int dim_nq, double *node_quat)
{
    (void)dim_nq;
    for (int bwi = 1; bwi < nv; bwi++) {
        Eigen::Vector4d quat(
            node_quat[4*bwi],
            node_quat[4*bwi+1],
            node_quat[4*bwi+2],
            node_quat[4*bwi+3]
        );
        updateThetaLoc(getThetaLoc(bwi, quat), bwi);
    }
    overall_rot = edges[nv].theta;
}

void RodCosserat4::resetTheta(const double theta_n, const double overall_rot_val)
{
    p_thetan = theta_n;
    initEdgeTheta(overall_rot_val);
}

void RodCosserat4::changeAlphaBeta(const double a_bar, const double b_bar)
{
    alpha_bar = a_bar;
    beta_bar = b_bar;
}

void RodCosserat4::calculateNabKbandNabPsi_sub2(const int start_i, const int end_i)
{
    for (int i = start_i; i < end_i; i++) {
        nodes[i].nabkb[0] = (
            (
                2 * Cosserat4Utils::createSkewSym(edges[i].e)
                + (nodes[i].kb * edges[i].e.transpose())
            )
            / (
                edges[i-1].e_bar * (edges[i].e_bar)
                + edges[i-1].e.dot(edges[i].e)
            )
        );
        nodes[i].nabkb[2] = (
            (
                2 * Cosserat4Utils::createSkewSym(edges[i-1].e)
                - (nodes[i].kb * edges[i-1].e.transpose())
            )
            / (
                edges[i-1].e_bar * (edges[i].e_bar)
                + edges[i-1].e.dot(edges[i].e)
            )
        );
        nodes[i].nabkb[1] = (- (nodes[i].nabkb[0] + nodes[i].nabkb[2]));

        for (int j = (i-1); j < (i+2); j++) {
            nodes[j].force += - (
                2. * alpha_bar
                * nodes[i].nabkb[j-i+1].transpose() * nodes[i].kb
            ) / edges[i].l_bar;
        }
    }
}

void RodCosserat4::addThetaTwistTorq()
{
    const Eigen::Vector3d fallback_tangent(1., 0., 0.);

    for (int i = 1; i < (nv + 1); i++) {
        const double seg_strain = edges[i].theta - edges[i - 1].theta;

        Eigen::Vector3d tangent = nodes[i].pos - nodes[i - 1].pos;
        const double t_norm = tangent.norm();
        if (t_norm < 1e-12) {
            tangent = fallback_tangent;
        } else {
            tangent /= t_norm;
        }

        const Eigen::Vector3d tau_parent = (
            2. * beta_bar * seg_strain / edges[i].l_bar
        ) * tangent;

        nodes[i].torq += tau_parent * 0.01;
    }
}

void RodCosserat4::calculateCenterlineF2(int dim_nf, double *node_force)
{
    (void)dim_nf;
    for (int i = 0; i < (nv+2); i++) {
        nodes[i].force << 0., 0., 0.;
    }

    calculateNabKbandNabPsi_sub2(1, nv+1);

    for (int i = 0; i < (nv+2); i++) {
        for (int j = 0; j < 3; j++) {
            node_force[3*i+j] = nodes[i].force(j);
        }
    }
}

void RodCosserat4::calculateCenterlineTorq(
    int dim_nt, double *node_torq,
    int dim_nq, double *node_quat,
    int excl_jnts
)
{
    (void)dim_nt;
    Eigen::Vector3d dist_diff;

    excl_joints = excl_jnts;

    updateThetasFullDyn(dim_nq, node_quat);

    for (int i = 0; i < (nv+2); i++) {
        nodes[i].force << 0., 0., 0.;
        nodes[i].torq << 0., 0., 0.;
        nodes[i].quat << node_quat[4*i], node_quat[4*i+1], node_quat[4*i+2], node_quat[4*i+3];
        if (i > 0) {
            dist_diff = nodes[i].pos - nodes[i-1].pos;
            distmat[i].row(i-1) = dist_diff;
        }
        distmat[i].row(i) << 0., 0., 0.;
    }
    for (int i = 2; i < (nv+2); i++) {
        distmat[i].block(0,0,i-1,3) = \
            distmat[i-1].block(0,0,i-1,3).array().rowwise() \
            + distmat[i].row(i-1).array();
        for (int j = 0; j < (i); j++) {
            distmat[j].row(i) = distmat[i].row(j);
        }
    }
    distmat[0].row(1) = distmat[1].row(0);

    calculateNabKbandNabPsi_sub2(1, nv+1);

    calculateF2LocalTorq();
    addThetaTwistTorq();

    for (int i = 0; i < (nv+2); i++) {
        for (int j = 0; j < 3; j++) {
            node_torq[3*i+j] = nodes[i].torq(j);
        }
    }
}

void RodCosserat4::calculateF2LocalTorq()
{
    Eigen::MatrixXd torqvec(nv+2,3);
    Eigen::MatrixXd torqvec_indiv(nv+2,3);
    for (int i = 0; i < (nv+2); i++) {
        torqvec.row(i) << 0., 0., 0.;
    }
    for (int i = excl_joints; i < (nv+2-excl_joints); i++) {
        torqvec_indiv = distmat[i].array().rowwise().cross(nodes[i].force);
        torqvec += torqvec_indiv;
    }
    torqvec /= 2.0;
    for (int i = 0; i < (nv+2); i++) {
        nodes[i].torq = Cosserat4Utils::rotVecQuat(
            torqvec.row(i),
            Cosserat4Utils::inverseQuat(nodes[i].quat)
        );
    }
}

double RodCosserat4::calculateEnergy()
{
    double energy_total = 0.0;
    energy_total += calculateBendingEnergy();
    energy_total += calculateTwistingEnergy();
    return energy_total;
}

double RodCosserat4::calculateBendingEnergy()
{
    double energy_bending = 0.0;
    for (int i = 1; i < (nv+1); i++) {
        energy_bending += alpha_bar * nodes[i].kb.dot(nodes[i].kb)
            / edges[i].l_bar;
    }
    return energy_bending;
}

double RodCosserat4::calculateTwistingEnergy()
{
    double energy_twisting = 0.0;
    for (int i = 1; i < (nv + 1); i++) {
        const double dtheta = edges[i].theta - edges[i - 1].theta;
        energy_twisting += beta_bar * dtheta * dtheta / edges[i].l_bar;
    }
    return energy_twisting;
}

void RodCosserat4::initQe_o2m_loc(int dim_qo2m, double *q_o2m)
{
    (void)dim_qo2m;
    qe_o2m_loc.x() = q_o2m[0];
    qe_o2m_loc.y() = q_o2m[1];
    qe_o2m_loc.z() = q_o2m[2];
    qe_o2m_loc.w() = q_o2m[3];
    qe_o2m_loc = qe_o2m_loc.normalized();
    edges[0].qe_o2m_loc = qe_o2m_loc;
}

void RodCosserat4::calculateOf2Mf(
    int dim_mato, double *mat_o,
    int dim_matres, double *mat_res
)
{
    calculateOf2MfAtEdge(0, dim_mato, mat_o, dim_matres, mat_res);
}

void RodCosserat4::calculateOf2MfAtEdge(
    int edge_idx,
    int dim_mato, double *mat_o,
    int dim_matres, double *mat_res
)
{
    (void)dim_mato;
    (void)dim_matres;
    Eigen::Matrix3d mat_mato;
    Eigen::Matrix3d mat_result;
    mat_mato << mat_o[0], mat_o[1], mat_o[2],
        mat_o[3], mat_o[4], mat_o[5],
        mat_o[6], mat_o[7], mat_o[8];

    Eigen::Quaterniond q_mato(mat_mato);
    mat_result = (q_mato * edges[edge_idx].qe_o2m_loc).normalized().toRotationMatrix();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat_res[j+3*i] = mat_result(i,j);
        }
    }
}

double RodCosserat4::angBtwn3(
    int dim_v1, double *v1,
    int dim_v2, double *v2,
    int dim_va, double *va
)
{
    (void)dim_v1;
    (void)dim_v2;
    (void)dim_va;
    Eigen::Vector3d v1_c(v1[0], v1[1], v1[2]);
    Eigen::Vector3d v2_c(v2[0], v2[1], v2[2]);
    Eigen::Vector3d va_c(va[0], va[1], va[2]);
    double theta_diff, dot_norm_val;
    dot_norm_val = v1_c.dot(v2_c)/(v1_c.norm()*v2_c.norm());
    if (dot_norm_val > 1.) {dot_norm_val = 1.;}
    theta_diff = acos(dot_norm_val);
    if ((v1_c.cross(v2_c)).dot(va_c) < 0) {theta_diff *= -1.;}
    return theta_diff;
}
