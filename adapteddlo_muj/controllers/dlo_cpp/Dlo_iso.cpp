#include "Dlo_iso.h"
#include "Dlo_obj.h"

#include <iostream>
// #include <cmath>
// #include <vector>
// #include <chrono>
// #include <thread>
// #include <mutex>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Geometry"

/*
note:
*A* Nodes and torques are defined in the same manner as in MuJoCo.

*B* Distance matrix defined:
    final distmat[a].row(b) = distmat[b].row(a) 
    is the distance from a to b (b-a)
    such that a < b,
    JUST USE negative to get a to b!
    this distance.cross(force) will give the torque direction in *A*.
*/

// std::mutex mtx;

DLO_iso::DLO_iso(
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

    // init variables
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
    
    edges[nv].theta = overall_rot;
    p_thetan = fmod(edges[nv].theta, (2. * M_PI));
    if (p_thetan > M_PI) {p_thetan -= 2 * M_PI;}

    initVars(dim_np, node_pos, dim_bf0, bf0sim);

}

void DLO_iso::initVars(
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
    // initBF(bf0sim);
    transfBF(bf0mat);

    // if abs(p_thetan - ptn) > 1e-6:
    //     raise Exception("Overall_rot and defined frame different")
    // initThreads();
}

bool DLO_iso::updateVars(
    int dim_np,
    double *node_pos,
    int dim_bf0,
    double *bf0sim,
    int dim_bfe,
    double *bfesim
)
{
    bf0mat << bf0sim[0], bf0sim[1], bf0sim[2],
        bf0sim[3], bf0sim[4], bf0sim[5],
        bf0sim[6], bf0sim[7], bf0sim[8];

    bool bf_align;
    bf_align = true;
    
    update_XVecs(node_pos);
    updateX2E();
    updateE2K();
    updateE2Kb();
    // initBF(bf0sim);
    bf_align = transfBF(bf0mat);
    // updateTheta(theta_n);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            bfesim[3*i+j] = edges[nv].bf(i,j);
        }
    }
    return bf_align;
}

void DLO_iso::update_XVecs(const double *node_pos)
{
    for (int i = 0; i < nv+2; i++) {
        nodes[i].pos << node_pos[3*i], node_pos[3*i+1], node_pos[3*i+2];
    }
}

void DLO_iso::updateX2E()
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

void DLO_iso::updateE2K()
{
    nodes[0].phi_i = M_PI;
    nodes[nv+1].phi_i = M_PI;
    // node[0 and nv+1].k should be infinite but will leave as 0 as value is not used
    nodes[0].k = 0.0;
    nodes[nv+1].k = 0.0;
    for (int i = 1; i < nv+1; i++) {
        nodes[i].phi_i = DloUtils::calculateAngleBetween(edges[i-1].e, edges[i].e);
        nodes[i].k = 2. * tan(nodes[i].phi_i / 2.);
    }
}

void DLO_iso::updateE2Kb()
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

bool DLO_iso::transfBF(const Eigen::Matrix3d &bf_0)
{
    bool bf_align;
    bf_align = true;

    edges[0].bf = bf_0;
    
    for (int i = 1; i < nv+1; i++) {
        edges[i].bf.row(0) = edges[i].e / edges[i].e.norm();
        if (nodes[i].kb.norm() == 0) {edges[i].bf.row(1) = edges[i-1].bf.row(1);}
        else {
            edges[i].bf.row(1) = DloUtils::rotateVector3(
                edges[i-1].bf.row(1),
                nodes[i].kb / nodes[i].kb.norm(),
                nodes[i].phi_i
            );
            if (std::abs(edges[i].bf.row(1).dot(edges[i].bf.row(0))) > 1e-1) {
                bf_align = false;
                // throw std::invalid_argument("Bishop transfer error: axis 1 not perpendicular to axis 0.");
            }
        }
        edges[i].bf.row(2) = edges[i].bf.row(0).cross(edges[i].bf.row(1));
    }
    return bf_align;
}

void DLO_iso::updateThetaN_old(const double theta_n)
{
    double diff_theta = theta_n - p_thetan;
    int t_n_whole;

    // acct for 2pi rotation
    if (abs(diff_theta) < (M_PI / 4)) {
        edges[nv].theta += diff_theta;
    }
    else if (diff_theta > 0.) {
        edges[nv].theta += diff_theta - (2 * M_PI);
    }
    else {
        edges[nv].theta += diff_theta + (2 * M_PI);
    }
    // correction step: match self.theta[-1] to theta_n
    t_n_whole = (int) (edges[nv].theta / (2 * M_PI));
    // accounting for int rounding problem
    // (+ve rounds down, -ve rounds up)
    if ((edges[nv].theta < 0.) && (theta_n > 0.)) {
        t_n_whole -= 1;
    }
    if ((edges[nv].theta > 0.) && (theta_n < 0.)) {
        t_n_whole += 1;
    }

    edges[nv].theta = t_n_whole * (2 * M_PI) + theta_n;
    p_thetan = theta_n;
}

void DLO_iso::updateThetaN(const double theta_n)
{
    // TEST THIS. should work with unalgined overall_rot
    double diff_theta = theta_n - p_thetan;
    int t_n_whole;

    // acct for 2pi rotation
    if (abs(diff_theta) < (M_PI)) {
        edges[nv].theta += diff_theta;
    }
    else if (diff_theta > 0.) {
        edges[nv].theta += diff_theta - (2 * M_PI);
    }
    else {
        edges[nv].theta += diff_theta + (2 * M_PI);
    }
    p_thetan = theta_n;
}

double DLO_iso::updateTheta(const double theta_n)
{
    double d_theta;

    updateThetaN(theta_n);
    
    // if (std::isnan(edges[nv].theta)) {std::cout << edges[nv].theta << std::endl;}
    // if (std::isnan(edges[nv].theta)) {std::cout << 't' << std::endl;}
    
    d_theta = (edges[nv].theta - edges[0].theta) / nv;
    for (int i = 0; i < (nv+1); i++) {
        edges[i].theta = d_theta * i;
    }
    return edges[nv].theta;
}

void DLO_iso::resetTheta(const double theta_n, const double overall_rot)
{
    p_thetan = theta_n;
    edges[nv].theta = overall_rot;
}

void DLO_iso::changeAlphaBeta(const double a_bar, const double b_bar)
{
    alpha_bar = a_bar;
    beta_bar = b_bar;
}

void DLO_iso::calculateNabKbandNabPsi_sub2(const int start_i, const int end_i)
{
    for (int i = start_i; i < end_i; i++) {
        nodes[i].nabkb[0] = (
            (
                2 * DloUtils::createSkewSym(edges[i].e)
                + (nodes[i].kb * edges[i].e.transpose())
            )
            / (
                edges[i-1].e_bar * (edges[i].e_bar)
                + edges[i-1].e.dot(edges[i].e)
            )
        );
        nodes[i].nabkb[2] = (
            (
                2 * DloUtils::createSkewSym(edges[i-1].e)
                - (nodes[i].kb * edges[i-1].e.transpose())
            )
            / (
                edges[i-1].e_bar * (edges[i].e_bar)
                + edges[i-1].e.dot(edges[i].e)
            )
        );
        nodes[i].nabkb[1] = (- (nodes[i].nabkb[0] + nodes[i].nabkb[2]));
        nodes[i].nabpsi.row(0) = nodes[i].kb / (2 * edges[i-1].e_bar);
        nodes[i].nabpsi.row(2) = - nodes[i].kb / (2 * edges[i].e_bar);
        nodes[i].nabpsi.row(1) = - (nodes[i].nabpsi.row(0) + nodes[i].nabpsi.row(2));
        
        
        // nodes[i].force << 0., 0., 0.;
        for (int j = (i-1); j < (i+2); j++) {
            // if ((j > nv) || (j<1)) {continue;}
            nodes[j].force += - (
                2. * alpha_bar
                * nodes[i].nabkb[j-i+1].transpose() * nodes[i].kb
            ) / edges[i].l_bar;
            nodes[j].force += (
                beta_bar * (edges[nv].theta - edges[0].theta)
                * nodes[i].nabpsi.row(j-i+1)
            ) / bigL_bar;
            // std::cout << j << std::endl;
            // if (j == 0) {
            //     std::cout << nodes[j].force << std::endl;
            // }
        }
    }
}

void DLO_iso::calculateCenterlineF2(int dim_nf, double *node_force)
{
    for (int i = 0; i < (nv+2); i++) {
        nodes[i].force << 0., 0., 0.;
        // nodes[i].force_sub << 0., 0., 0.;
    }

    calculateNabKbandNabPsi_sub2(1, nv+1);

    // update force on actual simulation
    for (int i = 0; i < (nv+2); i++) {
        // std::cout << ' ' << std::endl;
        // std::cout << i << std::endl;
        for (int j = 0; j < 3; j++) {
            node_force[3*i+j] = nodes[i].force(j);
            // std::cout << nodes[i].force(j) << std::endl;
        }
    }
}

void DLO_iso::calculateCenterlineTorq(
    int dim_nt, double *node_torq,
    // int dim_ntg, double *node_torq_global,
    int dim_nq, double *node_quat,
    int excl_jnts
)
{
    // <std::vector <Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > adj_dist;
    Eigen::Vector3d dist_diff;

    excl_joints = excl_jnts;

    for (int i = 0; i < (nv+2); i++) {
        nodes[i].force << 0., 0., 0.;
        nodes[i].torq << 0., 0., 0.;
        nodes[i].quat << node_quat[4*i], node_quat[4*i+1], node_quat[4*i+2], node_quat[4*i+3];
        if (i > 0) {
            dist_diff = nodes[i].pos - nodes[i-1].pos;
            // adj_dist.push_back(dist_diff);
            distmat[i].row(i-1) = dist_diff;
        }
        distmat[i].row(i) << 0., 0., 0.;
    }
    for (int i = 2; i < (nv+2); i++) {
        distmat[i].block(0,0,i-1,3) = \
            distmat[i-1].block(0,0,i-1,3).array().rowwise() \
            + distmat[i].row(i-1).array();
        // distmat[i-1] = - distmat[i-1];
        for (int j = 0; j < (i); j++) {
            distmat[j].row(i) = distmat[i].row(j);
        }
    }
    // extra for id 1:
    distmat[0].row(1) = distmat[1].row(0); // not negative - alr done
    // // extra for id nv+1:
    // distmat[nv+1] = - distmat[nv+1];
    // for (int i = 2; i < (nv+2); i++) {
    //     std::cout << i << std::endl;
    //     std::cout << distmat[i].block(0,0,i-1,3) << std::endl;
    // }
    // final distmat[a].row(b) = distmat[b].row(a) 
    // is the distance from a to b (b-a)
    // such that b < a,
    // JUST USE negative to get a to b!

    calculateNabKbandNabPsi_sub2(1, nv+1);

    calculateF2LocalTorq();

    // update force on actual simulation
    for (int i = 0; i < (nv+2); i++) {
        for (int j = 0; j < 3; j++) {
            node_torq[3*i+j] = nodes[i].torq(j);
            // node_torq_global[3*i+j] = nodes[nv+1-i].torq_global(j);
        }
    }
}

void DLO_iso::calculateF2LocalTorq()
{
    // Eigen::MatrixXd torqvec_indiv(nv+2,3);
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
        nodes[i].torq = DloUtils::rotVecQuat(
            torqvec.row(i),
            DloUtils::inverseQuat(nodes[i].quat)
        );
        // nodes[i].torq_global = torqvec.row(i);
    }
}

double DLO_iso::calculateEnergy()
{
    double energy_total;
    energy_total = 0.0;
    energy_total += calculateBendingEnergy();
    // energy_total += calculateTwistingEnergy();
    return energy_total;
}

double DLO_iso::calculateBendingEnergy()
{
    double energy_bending;
    energy_bending = 0.0;
    for (int i = 1; i < (nv+1); i++) {
        energy_bending += alpha_bar * nodes[i].kb.dot(nodes[i].kb) \
            / edges[i].l_bar;
    }
    return energy_bending;
}

double DLO_iso::calculateTwistingEnergy()
{
    double energy_twisting;
    double rot_ends;
    energy_twisting = 0.0;
    rot_ends = (edges[nv].theta - edges[0].theta);
    energy_twisting += beta_bar * rot_ends * rot_ends\
        / (2.0 * bigL_bar);

    return energy_twisting;
}


// ========================| start: misc calc |========================

void DLO_iso::initQe_o2m_loc(
    int dim_qo2m,
    double *q_o2m
    // int dim_qm2o,
    // double *q_m2o,
)
{
    qe_o2m_loc.x() = q_o2m[0];
    qe_o2m_loc.y() = q_o2m[1];
    qe_o2m_loc.z() = q_o2m[2];
    qe_o2m_loc.w() = q_o2m[3];

    qe_o2m_loc = qe_o2m_loc.normalized();

    // qe_m2o_loc.x() = q_m2o[0];
    // qe_m2o_loc.y() = q_m2o[1];
    // qe_m2o_loc.z() = q_m2o[2];
    // qe_m2o_loc.w() = q_m2o[3];
}

void DLO_iso::calculateOf2Mf(
    int dim_mato, double *mat_o,
    int dim_matres, double *mat_res
)
{
    Eigen::Matrix3d mat_mato;
    Eigen::Matrix3d mat_result;
    mat_mato << mat_o[0], mat_o[1], mat_o[2],
        mat_o[3], mat_o[4], mat_o[5],
        mat_o[6], mat_o[7], mat_o[8];

    Eigen::Quaterniond q_mato(mat_mato);
    // Eigen::Quaterniond q_mid;
    // q_mid = q_mato * qe_o2m_loc
    mat_result = (q_mato * qe_o2m_loc).normalized().toRotationMatrix();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat_res[j+3*i] = mat_result(i,j);
        }
    }
}

double DLO_iso::angBtwn3(
    int dim_v1, double *v1,
    int dim_v2, double *v2,
    int dim_va, double *va
)
{
    Eigen::Vector3d v1_c(v1[0], v1[1], v1[2]);
    Eigen::Vector3d v2_c(v2[0], v2[1], v2[2]);
    Eigen::Vector3d va_c(va[0], va[1], va[2]);
    double theta_diff, dot_norm_val;
    // v1_c << v1[0], v1[1], v1[2];
    // v2_c << v2[0], v2[1], v2[2];
    // va_c << va[0], va[1], va[2];
    dot_norm_val = v1_c.dot(v2_c)/(v1_c.norm()*v2_c.norm());
    if (dot_norm_val > 1.) {dot_norm_val = 1.;}
    theta_diff = acos(dot_norm_val);
    if ((v1_c.cross(v2_c)).dot(va_c) < 0) {theta_diff *= -1.;}
    return theta_diff;
}
// ========================| end: misc calc |========================