#include "Dlo_s2f.h"
#include "Ds2f_obj.h"

#include <iostream>
#include <iomanip>
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
*A* Nodes are in reversed order from MuJoCo (incl. the python part)
    (i.e., child defined as node 0, parents as node n)
    torque defined is from parent on child, from n to 0
    therefore it is same sign but reversed order from MuJoCo.
    e.g.,   MuJoCo_numbering:   0a   1a   2a   3a
            DER_numbering:      3b   2b   1b   0b
        torques are from 3b to 2b (consistent w MuJ - 0a to 1a),
        define in nodes order 0b to 3b (opposite of MuJ - 3a to 0a)

*B* Distance matrix defined:
    final distmat[a].row(b) = distmat[b].row(a) 
    is the distance from a to b (b-a)
    such that b < a,
    JUST USE negative to get a to b!
    this distance.cross(force) will give the torque direction in *A*.
*/

// std::mutex mtx;

DLO_s2f::DLO_s2f(
    const double rlength,
    const int rpieces,
    const double r_weight,  // note: in weight, not mass
    const bool boolErrs
)
{
    Vecnodes x1;

    // init variables
    nv = rpieces-1;
    length = rlength;
    w_perpiece << 0.0, 0.0, -r_weight / rpieces;
    raiseErrs = boolErrs;

    Eigen::Matrix<double, Eigen::Dynamic, 3> dist1(nv+2, 3);
    for (int i = 0; i < (nv+2); i++) {
        nodes.push_back(x1);
        distmat.push_back(dist1);
    }
}

// Nodes are in reversed order from MuJoCo 
// (i.e., child defined as node 0, parents as node n)
// torque defined is from child on parents, from 0 to n
// therefore it is reversed sign and order from MuJoCo.


bool DLO_s2f::calculateExternalForces(
    int dim_efp, double *ext_forcepos,
    int dim_ef, double *ext_force,
    int dim_et, double *ext_torq,
    int dim_nt, double *node_torq,
    int dim_np, double *node_pos,
    int dim_nq, double *node_quat
)
{
    boolUnreliable = false;
    updateState(
        dim_nt, node_torq,
        dim_np, node_pos,
        dim_nq, node_quat
    );
    classifyExtForces(dim_efp, ext_forcepos);
    
    bool solvableBool = solvable_checks();
    if (!solvableBool) {return solvableBool;}
    
    /*
    SOLVE~
    - Evaluations are all done wrt smaller id nodes.
        A. Solution will first solve a set of linear equations 
        for the external forces. This is done in the form [A]f = [B]
        B. Torques will be solved by evaluating the torques
        around each piece where ft is applied. 
        */ 
       // Part A.
       // forming matrix [A] and vector [B]
    getMidSections();
    formSolvingMatrices();
    // solving for f
    fSol.resize(n_force*3);
    fSol = Ds2fUtils::solveUTXBig(matA, vecB);
    
    // Part B.
    // directly solving for torques
    tSol.resize(n_force*3);
    solveTorques();
    
    allocateFT(
        dim_ef, ext_force,   
        dim_et, ext_torq   
    );
    
    return solvableBool;
}

void DLO_s2f::updateState(
    int dim_nt, double *node_torq,
    int dim_np, double *node_pos,
    int dim_nq, double *node_quat
)
{
    Eigen::Vector3d dist_diff;
    nodeposMat.resize(nv+2, 3);
    // forceMat.resize(n_force, 3);

    for (int i = 0; i < (nv+2); i++) {
        nodes[i].torq << 0., 0., 0.;
        nodes[i].pos << node_pos[3*i], node_pos[3*i+1], node_pos[3*i+2];
        nodeposMat.row(i) << node_pos[3*i], node_pos[3*i+1], node_pos[3*i+2];
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
        distmat[i-1] = - distmat[i-1];
        for (int j = 0; j < (i); j++) {
            distmat[j].row(i) = - distmat[i].row(j);
        }
    }
    // extra for id 1:
    distmat[0].row(1) = distmat[1].row(0); // not negative - alr done
    // extra for id nv+1:
    distmat[nv+1] = - distmat[nv+1];
    
    // update torque from actual simulation (reversed order)
    for (int i = 0; i < (nv+2); i++) {
        for (int j = 0; j < 3; j++) {
            nodes[i].torq(j) = node_torq[3*i+j];
        }
    }
    calculateLocal2GlobalTorq();
}

void DLO_s2f::calculateLocal2GlobalTorq()
{
    // Eigen::Matrix<double, Eigen::Dynamic, 3> torqvec(nv+2,3);
    // for (int i = 0; i < (nv+2); i++) {
        // torqvec.row(i) << 0., 0., 0.;
    // }
    for (int i = 0; i < (nv+2); i++) {
        nodes[i].torq = Ds2fUtils::rotVecQuat(
            nodes[i].torq,
            nodes[i].quat
        );
        // std::cout << 'b' << std::endl;
        // std::cout << i << std::endl;
        // std::cout << nodes[i].torq << std::endl;
    }
    // torq_glob = torqvec;
}

void DLO_s2f::classifyExtForces(
    int dim_efp, double *ext_forcepos
)
{
    Extforces ef1;
    Eigen::Vector3d efp; // extra forces acting on system
    Eigen::VectorXd n2f_mag; // distance between nodes and external force
    std::vector<std::pair<int, int>> indexedLinkId;

    n_force = (int) (dim_efp/3);
    // find closest and 2nd closest nodes --> closest link
    for (int i = 0; i < n_force; i++) {
        extforces.push_back(ef1);

        efp << ext_forcepos[3*i], ext_forcepos[3*i+1], ext_forcepos[3*i+2];
        n2f_mag = (nodeposMat.rowwise() - efp.transpose()).rowwise().norm();
        // Find the index of the minimum row norm
        int minIndex = -1;
        double minValue = std::numeric_limits<double>::infinity();
        // Find the index of the minimum row norm
        for (int i = 0; i < n2f_mag.size(); ++i) {
            if (n2f_mag(i) < minValue) {
                minValue = n2f_mag(i);
                minIndex = i;
            }
        }
        // Ensure we're within bounds before accessing adjacent elements
        double adjacentMin = std::numeric_limits<double>::infinity();
        int lidTmp = -1;

        // Check adjacent values (ensure not out of bounds)
        if (minIndex > 0 && n2f_mag(minIndex - 1) < adjacentMin) {
            adjacentMin = n2f_mag(minIndex - 1);
            lidTmp = minIndex-1; // force belongs to previous link
        }
        // added buffer of +1e-5 to ensure forces defined at nodes
        // have a bias towards the next linkid
        // (e.g., force at node[i] will be attributed to link[i] and not link[i-1])
        if (minIndex < n2f_mag.size() - 1 && n2f_mag(minIndex + 1) < adjacentMin+1e-5) {
            adjacentMin = n2f_mag(minIndex + 1);
            lidTmp = minIndex; // force belongs to current link
        }
        indexedLinkId.push_back({lidTmp, i});  // Store value and its index
    }

    // Sort the vector of pairs based on the value
    std::sort(indexedLinkId.begin(), indexedLinkId.end());

    // Update the original vector to be sorted and store the indices of the reorder
    for (int i = 0; i < indexedLinkId.size(); ++i) {
        int oordTmp = indexedLinkId[i].second;
        extforces[i].old_order = oordTmp;  // Store the original index
        extforces[i].link_id = indexedLinkId[i].first;
        extforces[i].pos << ext_forcepos[3*oordTmp], ext_forcepos[3*oordTmp+1], ext_forcepos[3*oordTmp+2];
        extforces[i].rel_pos = extforces[i].pos.transpose() - nodeposMat.row(extforces[i].link_id);
        // std::cout<<'t'<<std::endl;
        // std::cout<<extforces[i].pos.transpose()<<std::endl;
        // std::cout<<nodeposMat.row(extforces[i].link_id)<<std::endl;
    }
}

bool DLO_s2f::solvable_checks()
{
    for (int i = 0; i < n_force-1; ++i) {
        if ((extforces[i+1].link_id - extforces[i].link_id) < 3) {
            // Condition A
            // ensure that there is at least one link space between forces
            // link_id of adjacent forces must be at least 1 apart (diff of 2).
            boolUnreliable = true;
            std::cerr << "Error: forces are side-by-side and cannot be solved." << std::endl;
            if (raiseErrs) {
                throw std::runtime_error(
                    "Solvable checks failed.. not enought pieces in each section (min. 2)."
                );                
            }
            return false;
        }
    }
    return true;
}

void DLO_s2f::getMidSections()
{
    secMid1.resize(n_force-1);
    secMid2.resize(n_force-1);
    for (int i = 0; i < n_force-1; i++) {
        double id_diff = extforces[i+1].link_id - extforces[i].link_id;
        secMid1[i] = (int)(
            id_diff/3 + extforces[i].link_id + 0.5
        );
        secMid2[i] = (int)(
            2*id_diff/3 + extforces[i].link_id + 0.5
        );
        // std::cout << 'm' << std::endl;
        // std::cout << secMid1[i] << std::endl;
        // std::cout << secMid2[i] << std::endl;
    }
}

void DLO_s2f::formSolvingMatrices()
{
    Eigen::Vector3d distVec1;
    Eigen::Vector3d distVec2;
    matA.resize(n_force*2*3,n_force*3);
    matA.setZero();
    vecB.resize(n_force*2*3);
    vecB.setZero();
    // first row: sum(F) = 0
    for (int i = 0; i < n_force; i++) {
        matA.block(0,i*3,3,3).setIdentity();
    }
    vecB(Eigen::seqN(0,3)) << 0.0, 0.0, 0.0;
    if (w_perpiece.norm() > 1e-10) {
        vecB(Eigen::seqN(0,3)) += - w_perpiece*(nv+1);
        // std::cout << vecB(Eigen::seqN(0,3)) << std::endl;
        // std::cout << w_perpiece << std::endl;
        // std::cout << 'O' << std::endl;
    }
    // rest of the rows
    for (int i = 1; i < n_force; i++) {
        /*
        Note: all calculations are done while 
              taking moments from node[i] for link [i].
        - each row*3 is a section
        - each column*3 is the effect of force on that section
        get distance from node_a to node_b (dist=b_pos-a_pos) for s_(i-1),
        where a < b. use the skewsymmetric as each block in that row.
        */
        // std::cout << 'A' << std::endl;
        // std::cout << secMid1[i-1] << std::endl;
        // std::cout << secMid2[i-1] << std::endl;
        // calc dist skewsym
        distVec1 = (
            nodes[secMid1[i-1]+1].pos - nodes[secMid1[i-1]].pos
        );
        distVec2 = (
            nodes[secMid2[i-1]+1].pos - nodes[secMid2[i-1]].pos
        );
        // std::cout << 'd' << std::endl;
        // std::cout << '1' << std::endl;
        // std::cout << secMid1[i-1] << std::endl;
        // std::cout << nodes[secMid1[i-1]].pos << std::endl;
        // std::cout << nodes[secMid1[i-1]+1].pos << std::endl;
        // std::cout << distVec1 << std::endl;
        // std::cout << '2' << std::endl;
        // std::cout << secMid2[i-1] << std::endl;
        // std::cout << nodes[secMid2[i-1]].pos << std::endl;
        // std::cout << nodes[secMid2[i-1]+1].pos << std::endl;
        // std::cout << distVec2 << std::endl;


        if ((distVec1.cross(distVec2)).norm() < 1e-10) {
            // Condition C
            boolUnreliable = true;
            std::cerr << "Error: pieces in the section are parallel." << std::endl;
            if (raiseErrs) {
                throw std::runtime_error(
                    "Linear independence checks failed.. pieces are not linearly independent."
                );                
            }
        }

        // allocate to each block in the row*3
        for (int j = i; j < n_force; j++) {
            matA.block(i*2*3,j*3,3,3) = Ds2fUtils::createSkewSym(distVec1);
            matA.block(i*2*3+3,j*3,3,3) = Ds2fUtils::createSkewSym(distVec2);
        }
        /*
        calc RHS - torque diff between L and R ends
        negate because totaltorquefromforce + totaltorquediff = 0
        totaltorquefromforce = -totaltorquediff
        */
        vecB(Eigen::seqN(i*2*3,3)) = - (
            nodes[secMid1[i-1]].torq - nodes[secMid1[i-1]+1].torq
        );
        vecB(Eigen::seqN(i*2*3+3,3)) = - (
            nodes[secMid2[i-1]].torq - nodes[secMid2[i-1]+1].torq
        );
        // std::cout << 't' << std::endl;
        // std::cout << '1' << std::endl;
        // std::cout << secMid1[i-1] << std::endl;
        // std::cout << nodes[secMid1[i-1]].torq << std::endl;
        // std::cout << nodes[secMid1[i-1]+1].torq << std::endl;
        // std::cout << vecB(Eigen::seqN(i*2*3,3)) << std::endl;
        // std::cout << '2' << std::endl;
        // std::cout << secMid2[i-1] << std::endl;
        // std::cout << nodes[secMid2[i-1]].torq << std::endl;
        // std::cout << nodes[secMid2[i-1]+1].torq << std::endl;
        // std::cout << vecB(Eigen::seqN(i*2*3+3,3)) << std::endl;

        if (w_perpiece.norm() > 1e-10) {
            // if weight > 0, 'add' torque contribution 
            // of weight from R side (higher id)
            // number of links on the right of current link
            double n_right1 = nv - secMid1[i-1];
            double n_right2 = nv - secMid2[i-1];
            // distVec.cross(w_perpiece/2) (adding 0.5) is 
            // the torque contribution from the weight of current link itself
            vecB(Eigen::seqN(i*2*3,3)) += - (
                distVec1.cross((n_right1+0.5)*w_perpiece)
            );
            vecB(Eigen::seqN(i*2*3+3,3)) += - (
                distVec2.cross((n_right2+0.5)*w_perpiece)
            );
        }
        // std::cout << 'd' << std::endl;
        // std::cout << distVec1 << std::endl;
        // std::cout << vecB(Eigen::seqN(i*2*3,3)) << std::endl;
        // std::cout << std::abs(distVec1.dot(vecB(Eigen::seqN(i*2*3,3)))) << std::endl;
        std::cout << std::abs(distVec1.dot(vecB(Eigen::seqN(i*2*3,3)))) << std::endl;
        std::cout << std::abs(distVec2.dot(vecB(Eigen::seqN(i*2*3+3,3)))) << std::endl;
        if (
            std::abs(distVec1.dot(vecB(Eigen::seqN(i*2*3,3)))) > 1e-10
            || std::abs(distVec2.dot(vecB(Eigen::seqN(i*2*3+3,3)))) > 1e-10
        ) {
            boolUnreliable = true;
            std::cerr << "Error: DLO not in static equilibrium." << std::endl;
            if (raiseErrs) {
                throw std::runtime_error(
                    "Static equilibrium checks failed.. DLO is not in SE and torque can't be balanced."
                );
            }
        }
    }
}

void DLO_s2f::solveTorques()
{
    int iLId;
    Eigen::Vector3d torqueImbal;
    torqueImbal << 0.0, 0.0, 0.0;
    Eigen::Vector3d distVec;
    Eigen::Vector3d fVec;
    for (int i = 0; i < n_force; i++) {
        torqueImbal << 0.0, 0.0, 0.0;
        iLId = extforces[i].link_id;
        distVec = nodes[iLId+1].pos - nodes[iLId].pos;
        // std::cout << 'a' << std::endl;
        // std::cout << iLId << std::endl;
        // std::cout << distVec << std::endl;
        // std::cout << torqueImbal << std::endl;
        // consider removing the iLId condition checks
        // can assume is always within range based on classifyExtForces func.
        // this part excludes nodes[0].torq and assumes it is 0. 
        torqueImbal += nodes[iLId].torq - nodes[iLId+1].torq;
        // std::cout << 'b' << std::endl;
        // std::cout << nodes[iLId].torq << std::endl;
        // std::cout << nodes[iLId+1].torq << std::endl;
        // std::cout << torqueImbal << std::endl;
        // if (iLId > -1) {
            // // add L torque
            // torqueImbal += nodes[iLId].torq;
            // std::cout << 'a' << std::endl;
            // std::cout << nodes[iLId].torq << std::endl;
            // std::cout << torqueImbal << std::endl;
        // }
        // if (iLId < nv+1) {
            // // add R torque
            // torqueImbal -= nodes[iLId+1].torq;
            // std::cout << 'b' << std::endl;
            // std::cout << nodes[iLId+1].torq << std::endl;
            // std::cout << torqueImbal << std::endl;
        // }
        // torque contribution from extforces,
        // taking moments from node[i] for link [i]

        fVec = fSol(Eigen::seqN(i*3,3));
        torqueImbal += extforces[i].rel_pos.cross(fVec);
        // std::cout << 'c' << std::endl;
        // std::cout << extforces[i].rel_pos.cross(fVec) << std::endl;
        // std::cout << extforces[i].rel_pos << std::endl;
        // std::cout << fVec << std::endl;
        // // std::cout << std::setprecision(17) << std::scientific << torqueImbal << std::endl;
        // std::cout << torqueImbal << std::endl;
        for (int j = i+1; j < n_force; j++) {
            fVec = fSol(Eigen::seqN(j*3,3));
            torqueImbal += distVec.cross(fVec);
            // std::cout << 'd' << std::endl;
            // std::cout << distVec.cross(fVec) << std::endl;
            // std::cout << torqueImbal << std::endl;
        }
        if (w_perpiece.norm() > 1e-10) {
            torqueImbal += distVec.cross((nv-iLId+0.5)*w_perpiece);
            // std::cout << 'e' << std::endl;
            // std::cout << distVec.cross((nv-iLId+0.5)*w_perpiece) << std::endl;
            // std::cout << nv-iLId+0.5 << std::endl;
            // std::cout << std::setprecision(17) << std::scientific << torqueImbal << std::endl;
        }
        tSol(Eigen::seqN(i*3,3)) = -torqueImbal;
    }
}

void DLO_s2f::allocateFT(
    int dim_ef, double *ext_force,
    int dim_et, double *ext_torq
)
{
    Eigen::Map<Eigen::VectorXd> extF(ext_force, dim_ef);
    Eigen::Map<Eigen::VectorXd> extT(ext_torq, dim_et);
    for (int i = 0; i < n_force; i++) {
        // std::cout << 'p' << std::endl;
        // std::cout << extforces[i].old_order << std::endl;
        extF(Eigen::seqN(3*extforces[i].old_order,3)) = \
            fSol(Eigen::seqN(3*i,3));
        extT(Eigen::seqN(3*extforces[i].old_order,3)) = \
            tSol(Eigen::seqN(3*i,3));
    }
}