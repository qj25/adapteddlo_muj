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
*/

// std::mutex mtx;

DLO_s2f::DLO_s2f(
    const double rlength,
    const int rpieces,
    const double r_weight,  // note: in weight, not mass
    const bool boolErrs,
    const bool boolSolveTorq
)
{
    // init variables
    nv = rpieces-1;
    length = rlength;
    w_perpiece << 0.0, 0.0, -r_weight / rpieces;
    raiseErrs = boolErrs;
    bSolveTorq = boolSolveTorq;

    nodeposMat.resize(nv+2, 3);
    nodetorqMat.resize(nv+2, 3);
}

bool DLO_s2f::calculateExternalForces(
    int dim_nt, double *node_torq,
    int dim_np, double *node_pos,
    int dim_nq, double *node_quat
)
{
    // Clear previous force sections
    force_sections.clear();
    
    updateState(
        dim_nt, node_torq,
        dim_np, node_pos,
        dim_nq, node_quat
    );
    
    // Part A.
    // solving for forces while splitting sections
    std::cout << "Entering checks =============================" << std::endl;
    bool consistPass = checkConsistency();
    std::cout << "End checks" << std::endl;
    if (!consistPass) {return false;}
    // Part B.
    // directly solving for torques
    n_force = disturbed_sections_.size();
    force_sections.resize(n_force);

    if (bSolveTorq) {solveTorques();}
    else {solveForcePos();}
    
    return true;
}

void DLO_s2f::updateState(
    int dim_nt, double *node_torq,
    int dim_np, double *node_pos,
    int dim_nq, double *node_quat
)
{
    if (dim_nt/3!=nv+2||dim_np/3!=nv+2||dim_nq/4!=nv+2) {
        std::cerr << "Error: Dimension of input does not match specified nv." << std::endl;
        throw std::runtime_error(
            "Error: Dimension of input does not match specified nv."
        );                
    }
    nodeposMat.resize(nv+2, 3);
    nodetorqMat.resize(nv+2, 3);
    Eigen::Vector4d quat;
    
    nodeposMat.setZero();
    nodetorqMat.setZero();
    for (int i = 0; i < (nv+2); i++) {
        // Update torqs
        nodetorqMat.row(i) << node_torq[3*i], node_torq[3*i+1], node_torq[3*i+2];
        // Globalize torque
        quat << node_quat[4*i], node_quat[4*i+1], node_quat[4*i+2], node_quat[4*i+3];
        nodetorqMat.row(i) = Ds2fUtils::rotVecQuat(
            nodetorqMat.row(i).transpose(),
            quat
        ).transpose();
        // Update pos
        nodeposMat.row(i) << node_pos[3*i], node_pos[3*i+1], node_pos[3*i+2];
    }
}

void DLO_s2f::findParallelEndSections()
{
    // Write new func to find zero force:
    // If the torques at the end piece balances out well without
    // additional external force, then it is considered undisturbed.
    // If so, evaluate the next section.

    // Reset the indices
    undistStart = -1;
    undistEnd = -1;

    // Check start of wire
    Eigen::Vector3d start_avg_dir = Eigen::Vector3d::Zero();
    int start_parallel_count = 0;
    bool start_parallel = true;
    
    // Check first two pieces
    Eigen::Vector3d first_dir = (nodeposMat.row(1) - nodeposMat.row(0)).normalized();
    start_avg_dir = first_dir;
    start_parallel_count = 1;
    
    // Continue checking until we find non-parallel pieces
    for (int i = 1; i < nv+1 && start_parallel; i++) {
        Eigen::Vector3d current_dir = (nodeposMat.row(i+1) - nodeposMat.row(i)).normalized();
        double dot_product = std::abs(current_dir.dot(start_avg_dir));
        
        if (dot_product > 0.999) {  // Nearly parallel (cos(angle) ≈ 1)
            start_avg_dir = (start_avg_dir * start_parallel_count + current_dir) / (start_parallel_count + 1);
            start_parallel_count++;
            undistStart = i;
        } else {
            start_parallel = false;
        }
    }
    
    // Check end of wire
    Eigen::Vector3d end_avg_dir = Eigen::Vector3d::Zero();
    int end_parallel_count = 0;
    bool end_parallel = true;
    
    // Check last two pieces
    Eigen::Vector3d last_dir = (nodeposMat.row(nv+1) - nodeposMat.row(nv)).normalized();
    end_avg_dir = last_dir;
    end_parallel_count = 1;
    
    // Continue checking until we find non-parallel pieces
    for (int i = nv; i > 0 && end_parallel; i--) {
        Eigen::Vector3d current_dir = (nodeposMat.row(i) - nodeposMat.row(i-1)).normalized();
        double dot_product = std::abs(current_dir.dot(end_avg_dir));
        
        if (dot_product > 0.999) {  // Nearly parallel (cos(angle) ≈ 1)
            end_avg_dir = (end_avg_dir * end_parallel_count + current_dir) / (end_parallel_count + 1);
            end_parallel_count++;
            undistEnd = i-1;
        } else {
            end_parallel = false;
        }
    }
    if (undistStart == -1) {undistStart = 0;}
    if (undistEnd == -1) {undistEnd = nv;}
}

void DLO_s2f::checkUDEndSections()
{
    // Reset the indices
    undistStart = -1;
    undistEnd = -1;

    const double torque_tolerance = 1e-8;  // Tolerance for torque norm

    // Check end of wire
    // Continue checking until we find non-zero force-torque piece
    for (int i = 0; i < nv+1; i++) {
        Eigen::Vector3d distVec = (nodeposMat.row(i) - nodeposMat.row(i+1)); // from larger to smaller id
        Eigen::Vector3d Ctorq = (nodetorqMat.row(i+1) - nodetorqMat.row(i));
        // Add weight contribution to Ctorq if present
        if (w_perpiece.norm() > 1e-10) {
            int nLeft = i;
            Ctorq += (distVec.cross((nLeft+0.5)*w_perpiece));
        }
        
        // Check if torque norm exceeds tolerance
        if (Ctorq.norm() > torque_tolerance) {
            undistStart = i;
            break;
        }
    }
    
    // Check end of wire
    // Continue checking until we find non-zero force-torque piece
    for (int i = nv; i >= 0; i--) {
        Eigen::Vector3d distVec = (nodeposMat.row(i+1) - nodeposMat.row(i)); // from smaller to larger id
        Eigen::Vector3d Ctorq = (nodetorqMat.row(i+1) - nodetorqMat.row(i));
        // Add weight contribution to Ctorq if present
        if (w_perpiece.norm() > 1e-10) {
            int nRight = nv-i;
            Ctorq += (distVec.cross((nRight+0.5)*w_perpiece));
        }
        
        // Check if torque norm exceeds tolerance
        if (Ctorq.norm() > torque_tolerance) {
            undistEnd = i;
            break;
        }
    }

    // Set default values if no sections were found
    if ((undistStart == -1 || undistEnd == -1) && raiseErrs) {
        std::cerr << "Rod is completely undisturbed" << std::endl;
    }
}

bool DLO_s2f::checkConsistency()
{
    const double tolC2 = 1e-3;
    const double tolC3 = 1e-3;

    Eigen::Vector3d avg_F;
    avg_F.setZero();
    int force_count = 0;

    // Clear previous sections and forces
    undisturbed_sections_.clear();
    disturbed_sections_.clear();

    std::vector<Section> undisturbed_sections;
    int current_undisturbed_start = -1;
    int current_undisturbed_end = -1;
    bool in_undisturbed_section = false;

    // // Find parallel sections at both ends
    // // findParallelEndSections();
    checkUDEndSections();
    if (undistEnd <= undistStart) {
        return false;
    }

    // Check consistency for each consecutive pair of vectors
    // from 0 to nv-2 (id of 3rd last piece) because of triple piece checks
    for (int i = undistStart; i < undistEnd-1; i++) {
        std::cout << "i = " << i << std::endl;
        Eigen::Vector3d distVec1 = (nodeposMat.row(i+1) - nodeposMat.row(i));
        Eigen::Vector3d distVec2 = (nodeposMat.row(i+2) - nodeposMat.row(i+1));
        
        // Calculate C1 and C2 (torque differences)
        Eigen::Vector3d C1 = (nodetorqMat.row(i+1) - nodetorqMat.row(i));
        Eigen::Vector3d C2 = (nodetorqMat.row(i+2) - nodetorqMat.row(i+1));

        // Add weight contribution if present
        if (w_perpiece.norm() > 1e-10) {
            int nRight = nv - i;
            C1 += - (distVec1.cross((nRight+0.5)*w_perpiece));
            nRight = nv - (i+1);
            C2 += - (distVec2.cross((nRight+0.5)*w_perpiece));
        }

        // Check consistency condition: distVec1.C2 + distVec2.C1 < tolerance
        if (!in_undisturbed_section) {
            double consistency_value = distVec1.dot(C2) + distVec2.dot(C1);
            std::cout << "consist_val2 = " << consistency_value << std::endl;
            if (std::abs(consistency_value) > tolC2) {
                // Only have to test if !in_undisturbed_section:
                // - if in_undisturbed_section, C3consist would pass for 1 2 3
                //    and C2 for 2 and 3 would logically pass
                // Pieces do not belong to the same undisturbed section
                // and force difference is not in the plane spanned by 
                // distVec1 and distVec2
                continue;
            }
        }

        // if consistent, do checks for 3 pieces consistency
        // Calculate A1 and A2 matrices
        Eigen::Vector3d distVec3 = (nodeposMat.row(i+3) - nodeposMat.row(i+2));
        if (hasParallelVectors(distVec1, distVec2, distVec3)) {
            // Parallel vectors condition
            continue;
        }
        Eigen::Matrix3d A1 = Ds2fUtils::createSkewSym(distVec1);
        Eigen::Matrix3d A2 = Ds2fUtils::createSkewSym(distVec2);
        Eigen::Matrix3d A3 = Ds2fUtils::createSkewSym(distVec3);
        Eigen::Vector3d C3 = (nodetorqMat.row(i+3) - nodetorqMat.row(i+2));

        // Add weight contribution to C3 if present
        if (w_perpiece.norm() > 1e-10) {
            int nRight = nv - (i+2);
            C3 += - (distVec3.cross((nRight+0.5)*w_perpiece));
        }

        bool passConsist3 = false;
        
        // Stack the matrices and vectors for least squares computation
        Eigen::MatrixXd A(9, 3);
        A.block(0, 0, 3, 3) = A1;
        A.block(3, 0, 3, 3) = A2;
        A.block(6, 0, 3, 3) = A3;
        
        Eigen::VectorXd c(9);
        c.segment(0, 3) = C1;
        c.segment(3, 3) = C2;
        c.segment(6, 3) = C3;

        // Compute least squares solution and check residual
        std::pair<Eigen::Vector3d, double> result = Ds2fUtils::computeLeastSquares(A, c);
        passConsist3 = result.second < tolC3;
        std::cout << "consist_val3 = " << result.second << std::endl;

        if (passConsist3) {
            if (!in_undisturbed_section) {
                current_undisturbed_start = i;
                in_undisturbed_section = true;
                avg_F.setZero();
                force_count = 0;
            }
            // Update the end index to include all three pieces
            current_undisturbed_end = i + 2;
            avg_F += result.first;
            force_count++;
        } else {
            if (in_undisturbed_section) {
                // End current undisturbed section
                if (force_count > 0) {
                    avg_F /= force_count;
                }
                Section new_section(current_undisturbed_start, current_undisturbed_end);
                new_section.avg_force = avg_F;
                undisturbed_sections.push_back(new_section);
                in_undisturbed_section = false;
            }
        }
    }

    // Handle the last section if still in progress
    if (in_undisturbed_section) {
        if (force_count > 0) {
            avg_F /= force_count;
        }
        Section new_section(current_undisturbed_start, undistEnd);
        new_section.avg_force = avg_F;
        undisturbed_sections.push_back(new_section);
    }
    if (undisturbed_sections.empty()) {return false;}

    // Derive disturbed sections from undisturbed sections
    std::vector<Section> disturbed_sections;
    int last_end = -1;
    
    for (const auto& section : undisturbed_sections) {
        // If there's a gap between the last end and this section's start
        if (section.start_idx > last_end + 1) {
            disturbed_sections.emplace_back(last_end + 1, section.start_idx - 1);
        }
        last_end = section.end_idx;
    }
    
    // Handle any remaining section after the last undisturbed section
    if (last_end < undistEnd) {
        disturbed_sections.emplace_back(last_end + 1, undistEnd);
    }

    // Check more disturbed sections than undisturbed
    if (disturbed_sections.size() - undisturbed_sections.size() != 1) {
        if (raiseErrs) {
            throw std::runtime_error("DS-UDS != 1");
        }
        return false;
    }

    // // Print undisturbed and disturbed sections
    // std::cout << "\nUndisturbed Sections:" << std::endl;
    // for (const auto& section : undisturbed_sections) {
    //     std::cout << "start_idx: " << section.start_idx << ", end_idx: " << section.end_idx << std::endl;
    // }
    // std::cout << "\nDisturbed Sections:" << std::endl;
    // for (const auto& section : disturbed_sections) {
    //     std::cout << "start_idx: " << section.start_idx << ", end_idx: " << section.end_idx << std::endl;
    // }
    // std::cout << std::endl;

    // Iterate through undisturbed sections from end to beginning
    int idStart = disturbed_sections.size()-1;
    disturbed_sections[idStart].avg_force = undisturbed_sections[idStart-1].avg_force;
    for (int i = idStart-1; i > 0; i--) {
        disturbed_sections[i].avg_force = undisturbed_sections[i-1].avg_force;
        disturbed_sections[i].avg_force -= undisturbed_sections[i].avg_force;
    }
    // calculate last DS.avg_force using sum(F) = 0
    disturbed_sections[0].avg_force = -undisturbed_sections[0].avg_force;

    // Store the sections in class members for later use
    undisturbed_sections_ = undisturbed_sections;
    disturbed_sections_ = disturbed_sections;

    // Return false if no undisturbed sections found, true otherwise
    return true;
}

Eigen::Vector3d DLO_s2f::findExactMidpoint(int startId, int endId) {
    // NOT in use due to w_perpiece not being specified for each custom piece.
    // currently this class is only able to handle fixed edge length and weight.
    // Calculate total arc length of the section using matrix operations
    Eigen::MatrixXd diff = nodeposMat.block(startId+1, 0, endId-startId, 3) - nodeposMat.block(startId, 0, endId-startId, 3);
    double total_length = diff.rowwise().norm().sum();
    
    // Find exact midpoint position
    double target_length = total_length / 2.0;
    double current_length = 0.0;
    
    // Accumulate lengths until we reach or exceed the target length
    for (int i = 0; i < diff.rows(); i++) {
        double segment_length = diff.row(i).norm();
        if (current_length + segment_length >= target_length) {
            // Calculate how far along this segment we need to go
            double remaining_length = target_length - current_length;
            double ratio = remaining_length / segment_length;
            
            // Interpolate between current node and next node
            return nodeposMat.row(startId + i) - nodeposMat.row(startId) + 
                   ratio * diff.row(i);
        }
        current_length += segment_length;
    }
    
    // If we somehow get here, return the last node position
    return nodeposMat.row(endId) - nodeposMat.row(startId);
}

Eigen::Vector3d DLO_s2f::findMiddleNodePosition(int startId, int endId) {
    double midIndex = (endId + startId) / 2.0;
    int lowerNode = std::floor(midIndex);
    int upperNode = std::ceil(midIndex);
    
    // If we're exactly at a node
    if (lowerNode == upperNode) {
        return nodeposMat.row(lowerNode) - nodeposMat.row(startId);
    }
    
    // If we're between nodes, return average position
    return (nodeposMat.row(lowerNode) + nodeposMat.row(upperNode)) / 2.0 - nodeposMat.row(startId);
}

void DLO_s2f::solveTorques()
{
    int startId;
    int endId;
    Eigen::Vector3d torqueImbal;
    torqueImbal << 0.0, 0.0, 0.0;
    Eigen::Vector3d distVec;
    Eigen::Vector3d fVec;
    Eigen::Vector3d relpos;
    for (int i = 0; i < n_force; i++) {
        torqueImbal << 0.0, 0.0, 0.0;
        endId = disturbed_sections_[i].end_idx + 1;
        startId = disturbed_sections_[i].start_idx;
        distVec = nodeposMat.row(endId) - nodeposMat.row(startId);
        torqueImbal += nodetorqMat.row(startId) - nodetorqMat.row(endId);
                
        // torq from force addition                
        relpos = findMiddleNodePosition(startId, endId);
        fVec = disturbed_sections_[i].avg_force;
        torqueImbal += relpos.cross(fVec);
        for (int j = i+1; j < n_force; j++) {
            fVec = disturbed_sections_[j].avg_force;
            torqueImbal += distVec.cross(fVec);
        }

        // torq from weight
        if (w_perpiece.norm() > 1e-10) {
            torqueImbal += distVec.cross((nv-endId+1)*w_perpiece);
            // add the internal weight torques
            // Create matrix of position differences between edge midpoints and start position
            Eigen::MatrixXd edgeMidpoints(endId - startId, 3);
            for (int i = 0; i < endId - startId; i++) {
                edgeMidpoints.row(i) = (nodeposMat.row(startId + i + 1) + nodeposMat.row(startId + i)) / 2.0 - nodeposMat.row(startId);
                Eigen::Vector3d edgeVec = edgeMidpoints.row(i).transpose();
                torqueImbal += edgeVec.cross(w_perpiece);
            }
        }
        // set to force_sections
        force_sections[i].torque = -torqueImbal;
        force_sections[i].force = disturbed_sections_[i].avg_force;
        force_sections[i].start_pos = nodeposMat.row(startId);
        force_sections[i].end_pos = nodeposMat.row(endId);
    }
}

void DLO_s2f::solveForcePos()
{
    int startId;
    int endId;
    Eigen::Vector3d torqueImbal;
    torqueImbal << 0.0, 0.0, 0.0;
    Eigen::Vector3d distVec;
    Eigen::Vector3d fVec;
    Eigen::Vector3d force_pos;
    force_pos << 0.0, 0.0, 0.0;
    for (int i = 0; i < n_force; i++) {
        torqueImbal << 0.0, 0.0, 0.0;
        endId = disturbed_sections_[i].end_idx + 1;
        startId = disturbed_sections_[i].start_idx;
        distVec = nodeposMat.row(endId) - nodeposMat.row(startId);
        torqueImbal += nodetorqMat.row(endId) - nodetorqMat.row(startId);

        // subsequent torq from force addition                
        for (int j = i+1; j < n_force; j++) {
            fVec = disturbed_sections_[j].avg_force;
            torqueImbal += distVec.cross(fVec);
        }

        // torq from weight
        if (w_perpiece.norm() > 1e-10) {
            torqueImbal += distVec.cross((nv-endId+1)*w_perpiece);
            // add the internal weight torques
            // Create matrix of position differences between edge midpoints and start position
            Eigen::MatrixXd edgeMidpoints(endId - startId, 3);
            for (int i = 0; i < endId - startId; i++) {
                edgeMidpoints.row(i) = (nodeposMat.row(startId + i + 1) + nodeposMat.row(startId + i)) / 2.0 - nodeposMat.row(startId);
                Eigen::Vector3d edgeVec = edgeMidpoints.row(i).transpose();
                torqueImbal += edgeVec.cross(w_perpiece);
            }
        }

        // finding relpos along the wire
        fVec = disturbed_sections_[i].avg_force;
        force_pos = findTorqueBalancePoint(startId, endId, fVec, -torqueImbal);

        // set to force_sections
        force_sections[i].force_pos = force_pos;
        force_sections[i].force = disturbed_sections_[i].avg_force;
        force_sections[i].start_pos = nodeposMat.row(startId);
        force_sections[i].end_pos = nodeposMat.row(endId);
    }
}

bool DLO_s2f::hasParallelVectors(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const Eigen::Vector3d& v3, double threshold)
{
    // Normalize the vectors
    Eigen::Vector3d n1 = v1.normalized();
    Eigen::Vector3d n2 = v2.normalized();
    Eigen::Vector3d n3 = v3.normalized();

    // Check all possible pairs
    double dot12 = std::abs(n1.dot(n2));
    double dot13 = std::abs(n1.dot(n3));
    double dot23 = std::abs(n2.dot(n3));

    // Return true if any pair is parallel (dot product close to 1)
    return (dot12 > threshold || dot13 > threshold || dot23 > threshold);
}

Eigen::Vector3d DLO_s2f::findTorqueBalancePoint(int startId, int endId, const Eigen::Vector3d& force, const Eigen::Vector3d& targetTorque) {
    // finds Closest point to balance torque
    // Step 1: Find relpos that satisfies torqueImbal = relpos.cross(fVec)
    // and is orthogonal to both torqueImbal and fVec
    Eigen::Vector3d relpos = force.cross(targetTorque) / force.squaredNorm();
    
    // Step 2: Find the closest point on the wire section to the line
    // parallel to fVec passing through startPos + relpos
    Eigen::Vector3d startPos = nodeposMat.row(startId);
    Eigen::Vector3d linePoint = startPos + relpos;
    
    double minDist = std::numeric_limits<double>::max();
    Eigen::Vector3d closestPoint;
    
    // Check each segment of the wire
    for (int i = startId; i < endId; i++) {
        Eigen::Vector3d p1 = nodeposMat.row(i);
        Eigen::Vector3d p2 = nodeposMat.row(i + 1);
        
        // Vector from line point to segment start
        Eigen::Vector3d v = p1 - linePoint;
        
        // Project v onto force direction
        double t = v.dot(force) / force.squaredNorm();
        
        // Point on the line
        Eigen::Vector3d linePoint_t = linePoint + t * force;
        
        // Vector from line point to segment
        Eigen::Vector3d segmentVec = p2 - p1;
        double segmentLength = segmentVec.norm();
        segmentVec.normalize();
        
        // Vector from line point to segment start
        Eigen::Vector3d toSegment = linePoint_t - p1;
        
        // Project toSegment onto segment direction
        double s = toSegment.dot(segmentVec);
        
        // Clamp s to segment bounds
        s = std::max(0.0, std::min(s, segmentLength));
        
        // Calculate closest point on segment
        Eigen::Vector3d pointOnSegment = p1 + s * segmentVec;
        
        // Calculate distance to line
        double dist = (pointOnSegment - linePoint_t).norm();
        
        if (dist < minDist) {
            minDist = dist;
            closestPoint = pointOnSegment;
        }
    }
    
    return closestPoint;
}