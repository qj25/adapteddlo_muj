#ifndef DLOOBJ_H
#define DLOOBJ_H

#include <vector>
#include "Eigen/Core"

// struct Frame_rot
// {
// public:
//     // Frame_rot(){}
//     Eigen::Vector3d x;
//     Eigen::Vector3d y;
//     Eigen::Vector3d z;
// };

struct Vecnodes
{
public:
    // Vecnodes(){}
    Eigen::Vector3d pos;
    // Eigen::Vector3d vel;
    Eigen::Vector3d force;
    Eigen::Vector3d force_sub;

    Eigen::Vector3d torq;
    // Eigen::Vector3d torq_global;
    Eigen::Vector4d quat;

    double phi_i;
    double k;
    Eigen::Vector3d kb;

    std::vector <Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> > nabkb;
    Eigen::Matrix3d nabpsi;
};

struct SegEdges
{
public:
    // SegEdges(){}
    Eigen::Vector3d e;
    Eigen::Matrix3d bf;
    double theta;
    double e_bar;
    double l_bar;
};

#endif
