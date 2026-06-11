#ifndef COSSERAT5_OBJ_H
#define COSSERAT5_OBJ_H

#include <vector>
#include "Eigen/Core"

struct Vecnodes
{
public:
    Eigen::Vector3d pos;
    Eigen::Vector3d force;
    Eigen::Vector3d torq;
    Eigen::Vector4d quat;

    double phi_i;
    double k;
    Eigen::Vector3d kb;

    // JTill2017 material-frame curvature u and rest u*
    Eigen::Vector3d u;
    Eigen::Vector3d u_star;

    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> nabkb;
    Eigen::Matrix3d nabpsi;
};

struct SegEdges
{
public:
    Eigen::Vector3d e;
    Eigen::Matrix3d bf;
    double theta;
    double theta_star;
    double e_bar;
    double l_bar;
};

#endif
