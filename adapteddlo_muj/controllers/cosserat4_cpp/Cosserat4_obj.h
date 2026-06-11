#ifndef COSSERAT4OBJ_H
#define COSSERAT4OBJ_H

#include <vector>
#include "Eigen/Core"
#include "Eigen/Geometry"

struct Vecnodes
{
public:
    Eigen::Vector3d pos;
    Eigen::Vector3d force;
    Eigen::Vector3d force_sub;

    Eigen::Vector3d torq;
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
    Eigen::Vector3d e;
    Eigen::Matrix3d bf;
    double theta;
    double p_thetaloc;
    double theta_displace;
    Eigen::Quaterniond qe_o2m_loc;
    double e_bar;
    double l_bar;
};

#endif
