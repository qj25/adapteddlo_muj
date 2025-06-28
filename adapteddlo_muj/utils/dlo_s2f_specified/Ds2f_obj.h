#ifndef DS2FOBJ_H
#define DS2FOBJ_H

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
    Eigen::Vector4d quat;
    Eigen::Vector3d torq;
    // global torque, converted in DLO_s2f::calculateLocal2GlobalTorq
};

struct Extforces
{
public:
    // SegEdges(){}
    Eigen::Vector3d pos;

    int old_order; // original input index of the force positions
    int link_id; // 'nearest link': index of the link on which force is applied
    Eigen::Vector3d rel_pos;
    // rel_pos wrt node[id].pos such that id = link_id
};

#endif
