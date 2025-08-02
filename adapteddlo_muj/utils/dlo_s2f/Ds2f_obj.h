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

struct Section {
    int start_idx;
    int end_idx;
    int n_force;
    Eigen::Vector3d avg_force;
    std::vector<Eigen::Vector3d> indiv_forces;
    std::vector<double> c3;
    Section(int start, int end) : start_idx(start), end_idx(end), avg_force(Eigen::Vector3d::Zero()) {}
};

struct ForceSection {
    Eigen::Vector3d start_pos;
    Eigen::Vector3d end_pos;
    Eigen::Vector3d force;
    Eigen::Vector3d torque;
    Eigen::Vector3d force_pos;
    
    ForceSection() : 
        start_pos(Eigen::Vector3d::Zero()),
        end_pos(Eigen::Vector3d::Zero()),
        force(Eigen::Vector3d::Zero()),
        torque(Eigen::Vector3d::Zero()),
        force_pos(Eigen::Vector3d::Zero()) {}

    // Getters for Python
    Eigen::Vector3d get_start_pos() const { return start_pos; }
    Eigen::Vector3d get_end_pos() const { return end_pos; }
    Eigen::Vector3d get_force() const { return force; }
    Eigen::Vector3d get_torque() const { return torque; }
    Eigen::Vector3d get_force_pos() const { return force_pos; }

    // Setters for Python
    void set_start_pos(const Eigen::Vector3d& pos) { start_pos = pos; }
    void set_end_pos(const Eigen::Vector3d& pos) { end_pos = pos; }
    void set_force(const Eigen::Vector3d& f) { force = f; }
    void set_torque(const Eigen::Vector3d& t) { torque = t; }
    void set_force_pos(const Eigen::Vector3d& r) { force_pos = r; }
};

#endif