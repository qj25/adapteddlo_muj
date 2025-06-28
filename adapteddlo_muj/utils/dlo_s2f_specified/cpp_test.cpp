#include "Der_obj.h"
#include "Der_utils.h"
#include "Der_iso3.h"
#include <iostream>
#include <chrono>
#include <typeinfo>

#include <Eigen/Core>
#include <Eigen/Dense>

int main()
{
    // Eigen::Matrix3d bf1, bf2;
    // double a1, a2;
    // Eigen::Vector3d v1, v2, v3;

    // DerUtils derutils;
    // v1 << 1., 0., 0.;
    // v2 << 0., 1., 0.;
    // bf1.row(0) << 1., 0., 0.;
    // bf1.row(1) << 0., 1., 0.;
    // bf1.row(2) << 0., 0., 1.;
    // bf2 = bf1;
    // bf2.row(0) << 0., 0., 2.;
    // std::cout << bf2(0,2) << std::endl;
    // std::cout << bf1(0,2) << std::endl;

    // bf2.row(0) = DerUtils::rotateVector3(bf1.row(0), bf1.row(1), M_PI/8);
    // std::cout << bf2(0,2) << std::endl;
    // a1 = DerUtils::calculateAngleBetween(bf1.row(0), bf1.row(1));
    // a2 = DerUtils::calculateAngleBetween2(bf1.row(0), bf1.row(1), bf1.row(2));
    // std::cout << a1 << std::endl;
    // std::cout << a2 << std::endl;
    // bf1.row(2) = bf1.row(0).cross(-bf1.row(1));
    // a1 = bf1.row(0).dot(bf1.row(1));
    // std::cout << bf1.row(2) << std::endl;
    // std::cout << a1 << std::endl;


    // std::vector <Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > node_pos;
    // Eigen::Vector3d curr_pos;
    // Eigen::Matrix3d bf0sim;
    std::vector <double> node_pos;
    std::vector <double> node_force;
    std::vector <double> bf0sim;

    double theta_n;
    double overall_rot;
    double r_len, sec_len;
    int n_nodes;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == j) {bf0sim.push_back(1.);}
            else {bf0sim.push_back(0.);}
        }
    }
    // bf0sim << 1., 0., 0.,
    //     0., 1., 0.,
    //     0., 0., 1.;
    theta_n = 0.;
    overall_rot = 0.;
    
    int n_test;
    n_test = 1000;

    n_nodes = 5;
    r_len = 1.;
    sec_len = r_len / (n_nodes-1);

    for (int i = 0; i < n_nodes; i++) {
        for (int j = 0; j < 3; j++) {
            // curr_pos << (sec_len*(i)), 0., 0.;
            if (j == 0) {node_pos.push_back(sec_len*i);}
            else {node_pos.push_back(0.);}
            node_force.push_back(sec_len*i);
            // std::cout << curr_pos << std::endl;
        }
    }

    // node_pos[2](1) = -0.01;
    node_pos[7] = -0.01;

    // std::cout << node_pos.size() << std::endl;
    
    Der_iso3 DerSampleRod(
        node_pos.size(), &node_pos[0], 
        bf0sim.size(), &bf0sim[0],
        theta_n, overall_rot
    );

    using namespace std::literals::chrono_literals;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_test; i++) {
        DerSampleRod.updateVars(
            node_pos.size(), &node_pos[0], 
            bf0sim.size(), &bf0sim[0]
        );
        DerSampleRod.calculateCenterlineF2(node_force.size(), &node_force[0]);
    }
    // if (__cplusplus == 201103L) std::cout << "y" << std::endl;
    // else std::cout << "n" << std::endl;
    
    auto stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> duration;
    duration = stop - start;
    std::cout << duration.count() << "s" << std::endl;

    for (int i = 0; i < n_nodes; i++) {
        std::cout << 999999999999999999 << std::endl;
        std::cout << i << std::endl;
        // for (int j = 0; j < 3; j++) {
        //     std::cout << DerSampleRod.nodes[i].nabkb[j] << std::endl;
        //     std::cout << 111111111111 << std::endl;
        // }
        std::cout << DerSampleRod.nodes[i].force << std::endl;
        // std::cout << DerSampleRod.edges[i].theta << std::endl;
    }

    return 0;
}

// g++ -g cpp_test.cpp Der_utils.cpp Der_iso3.cpp -o main -std=c++14 -O2 -lpthread