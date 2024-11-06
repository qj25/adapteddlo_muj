%module Dlo_iso
%{
#define SWIG_FILE_WITH_INIT
// #include <iostream>
#include "Dlo_obj.h"
#include "Dlo_utils.h"
#include "Dlo_iso.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (int DIM1, double* IN_ARRAY1) {
    (int dim_np, double* node_pos),
    (int dim_bf0, double* bf0sim),
    (int dim_bfe, double* bfesim),
    (int dim_nf, double* node_force),
    (int dim_nt, double* node_torq),
    // (int dim_ntg, double* node_torq_global),
    (int dim_nq, double* node_quat),
    (int dim_qo2m, double* q_o2m),
    // (int dim_qm2o, double* q_m2o),
    (int dim_mato, double* mat_o),
    (int dim_matres, double* mat_res),
    (int dim_v1, double *v1),
    (int dim_v2, double *v2),
    (int dim_va, double *va)
};
// %apply (int DIM1, double* ARGOUT_ARRAY1) {(int dim_nf, double* node_force)};


%include "Dlo_iso.h"
// %include "Dlo_obj.h"
// %include "Dlo_utils.h"


// class DLO_iso
// {
// public:
//     DLO_iso(
//         int dim_np,
//         double *node_pos,
//         int dim_bf0,
//         double *bf0sim,
//         const double theta_n,
//         const double overall_rot
//     );

//     ~DLO_iso();

//     void updateVars(
//         int dim_np,
//         double *node_pos,
//         int dim_bf0,
//         double *bf0sim
//     );  //

//     void calculateCenterlineF2(int dim_nf, double *node_force);
// };

// %include "Dlo_obj.h"
// %include "Dlo_utils.h"


// #include <Eigen/Dense>
// #include <Eigen/Core>

// extern DLO_iso(
//     int dim_np,
//     double *node_pos,
//     int dim_bf0,
//     double *bf0sim,
//     const double theta_n,
//     const double overall_rot
// );

// extern DLO_iso::updateVars(
//         int dim_np,
//         double *node_pos,
//         int dim_bf0,
//         double *bf0sim
//     );  //

// extern DLO_iso::calculateCenterlineF2(int dim_nf, double *node_force);

/*
    python3-config --cflags
    swig -c++ -python -o Dlo_iso_wrap.cpp Dlo_iso.i

    g++ -c -fpic Dlo_iso.cpp Dlo_utils.cpp -std=c++14
    g++ -c -fpic Dlo_iso_wrap.cpp -I/home/qj/anaconda3/envs/rlenv/include/python3.6m -std=c++14
    g++ -shared Dlo_iso.o Dlo_utils.o Dlo_iso_wrap.o _Dlo_iso.so

    g++ -c Dlo_iso.cpp Dlo_iso_wrap.cpp -I/home/qj/anaconda3/envs/rlenv/include/python3.6m -fPIC -std=c++14
    ld -shared Dlo_iso.o Dlo_iso_wrap.o -o _Dlo_iso.so -fPIC

    g++ -Wl,--gc-sections -fPIC -shared -lstdc++ Dlo_iso.o Dlo_iso_wrap.o -o _Dlo_iso.so
*/