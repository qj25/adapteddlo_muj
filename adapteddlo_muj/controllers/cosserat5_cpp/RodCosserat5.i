%module RodCosserat5
%{
#define SWIG_FILE_WITH_INIT
#include "Cosserat5_obj.h"
#include "Cosserat5_utils.h"
#include "RodCosserat5.h"
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
    (int dim_nq, double* node_quat),
    (int dim_qo2m, double* q_o2m),
    (int dim_mato, double* mat_o),
    (int dim_matres, double* mat_res),
    (int dim_v1, double *v1),
    (int dim_v2, double *v2),
    (int dim_va, double *va)
};

%include "RodCosserat5.h"
