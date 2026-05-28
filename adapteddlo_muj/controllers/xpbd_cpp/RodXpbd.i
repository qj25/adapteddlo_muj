%module RodXpbd
%{
#define SWIG_FILE_WITH_INIT
#include "RodXpbd.h"
%}

%include "../dlo_cpp/numpy.i"

%init %{
import_array();
%}

%apply (int DIM1, double* IN_ARRAY1) {
    (int dim_x, const double* rest_x),
    (int dim_q, const double* rest_quat),
    (int dim_x, const double* x),
    (int dim_q, const double* quat),
    (int dim_m, const double* inv_mass),
    (int dim_i, const double* inv_inertia_w),
    (int dim_f, double* force_out),
    (int dim_t, double* torque_out)
};

%include "RodXpbd.h"
