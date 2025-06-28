%module Dlo_s2f
%{
#define SWIG_FILE_WITH_INIT
// #include <iostream>
// #include "Ds2f_obj.h"
// #include "Ds2f_utils.h"
#include "Dlo_s2f.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (int DIM1, double* IN_ARRAY1) {
    (int dim_efp, double* ext_forcepos),
    (int dim_ef, double* ext_force),
    (int dim_et, double* ext_torq),
    (int dim_nt, double* node_torq),
    (int dim_np, double* node_pos),
    (int dim_nq, double* node_quat)
};
// %apply (int DIM1, double* ARGOUT_ARRAY1) {(int dim_nf, double* node_force)};


%include "Dlo_s2f.h"