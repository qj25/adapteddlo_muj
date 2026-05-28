%module MassSpring
%{
#define SWIG_FILE_WITH_INIT
#include "MassSpring.h"
%}

%include "../dlo_cpp/numpy.i"

%init %{
import_array();
%}

%apply (int DIM1, double* IN_ARRAY1) {
    (int dim_nq, double* neutral_quat),
    (int dim_cq, double* current_quat),
    (int dim_nt, double* node_torque)
};

%include "MassSpring.h"
