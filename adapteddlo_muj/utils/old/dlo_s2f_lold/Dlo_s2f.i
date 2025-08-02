%module Dlo_s2f
%{
#define SWIG_FILE_WITH_INIT
// // #include <iostream>
#include "Ds2f_obj.h"
// #include "Ds2f_utils.h"
#include "Dlo_s2f.h"
%}

%include "numpy.i"
%include "std_vector.i"
%include "std_string.i"
// %include "Eigen.i"

%init %{
import_array();
%}

%apply (int DIM1, double* IN_ARRAY1) {
    (int dim_nt, double* node_torq),
    (int dim_np, double* node_pos),
    (int dim_nq, double* node_quat)
};
// %apply (int DIM1, double* ARGOUT_ARRAY1) {(int dim_nf, double* node_force)};

// Convert Eigen::Vector3d to/from numpy array
%typemap(in) Eigen::Vector3d {
    if (!PyArray_Check($input)) {
        SWIG_exception_fail(SWIG_TypeError, "numpy array expected");
    }
    PyArrayObject* array = (PyArrayObject*)$input;
    if (PyArray_NDIM(array) != 1 || PyArray_DIM(array, 0) != 3) {
        SWIG_exception_fail(SWIG_ValueError, "array must be 1D with 3 elements");
    }
    $1 = Eigen::Map<Eigen::Vector3d>((double*)PyArray_DATA(array));
}

%typemap(out) Eigen::Vector3d {
    npy_intp dims[1] = {3};
    PyObject* array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    Eigen::Map<Eigen::Vector3d>((double*)PyArray_DATA((PyArrayObject*)array)) = $1;
    $result = array;
}

// Include the header files
%include "Ds2f_obj.h"
%include "Dlo_s2f.h"

// Declare vector of ForceSection
%template(ForceSectionVector) std::vector<ForceSection>;

// Extend ForceSection with Python helper functions
%extend ForceSection {
    %pythoncode %{
        def __str__(self):
            return f"ForceSection(force={self.force}, force_pos={self.force_pos}, start_pos={self.start_pos}, end_pos={self.end_pos})"
            
        def get_force(self):
            return self.force
            
        def get_torque(self):
            return self.torque

        def get_force_pos(self):
            return self.force_pos

        def get_start_pos(self):
            return self.start_pos

        def get_end_pos(self):
            return self.end_pos
    %}
}