%{
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
%}

%include <typemaps.i>

// Handle Eigen::Vector3d
%typemap(in) Eigen::Vector3d (Eigen::Vector3d temp) {
    if (!PyArray_Check($input)) {
        SWIG_exception_fail(SWIG_TypeError, "numpy array expected");
    }
    PyArrayObject* array = (PyArrayObject*)$input;
    if (PyArray_NDIM(array) != 1 || PyArray_DIM(array, 0) != 3) {
        SWIG_exception_fail(SWIG_ValueError, "array must be 1D with 3 elements");
    }
    double* array_data = (double*)PyArray_DATA(array);
    temp = Eigen::Map<Eigen::Vector3d>(array_data);
    $1 = temp;
}

%typemap(out) Eigen::Vector3d {
    npy_intp dims[1] = {3};
    PyObject* array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    double* array_data = (double*)PyArray_DATA((PyArrayObject*)array);
    Eigen::Map<Eigen::Vector3d>(array_data) = $1;
    $result = array;
}

// Handle Eigen::MatrixXd
%typemap(in) Eigen::MatrixXd (Eigen::MatrixXd temp) {
    if (!PyArray_Check($input)) {
        SWIG_exception_fail(SWIG_TypeError, "numpy array expected");
    }
    PyArrayObject* array = (PyArrayObject*)$input;
    if (PyArray_NDIM(array) != 2) {
        SWIG_exception_fail(SWIG_ValueError, "array must be 2D");
    }
    npy_intp rows = PyArray_DIM(array, 0);
    npy_intp cols = PyArray_DIM(array, 1);
    temp.resize(rows, cols);
    double* array_data = (double*)PyArray_DATA(array);
    temp = Eigen::Map<Eigen::MatrixXd>(array_data, rows, cols);
    $1 = temp;
}

%typemap(out) Eigen::MatrixXd {
    npy_intp dims[2] = {$1.rows(), $1.cols()};
    PyObject* array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    double* array_data = (double*)PyArray_DATA((PyArrayObject*)array);
    Eigen::Map<Eigen::MatrixXd>(array_data, $1.rows(), $1.cols()) = $1;
    $result = array;
}

// Handle Eigen::Vector4d
%typemap(in) Eigen::Vector4d (Eigen::Vector4d temp) {
    if (!PyArray_Check($input)) {
        SWIG_exception_fail(SWIG_TypeError, "numpy array expected");
    }
    PyArrayObject* array = (PyArrayObject*)$input;
    if (PyArray_NDIM(array) != 1 || PyArray_DIM(array, 0) != 4) {
        SWIG_exception_fail(SWIG_ValueError, "array must be 1D with 4 elements");
    }
    double* array_data = (double*)PyArray_DATA(array);
    temp = Eigen::Map<Eigen::Vector4d>(array_data);
    $1 = temp;
}

%typemap(out) Eigen::Vector4d {
    npy_intp dims[1] = {4};
    PyObject* array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    double* array_data = (double*)PyArray_DATA((PyArrayObject*)array);
    Eigen::Map<Eigen::Vector4d>(array_data) = $1;
    $result = array;
}

// Handle const references
%apply Eigen::Vector3d { const Eigen::Vector3d& };
%apply Eigen::MatrixXd { const Eigen::MatrixXd& };
%apply Eigen::Vector4d { const Eigen::Vector4d& };

// Handle pointers
%apply Eigen::Vector3d { Eigen::Vector3d* };
%apply Eigen::MatrixXd { Eigen::MatrixXd* };
%apply Eigen::Vector4d { Eigen::Vector4d* }; 