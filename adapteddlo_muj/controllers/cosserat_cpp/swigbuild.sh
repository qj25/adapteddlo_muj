#!/bin/bash

NUMPY_INCLUDE_PATH=$(python3 -c "import numpy; print(numpy.get_include())")
PYTHON_INCLUDE_PATH=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")
EIGEN_INCLUDE="${EIGEN_INCLUDE:-$HOME/eigen}"

swig -c++ -python -o RodCosserat_wrap.cpp RodCosserat.i
g++ -c RodCosserat.cpp RodCosserat_wrap.cpp \
    -I"${EIGEN_INCLUDE}" \
    -I/usr/include/eigen3 \
    -I"${NUMPY_INCLUDE_PATH}" \
    -I"${PYTHON_INCLUDE_PATH}" \
    -fPIC -std=c++17 -O2
g++ -shared RodCosserat.o RodCosserat_wrap.o -o _RodCosserat.so -fPIC
python3 -c "import _RodCosserat"
