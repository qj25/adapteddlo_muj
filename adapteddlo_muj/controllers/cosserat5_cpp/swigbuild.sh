#!/bin/bash

NUMPY_INCLUDE_PATH=$(python3 -c "import numpy; print(numpy.get_include())")
PYTHON_INCLUDE_PATH=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")
EIGEN_INC="${EIGEN_INCLUDE:-$HOME/eigen}"

swig -c++ -python -o RodCosserat5_wrap.cpp RodCosserat5.i
g++ -c RodCosserat5.cpp RodCosserat5_wrap.cpp Cosserat5_utils.cpp \
    -I"${EIGEN_INC}" -I/usr/include/eigen3 -I"$NUMPY_INCLUDE_PATH" -I"$PYTHON_INCLUDE_PATH" -fPIC -std=c++14 -O2
g++ -shared RodCosserat5.o RodCosserat5_wrap.o Cosserat5_utils.o -o _RodCosserat5.so -fPIC
python3 -c "import _RodCosserat5"
