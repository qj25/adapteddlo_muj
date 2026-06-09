#!/bin/bash

NUMPY_INCLUDE_PATH=$(python3 -c "import numpy; print(numpy.get_include())")
PYTHON_INCLUDE_PATH=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")
EIGEN_INC="${EIGEN_INCLUDE:-$HOME/eigen}"

swig -c++ -python -o RodCosserat3_wrap.cpp RodCosserat3.i
g++ -c RodCosserat3.cpp RodCosserat3_wrap.cpp Cosserat3_utils.cpp \
    -I"${EIGEN_INC}" -I"$NUMPY_INCLUDE_PATH" -I"$PYTHON_INCLUDE_PATH" -fPIC -std=c++14 -O2
g++ -shared RodCosserat3.o RodCosserat3_wrap.o Cosserat3_utils.o -o _RodCosserat3.so -fPIC
python3 -c "import _RodCosserat3"
