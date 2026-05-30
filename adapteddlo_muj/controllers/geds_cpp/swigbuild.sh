#!/bin/bash

NUMPY_INCLUDE_PATH=$(python3 -c "import numpy; print(numpy.get_include())")
PYTHON_INCLUDE_PATH=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")
EIGEN_INCLUDE="${EIGEN_INCLUDE:-$HOME/eigen}"

swig -c++ -python -o RodGeds_wrap.cpp RodGeds.i
g++ -c RodGeds.cpp CatmullRom.cpp MinimalFrame.cpp RodGeds_wrap.cpp \
    -I"${EIGEN_INCLUDE}" \
    -I"${NUMPY_INCLUDE_PATH}" \
    -I"${PYTHON_INCLUDE_PATH}" \
    -fPIC -std=c++14 -O2
g++ -shared RodGeds.o CatmullRom.o MinimalFrame.o RodGeds_wrap.o -o _RodGeds.so -fPIC
python3 -c "import _RodGeds"
