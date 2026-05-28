#!/bin/bash

NUMPY_INCLUDE_PATH=$(python3 -c "import numpy; print(numpy.get_include())")
PYTHON_INCLUDE_PATH=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")

swig -c++ -python -o MassSpring_wrap.cpp MassSpring.i
g++ -c MassSpring.cpp MassSpring_wrap.cpp -I$HOME/eigen -I$NUMPY_INCLUDE_PATH -I$PYTHON_INCLUDE_PATH -fPIC -std=c++14 -O2
g++ -shared MassSpring.o MassSpring_wrap.o -o _MassSpring.so -fPIC
python3 -c "import _MassSpring"
