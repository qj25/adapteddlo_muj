#!/bin/bash

NUMPY_INCLUDE_PATH=$(python3 -c "import numpy; print(numpy.get_include())")
PYTHON_INCLUDE_PATH=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")
swig -c++ -python -o RodCosserat4_wrap.cpp Cosserat4.i
g++ -c cosserat4.cpp RodCosserat4_wrap.cpp Cosserat4_utils.cpp -I$HOME/eigen -I$NUMPY_INCLUDE_PATH -I$PYTHON_INCLUDE_PATH -fPIC -std=c++14 -O2
g++ -shared cosserat4.o RodCosserat4_wrap.o Cosserat4_utils.o -o _RodCosserat4.so -fPIC
python3 -c "import _RodCosserat4"
