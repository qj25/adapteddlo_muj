#!/bin/bash

# export RLENVPATH=$(which python)
# RLENVPATH=$(echo "$RLENVPATH" | rev | cut -d'/' -f3- | rev)
NUMPY_INCLUDE_PATH=$(python3 -c "import numpy; print(numpy.get_include())")
PYTHON_INCLUDE_PATH=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")
swig -c++ -python -o Dlo_s2f_wrap.cpp Dlo_s2f.i
g++ -c Dlo_s2f.cpp Dlo_s2f_wrap.cpp Ds2f_utils.cpp -I$HOME/eigen -I$NUMPY_INCLUDE_PATH -I$PYTHON_INCLUDE_PATH -fPIC -std=c++14 -O2
g++ -shared Dlo_s2f.o Dlo_s2f_wrap.o Ds2f_utils.o -o _Dlo_s2f.so -fPIC
python -c "import _Dlo_s2f"

