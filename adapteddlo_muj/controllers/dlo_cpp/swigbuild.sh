#!/bin/bash

# export RLENVPATH=$(which python)
# RLENVPATH=$(echo "$RLENVPATH" | rev | cut -d'/' -f3- | rev)
NUMPY_INCLUDE_PATH=$(python3 -c "import numpy; print(numpy.get_include())")
PYTHON_INCLUDE_PATH=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")
swig -c++ -python -o Dlo_iso_wrap.cpp Dlo_iso.i
g++ -c Dlo_iso.cpp Dlo_iso_wrap.cpp Dlo_utils.cpp -I$HOME/eigen -I$NUMPY_INCLUDE_PATH -I$PYTHON_INCLUDE_PATH -fPIC -std=c++14 -O2
g++ -shared Dlo_iso.o Dlo_iso_wrap.o Dlo_utils.o -o _Dlo_iso.so -fPIC
python -c "import _Dlo_iso"

