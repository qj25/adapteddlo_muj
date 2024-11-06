#!/bin/bash

export RLENVPATH=$(which python)
RLENVPATH=$(echo "$RLENVPATH" | rev | cut -d'/' -f3- | rev)
swig -c++ -python -o Dlo_iso_wrap.cpp Dlo_iso.i
g++ -c Dlo_iso.cpp Dlo_iso_wrap.cpp Dlo_utils.cpp -I$HOME/eigen -I$RLENVPATH/lib/python3.8/site-packages/numpy/core/include -I$RLENVPATH/include/python3.8 -fPIC -std=c++14 -O2
g++ -shared Dlo_iso.o Dlo_iso_wrap.o Dlo_utils.o -o _Dlo_iso.so -fPIC
python -c "import _Dlo_iso"

