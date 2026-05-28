#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUMPY_INCLUDE_PATH=$(python3 -c "import numpy; print(numpy.get_include())")
PYTHON_INCLUDE_PATH=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")

EIGEN_INCLUDE="${EIGEN_INCLUDE:-$HOME/eigen}"

PBD_VENDOR="vendor/pbd"
PBD_POS="${PBD_VENDOR}/PositionBasedDynamics"

swig -c++ -python -o RodXpbd_wrap.cpp RodXpbd.i

g++ -c RodXpbd.cpp RodXpbd_wrap.cpp \
  "${PBD_POS}/MathFunctions.cpp" \
  "${PBD_POS}/PositionBasedElasticRods.cpp" \
  -I. \
  -I"${PBD_VENDOR}" \
  -I"${PBD_POS}" \
  -I"${EIGEN_INCLUDE}" \
  -I"${NUMPY_INCLUDE_PATH}" \
  -I"${PYTHON_INCLUDE_PATH}" \
  -fPIC -std=c++14 -O2

g++ -shared RodXpbd.o RodXpbd_wrap.o MathFunctions.o PositionBasedElasticRods.o \
  -o _RodXpbd.so -fPIC

python3 -c "import RodXpbd"
