#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

NUMPY_INCLUDE_PATH=$(python3 -c "import numpy; print(numpy.get_include())")
PYTHON_INCLUDE_PATH=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")

GLM_INCLUDE="${SCRIPT_DIR}/vendor/glm"
VENDOR_INCLUDES=(
    -I"${GLM_INCLUDE}"
    -I"${SCRIPT_DIR}/vendor/kitten"
    -I"${SCRIPT_DIR}/vendor/cosserat"
)

if [ ! -f "${GLM_INCLUDE}/glm/glm.hpp" ]; then
    echo "GLM not found. Run: git clone --depth 1 --branch 1.0.1 https://github.com/g-truc/glm.git vendor/glm"
    exit 1
fi

swig -c++ -python -o RodCosserat2_wrap.cpp RodCosserat2.i

SOURCES=(
    RodCosserat2.cpp
    vendor/cosserat/init.cpp
    vendor/cosserat/vbd.cpp
    vendor/cosserat/lambda.cpp
    RodCosserat2_wrap.cpp
)

g++ -c "${SOURCES[@]}" \
    "${VENDOR_INCLUDES[@]}" \
    -I"${NUMPY_INCLUDE_PATH}" \
    -I"${PYTHON_INCLUDE_PATH}" \
    -fPIC -std=c++17 -O2

OBJECTS=(
    RodCosserat2.o
    init.o
    vbd.o
    lambda.o
    RodCosserat2_wrap.o
)

g++ -shared "${OBJECTS[@]}" -o _RodCosserat2.so -fPIC
python3 -c "import _RodCosserat2"
echo "Built _RodCosserat2.so"
