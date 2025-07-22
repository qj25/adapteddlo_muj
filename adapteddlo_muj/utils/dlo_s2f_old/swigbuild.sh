#!/bin/bash

# Find Eigen include path
find_eigen() {
    # Common locations to search for Eigen
    local locations=(
        "/usr/include/eigen3"
        "/usr/local/include/eigen3"
        "/opt/eigen3/include/eigen3"
        "$HOME/eigen"
        "$HOME/.local/include/eigen3"
        "/opt/eigen3/include/eigen3"
    )
    
    # Check each location
    for loc in "${locations[@]}"; do
        if [ -d "$loc" ]; then
            echo "$loc"
            return 0
        fi
    done

    # If not found in common locations, try pkg-config
    if command -v pkg-config >/dev/null 2>&1; then
        local pkg_path=$(pkg-config --cflags eigen3 2>/dev/null | sed 's/-I//')
        if [ ! -z "$pkg_path" ]; then
            echo "$pkg_path"
            return 0
        fi
    fi

    echo "Error: Could not find Eigen installation" >&2
    return 1
}

EIGEN_INCLUDE_PATH=$(find_eigen)
if [ $? -ne 0 ]; then
    exit 1
fi

NUMPY_INCLUDE_PATH=$(python3 -c "import numpy; print(numpy.get_include())")
PYTHON_INCLUDE_PATH=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")

swig -c++ -python -o Dlo_s2f_wrap.cpp Dlo_s2f.i
g++ -c Dlo_s2f.cpp Dlo_s2f_wrap.cpp Ds2f_utils.cpp -I"$EIGEN_INCLUDE_PATH" -I"$NUMPY_INCLUDE_PATH" -I"$PYTHON_INCLUDE_PATH" -fPIC -std=c++14 # -O2
g++ -shared Dlo_s2f.o Dlo_s2f_wrap.o Ds2f_utils.o -o _Dlo_s2f.so -fPIC
python -c "import _Dlo_s2f"

