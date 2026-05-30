#!/usr/bin/env bash
# Build all C++ SWIG rope backends under adapteddlo_muj/controllers/*_cpp.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTROLLERS_DIR="${REPO_ROOT}/adapteddlo_muj/controllers"

# Backends in dependency-free build order (matches README).
ALL_BACKENDS=(dlo_cpp massspring_cpp xpbd_cpp geds_cpp)

usage() {
    cat <<'EOF'
Usage: build_rope_backends.sh [BACKEND ...]

Build C++ SWIG extensions for rope controllers. With no arguments, builds all
backends: dlo_cpp, massspring_cpp, xpbd_cpp, geds_cpp.

Environment:
  EIGEN_INCLUDE   Path to Eigen headers (default: $HOME/eigen)

Examples:
  bash scripts/build_rope_backends.sh
  bash scripts/build_rope_backends.sh geds_cpp xpbd_cpp
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

backends=()
if (($# > 0)); then
    for name in "$@"; do
        if [[ "${name}" != *_cpp ]]; then
            name="${name}_cpp"
        fi
        if [[ ! -f "${CONTROLLERS_DIR}/${name}/swigbuild.sh" ]]; then
            echo "error: unknown backend '${name}' (no swigbuild.sh)" >&2
            exit 1
        fi
        backends+=("${name}")
    done
else
    backends=("${ALL_BACKENDS[@]}")
fi

if [[ -z "${EIGEN_INCLUDE:-}" && ! -d "${HOME}/eigen" ]]; then
    echo "warning: EIGEN_INCLUDE is unset and ${HOME}/eigen not found" >&2
fi

echo "Building ${#backends[@]} rope backend(s) in ${CONTROLLERS_DIR}"
for name in "${backends[@]}"; do
    dir="${CONTROLLERS_DIR}/${name}"
    echo "==> ${name}"
    (cd "${dir}" && bash swigbuild.sh)
done
echo "Done."
