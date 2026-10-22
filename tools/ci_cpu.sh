#!/usr/bin/env bash
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
build_dir=${1:-"$source_dir/build-cpu"}
if (( $# > 1 )); then
    echo "Usage: $0 [build-directory]" >&2
    exit 2
fi
cmake -S "$source_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release -DSTRIX_TRELLIS_ENABLE_HIP=OFF \
    -DSTRIX_TRELLIS_BUILD_TESTING=ON
cmake --build "$build_dir" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
ctest --test-dir "$build_dir" --output-on-failure

PYTHONPATH="$source_dir/python${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON:-python3}" -B -m unittest discover -s "$source_dir/tests" -v
