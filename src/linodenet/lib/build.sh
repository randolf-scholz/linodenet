#!/usr/bin/env bash
mkdir -p build
cd build || exit
PROJECT_DIR=$(git rev-parse --show-toplevel | xargs echo -n)
echo "PROJECT_DIR: ${PROJECT_DIR}"

# if LIBTORCH_DIR exists, use it, else fallback to installed torch
if [ -d "${PROJECT_DIR}/libtorch" ]; then
    LIBTORCH_DIR="${PROJECT_DIR}/libtorch"
else
    LIBTORCH_DIR="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"
fi
# NOTE: cxx11 ABI throws error messages, use pre-cxx11 ABI

echo "LIBTORCH_DIR: ${LIBTORCH_DIR}"
cmake -DCMAKE_PREFIX_PATH="${LIBTORCH_DIR}" ..
# cmake -DCMAKE_PREFIX_PATH="${LIBTORCH_DIR}" -DCMAKE_CXX_CLANG_TIDY="clang-tidy;-header-filter=$(realpath ..)" ..
make -j
python -c "import linodenet.lib"
