#!/usr/bin/env bash
set -e
# prepend correct CUDA version
export PATH=/usr/local/cuda-11.7/bin:$PATH

# determine project dir
PROJECT_DIR=$(git rev-parse --show-toplevel | xargs echo -n)
echo "PROJECT_DIR: ${PROJECT_DIR}"

# check that libtorch exists
if [ ! -d "${PROJECT_DIR}/libtorch" ]; then
    wget https://download.pytorch.org/libtorch/cu117/libtorch-shared-with-deps-2.0.1%2Bcu117.zip
    unzip libtorch-shared-with-deps-2.0.1+cu117.zip
    rm libtorch-shared-with-deps-2.0.1+cu117.zip
    mv libtorch-shared-with-deps-2.0.1+cu117/libtorch "${PROJECT_DIR}/libtorch"
fi

# if LIBTORCH_DIR exists, use it, else fallback to installed torch
if [ -d "${PROJECT_DIR}/libtorch" ]; then
    LIBTORCH_DIR="${PROJECT_DIR}/libtorch"
else
    LIBTORCH_DIR="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"
fi
echo "LIBTORCH_DIR: ${LIBTORCH_DIR}"

echo "------------------------------------------------------------------------"
echo "Building..."
mkdir -p build
rm -rf build/*
cd build || exit
# NOTE: cxx11 ABI throws error messages, use pre-cxx11 ABI
cmake -DCMAKE_PREFIX_PATH="${LIBTORCH_DIR}" ..
# cmake -DCMAKE_PREFIX_PATH="${LIBTORCH_DIR}" -DCMAKE_CXX_CLANG_TIDY="clang-tidy;-header-filter=$(realpath ..)" ..
make -j

echo "------------------------------------------------------------------------"
echo "Running tests..."
source "${PROJECT_DIR}/.venv/bin/activate"
echo "Python env: $(which python)"
cd ..
pwd
pytest tests
python -n 4 --no-cov check_grad.py
