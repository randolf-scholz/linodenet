#!/usr/bin/env bash
set -e

# determine project dir
PROJECT_DIR=$(git rev-parse --show-toplevel | xargs echo -n)
echo "PROJECT_DIR: ${PROJECT_DIR}"

# activate correct python
source "${PROJECT_DIR}/.venv/bin/activate"
echo "Python env: $(which python)"

# prepend correct CUDA version
export PATH=/usr/local/cuda-12.1/bin:$PATH

# check that libtorch exists
if [ ! -d "libtorch/" ]; then
    echo "Downloading libtorch..."
    fname="libtorch-shared-with-deps-2.1.0+cu121.zip"
    hashval="4e893c727367193169bc2cfc3c0c519f88411de5bf243ef237a3e23b926bfb45"
    # replace + with %2B
    wget "https://download.pytorch.org/libtorch/cu121/${fname//+/%2B}"
    # check hash
    echo "Checking hash..."
    echo "$hashval $fname" | sha256sum --check
    # extract "libtorch" directory from the zip file
    echo "Extracting libtorch..."
    unzip -q "$fname" "libtorch/*"
fi

# assert that libtorch exists and set LIBTORCH_DIR
if [ ! -d "libtorch/" ]; then
    echo "Error: libtorch not found"
    exit 1
else
    LIBTORCH_DIR=$(realpath "libtorch/")
    echo "LIBTORCH_DIR: ${LIBTORCH_DIR}"
fi

echo "-------------------------------------------------------------------------"
echo "Building..."
mkdir -p build
rm -rf build/*
cd build || exit
# NOTE: cxx11 ABI throws error messages, use pre-cxx11 ABI
cmake -DCMAKE_PREFIX_PATH="${LIBTORCH_DIR}" ..
# cmake -DCMAKE_PREFIX_PATH="${LIBTORCH_DIR}" -DCMAKE_CXX_CLANG_TIDY="clang-tidy;-header-filter=$(realpath ..)" ..
make -j

echo "-------------------------------------------------------------------------"
echo "Running tests..."
cd ..
pwd

# ask if tests should be run
read -r -p "Run tests? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
    echo "Running tests..."
else
    echo "Skipping tests..."
    exit 0
fi

pytest tests -n 4 --no-cov
python check_grad.py
