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
    fname="libtorch-shared-with-deps-2.2.0+cu121.zip"
    hashval="0a1a034b1980199543ec5cbc8d42215f55b188ac188b3dac42d83aeb449922bb"
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
cd ..
pwd

# ask if tests should be run (default: yes)
read -r -p "Run tests? [Y/n] " run_tests
run_tests=${run_tests:-Y}
if [[ $run_tests =~ ^[Yy]$ ]]; then
    echo "Running tests..."
    # run tests
    pytest tests/liblinodenet/test_correctness.py  -n 0 --no-cov
    python check_grad.py
else
    echo "Skipping tests..."
fi
