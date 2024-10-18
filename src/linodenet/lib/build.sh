#!/usr/bin/env bash
set -e

# determine project dir
PROJECT_DIR=$(git rev-parse --show-toplevel | xargs echo -n)
echo "PROJECT_DIR: ${PROJECT_DIR}"

CUDA_VERSION="12.4"
TORCH_VERSION="2.5.0"
LIBTORCH_CUDA="cu${CUDA_VERSION//./}"
LIBTORCH_DIR="libtorch"
LIBTORCH_VERSION="$TORCH_VERSION+$LIBTORCH_CUDA"
LIBTORCH_ARCHIVE="libtorch-shared-with-deps-$LIBTORCH_VERSION.zip"
LIBTORCH_URL="https://download.pytorch.org/libtorch/$LIBTORCH_CUDA/$LIBTORCH_ARCHIVE"

# 2.2.0+cu121: 0a1a034b1980199543ec5cbc8d42215f55b188ac188b3dac42d83aeb449922bb
LIBTORCH_HASH="1e93450e8d2ef7d00b08e92318d4017897b4ba00c5c714a077579aca77f41424"

# check if libtorch folder exists
if [ -d $LIBTORCH_DIR ]; then
  # validate libtorch version
  echo "Checking libtorch version..."
  libtorch_version=$(cat $LIBTORCH_DIR/build-version)  # 2.2.0+cu121
  if [ "$libtorch_version" != "$LIBTORCH_VERSION" ]; then
    echo "Error: libtorch version mismatch!"
    echo "Expected: $LIBTORCH_VERSION"
    echo "Found: $libtorch_version"

    # ask if libtorch should be re-downloaded (default: yes)
    read -r -p "Re-download libtorch? [Y/n] " re_download
    re_download=${re_download:-Y}
    if [[ $re_download =~ ^[Yy]$ ]]; then
      echo "Re-downloading libtorch..."
      rm -rf $LIBTORCH_DIR
    else
      echo "Skipping re-download..."
    fi
  fi
fi

# check that libtorch exists
if [ ! -d "libtorch/" ]; then
  # check if libtorch archive exists
  if [ ! -f "$LIBTORCH_ARCHIVE" ]; then
      echo "Downloading libtorch..."
      # replace '+' with '%2B' in url
      wget "${LIBTORCH_URL//+/%2B}"
  fi
  # check hash
  echo "Checking hash..."
  echo "$LIBTORCH_HASH $LIBTORCH_ARCHIVE" | sha256sum --check
  # extract "libtorch" directory from the zip file
  echo "Extracting libtorch..."
  unzip -q "$LIBTORCH_ARCHIVE" "$LIBTORCH_DIR/*"
fi

# assert that libtorch exists and update LIBTORCH_DIR
if [ ! -d "$LIBTORCH_DIR/" ]; then
    echo "Error: libtorch not found!"
    exit 1
else
    LIBTORCH_DIR=$(realpath "libtorch/")
    echo "LIBTORCH_DIR: ${LIBTORCH_DIR}"
fi

# region build -------------------------------------------------------------------------
echo "-------------------------------------------------------------------------"

# activate correct python
source "${PROJECT_DIR}/.venv/bin/activate"
echo "Python env: $(which python)"

# prepend correct CUDA version
export PATH="/usr/local/cuda-$CUDA_VERSION/bin:$PATH"
# FIXME: https://github.com/pytorch/pytorch/issues/113948
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"

echo "Building..."
mkdir -p build
rm -rf build/*
cd build || exit
# NOTE: cxx11 ABI throws error messages, use pre-cxx11 ABI
cmake -DCMAKE_PREFIX_PATH="${LIBTORCH_DIR}" ..
# cmake -DCMAKE_PREFIX_PATH="${LIBTORCH_DIR}" -DCMAKE_CXX_CLANG_TIDY="clang-tidy;-header-filter=$(realpath ..)" ..
make -j

cd ..
pwd
echo "-------------------------------------------------------------------------"
# endregion build ----------------------------------------------------------------------


# region tests -------------------------------------------------------------------------
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
# endregion tests ----------------------------------------------------------------------
