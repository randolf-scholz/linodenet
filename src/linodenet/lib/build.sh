#!/usr/bin/env bash
set -e

# determine project dir
PROJECT_DIR=$(git rev-parse --show-toplevel | xargs echo -n)
echo "PROJECT_DIR: ${PROJECT_DIR}"

CUDA_VERSION="$(python -c 'import torch; print(torch.version.cuda)')"
TORCH_VERSION="$(python -c 'import torch; print(torch.__version__)')"  # e.g. 2.5.1+cu124
LIBTORCH_VERSION="$TORCH_VERSION"
LIBTORCH_CUDA="cu${CUDA_VERSION//./}"  # e.g. cu124
LIBTORCH_DIR="libtorch"
LIBTORCH_ARCHIVE="libtorch-shared-with-deps-$LIBTORCH_VERSION.zip"
LIBTORCH_URL="https://download.pytorch.org/libtorch/$LIBTORCH_CUDA/$LIBTORCH_ARCHIVE"

# 2.2.0+cu121: 0a1a034b1980199543ec5cbc8d42215f55b188ac188b3dac42d83aeb449922bb
# 2.5.1+cu124: 470ab7f7f56e96d28d1dc9ae34ceb2e0d8723cc2899c5d0192f4cb12b8f7843b
# 2.9.1+cu128: b052452965093db69f537b3cf376812d5acf6dca28819b20d28d7f0b171d7699
LIBTORCH_HASH="b052452965093db69f537b3cf376812d5acf6dca28819b20d28d7f0b171d7699"

# check if libtorch folder exists
if [ -d "$LIBTORCH_DIR" ]; then
	# validate libtorch version
	echo "Checking libtorch version..."
	libtorch_version="$(cat "$LIBTORCH_DIR/build-version")"
	if [ "$libtorch_version" != "$LIBTORCH_VERSION" ]; then
		echo "Error: libtorch version mismatch!"
		echo "Expected: $LIBTORCH_VERSION"
		echo "Found: $libtorch_version"

		# ask if libtorch should be re-downloaded (default: yes)
		read -r -p "Re-download libtorch? [Y/n] " re_download
		case "${re_download:-Y}" in
			y|Y) rm -rf $LIBTORCH_DIR ;;
			n|N) echo "Skipping re-download..." ;;
			*) echo "Invalid input. Skipping re-download..." ;;
		esac
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
	file_hash=$(sha256sum "$LIBTORCH_ARCHIVE" | cut -d ' ' -f 1)
	if [ "$file_hash" != "$LIBTORCH_HASH" ]; then
		echo "Error: libtorch hash mismatch!"
		echo "Expected: $LIBTORCH_HASH"
		echo "Found: $file_hash"
		# ask whether to continue
		read -r -p "Continue? [y/N] " choice
		case "${choice:-N}" in
			y|Y) echo "Continuing...";;
			n|N) echo "Exiting..."; exit 1;;
			*) echo "Invalid input. Exiting..."; exit 1;;
		esac
	fi

	# extract "libtorch" directory from the zip file
	echo "Extracting libtorch..."
	rm -rf "$LIBTORCH_DIR"
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
# NOTE: cxx11 ABI throws error messages, use pre-cxx11 ABI
echo "-------------------------------------------------------------------------"
echo "Building..."

# create build directory
mkdir -p build && rm -rf build/*
cd build || exit 1

# activate correct python
source "${PROJECT_DIR}/.venv/bin/activate"
echo "Python env: $(type python)"

# prepend correct CUDA version
export PATH="/usr/local/cuda-$CUDA_VERSION/bin:$PATH"
export CMAKE_INCLUDE_PATH="/usr/local/cuda-$CUDA_VERSION/include"
export LD_LIBRARY_PATH="/usr/local/cuda-$CUDA_VERSION/lib64:$LD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
cmake -DCMAKE_PREFIX_PATH="${LIBTORCH_DIR}" ..
make -j

cd ..  # exit build directory
echo "-------------------------------------------------------------------------"
# endregion build ----------------------------------------------------------------------


# region tests -------------------------------------------------------------------------
read -r -p "Run tests? [Y/n] " run_tests
case "${run_tests:-Y}" in
	y|Y) pytest tests/liblinodenet/test_correctness.py -n 0 --no-cov ;;
	n|N) echo "Skipping tests..." ;;
	*) echo "Invalid input. Exiting..."; exit 1 ;;
esac
# endregion tests ----------------------------------------------------------------------
