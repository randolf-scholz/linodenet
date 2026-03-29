#!/usr/bin/env bash
set -e

# determine project dir
PROJECT_DIR=$(git rev-parse --show-toplevel | xargs echo -n)
SOURCE_DIR="${PROJECT_DIR}/src/linodenet_special"
BUILD_DIR="${SOURCE_DIR}/build"
LIBTORCH_DIR="${SOURCE_DIR}/libtorch"

echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "SOURCE_DIR: ${SOURCE_DIR}"
echo "BUILD_DIR: ${BUILD_DIR}"
echo "LIBTORCH_DIR: ${LIBTORCH_DIR}"
mkdir -p "$BUILD_DIR"
cd "$SOURCE_DIR"

CUDA_VERSION="$(python -c 'import torch; print(torch.version.cuda)')"
TORCH_VERSION="$(python -c 'import torch; print(torch.__version__)')"  # e.g. 2.5.1+cu124
LIBTORCH_VERSION="$TORCH_VERSION"
LIBTORCH_CUDA="cu${CUDA_VERSION//./}"  # e.g. cu124
LIBTORCH_ARCHIVE="libtorch-shared-with-deps-$LIBTORCH_VERSION.zip"
LIBTORCH_URL="https://download.pytorch.org/libtorch/$LIBTORCH_CUDA/$LIBTORCH_ARCHIVE"

# map known libtorch versions to their expected sha256 hashes
declare -A LIBTORCH_HASHES=(
  ["2.2.0+cu121"]="0a1a034b1980199543ec5cbc8d42215f55b188ac188b3dac42d83aeb449922bb"
  ["2.5.1+cu124"]="470ab7f7f56e96d28d1dc9ae34ceb2e0d8723cc2899c5d0192f4cb12b8f7843b"
  ["2.9.1+cu128"]="b052452965093db69f537b3cf376812d5acf6dca28819b20d28d7f0b171d7699"
  ["2.10.0+cu128"]="429aa9fead3cf3d557e7c310442a1fae3879cdc14a469ff452043b39b61666a9"
  ["2.11.0+cu130"]="a163eff74ffc1eaf3827e808c8bad3a88338ca68b5733d0974c1cbc9bc033295"
)

LIBTORCH_HASH="${LIBTORCH_HASHES[$LIBTORCH_VERSION]:-}"

# validate any existing libtorch checkout before reusing it
if [ -d "$LIBTORCH_DIR" ]; then
	echo "Checking libtorch installation..."
	if [ ! -f "$LIBTORCH_DIR/build-version" ]; then
		echo "Existing libtorch directory is incomplete: missing build-version."
		rm -rf "$LIBTORCH_DIR"
	else
		libtorch_version="$(<"$LIBTORCH_DIR/build-version")"
		if [ "$libtorch_version" != "$LIBTORCH_VERSION" ]; then
			echo "Error: libtorch version mismatch!"
			echo "Expected: $LIBTORCH_VERSION"
			echo "Found: $libtorch_version"

			# ask if libtorch should be re-downloaded (default: yes)
			read -r -p "Re-download libtorch? [Y/n] " re_download
			case "${re_download:-Y}" in
				y|Y) rm -rf "$LIBTORCH_DIR" ;;
				n|N) echo "Skipping re-download..." ;;
				*) echo "Invalid input. Skipping re-download..." ;;
			esac
		fi
	fi
fi

# check that libtorch exists
if [ ! -d "$LIBTORCH_DIR" ]; then
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
	unzip -q "$LIBTORCH_ARCHIVE" "libtorch/*"
fi


# region build -------------------------------------------------------------------------
# NOTE: cxx11 ABI throws error messages, use pre-cxx11 ABI
echo "-------------------------------------------------------------------------"
echo "Building..."

# activate correct python
source "${PROJECT_DIR}/.venv/bin/activate"
echo "Python env: $(type python)"

# prepend correct CUDA version
export PATH="/usr/local/cuda-$CUDA_VERSION/bin:$PATH"
export CMAKE_INCLUDE_PATH="/usr/local/cuda-$CUDA_VERSION/include"
export LD_LIBRARY_PATH="/usr/local/cuda-$CUDA_VERSION/lib64:$LD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# create build directory and clean it
mkdir -p "$BUILD_DIR" && rm -rf "${BUILD_DIR:?}/"*
CMAKE_ARGS=(
  -S .                                     # source directory
  -B "$BUILD_DIR"                          # build directory
  -G "Ninja"                               # use Ninja generator
  -DCMAKE_PREFIX_PATH="${LIBTORCH_DIR}"    #
  -DCMAKE_BUILD_TYPE=Release               #
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON       #
)
printf 'Running: %q \n' cmake "${CMAKE_ARGS[*]}"
cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --config Release -j

echo "-------------------------------------------------------------------------"
# endregion build ----------------------------------------------------------------------


# region tests -------------------------------------------------------------------------
read -r -p "Run tests? [Y/n] " run_tests
case "${run_tests:-Y}" in
	y|Y) pytest "${PROJECT_DIR}/tests/linodenet_special" -n 0 --no-cov ;;
	n|N) echo "Skipping tests..." ;;
	*) echo "Invalid input. Exiting..."; exit 1 ;;
esac
# endregion tests ----------------------------------------------------------------------
