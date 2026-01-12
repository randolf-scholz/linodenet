cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(linodenet VERSION 0.1.0 LANGUAGES CXX CUDA)


# OPTIONS

# Global settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_CLANG_TIDY clang-tidy)


message(STATUS "Configuring linodenet...")

# determine project root
execute_process (
    COMMAND bash -c "git rev-parse --show-toplevel | xargs echo -n"
    OUTPUT_VARIABLE PROJECT_ROOT
)
message(STATUS "NOTE: PROJECT_ROOT=${PROJECT_ROOT}")

set(PYTHON_EXECUTABLE "${PROJECT_ROOT}/.venv/bin/python3")
if(NOT EXISTS "${PYTHON_EXECUTABLE}")
  message(STATUS "NOTE: venv python not found at ${PYTHON_EXECUTABLE} — falling back to system python3")
  set(PYTHON_EXECUTABLE "python3")
endif()
message(STATUS "NOTE: PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}")

# determine cuda version from calling torch.__version__ in python
execute_process (
    COMMAND ${PYTHON_EXECUTABLE} -c "import torch; print(torch.version.cuda, end='')"
    OUTPUT_VARIABLE CUDA_VERSION
)
message(STATUS "NOTE: CUDA_VERSION=${CUDA_VERSION}")


set(CMAKE_CUDA_COMPILER "/usr/local/cuda-${CUDA_VERSION}/bin/nvcc")
set(CMAKE_INCLUDE_PATH  "/usr/local/cuda-${CUDA_VERSION}/include")
set(CMAKE_LIBRARY_PATH "/usr/local/cuda-${CUDA_VERSION}/lib64")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-${CUDA_VERSION}")

# get dependencies
list(APPEND CMAKE_PREFIX_PATH "libtorch/share/cmake")



# set compiler flags
set(CAFFE2_USE_CUDNN true)
option(USE_CUDA "Use CUDA" ON)
option(USE_CUDNN "Use CUDNN" ON)
# option(CAFFE2_USE_CUDNN "Use CUDNN" ON)  # otherwise cudnn is not found

# Dependency discovery
find_package(Torch REQUIRED)
find_package(Python REQUIRED COMPONENTS Development)

if(LINODENET_USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit)
endif()


# Targets
add_library(
        linodenet
        SHARED
        src/liblinodenet/singular_triplet.cpp
        src/liblinodenet/singular_triplet_debug.cpp
        src/liblinodenet/singular_triplet_riemann.cpp
        src/liblinodenet/spectral_norm.cpp
        src/liblinodenet/spectral_norm_debug.cpp
        src/liblinodenet/spectral_norm_riemann.cpp
        src/liblinodenet/utils.cpp
        src/liblinodenet/matrix_root.cpp
)

# Target configuration
target_compile_features(linodenet PRIVATE cxx_std_17)
# Link against LibTorch
target_link_libraries(linodenet "${TORCH_LIBRARIES}")
