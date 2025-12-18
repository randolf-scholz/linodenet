cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(linodenet LANGUAGES CXX)

message(STATUS "Configuring linodenet...")

# determine project root
execute_process (
    COMMAND bash -c "git rev-parse --show-toplevel | xargs echo -n"
    OUTPUT_VARIABLE PROJECT_ROOT
)
message(STATUS "NOTE: PROJECT_ROOT=${PROJECT_ROOT}")

# determine cuda version from calling torch.__version__ in python
execute_process (
    COMMAND python3 -c "import torch; print(torch.version.cuda, end='')"
    OUTPUT_VARIABLE CUDA_VERSION
)
message(STATUS "NOTE: CUDA_VERSION=${CUDA_VERSION}")

# set global flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_CLANG_TIDY clang-tidy)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-${CUDA_VERSION}/bin/nvcc")
set(CMAKE_INCLUDE_PATH  "/usr/local/cuda-${CUDA_VERSION}/include")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-${CUDA_VERSION}")

# set compiler flags
set(USE_CUDA true)
set(USE_CUDNN true)
set(CAFFE2_USE_CUDNN true)
option(USE_CUDA "Use CUDA" ON)
option(USE_CUDNN "Use CUDNN" ON)
# option(CAFFE2_USE_CUDNN "Use CUDNN" ON)  # otherwise cudnn is not found

# Info


# get dependencies
list(APPEND CMAKE_PREFIX_PATH "libtorch/share/cmake/Torch")
find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#message(STATUS "NOTE: CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
#message(STATUS "NOTE: TORCH_CXX_FLAGS=${TORCH_CXX_FLAGS}")

# Define our library target
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

# Enable C++17
target_compile_features(linodenet PRIVATE cxx_std_17)
# Link against LibTorch
target_link_libraries(linodenet "${TORCH_LIBRARIES}")
# # Link against OpenCV
# target_link_libraries(spectral_norm opencv_core opencv_imgproc)
