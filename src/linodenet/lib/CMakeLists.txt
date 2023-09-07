cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(liblinodenet LANGUAGES CXX)

# run clang-tidy
set(CMAKE_CXX_CLANG_TIDY clang-tidy)

# set global flags
set(CMAKE_CXX_STANDARD 17)
set(USE_CUDA true)
set(USE_CUDNN true)
set(CAFFE2_USE_CUDNN true)
option(USE_CUDA "Use CUDA" ON)
option(USE_CUDNN "Use CUDNN" ON)
option(CAFFE2_USE_CUDNN "Use CUDNN" ON)  # otherwise cudnn is not found


# get dependencies
execute_process (
    COMMAND bash -c "git rev-parse --show-toplevel | xargs echo -n"
    OUTPUT_VARIABLE PROJECT_ROOT
)
list(APPEND CMAKE_PREFIX_PATH "${PROJECT_ROOT}/libtorch/share/cmake/Torch")

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(    STATUS "NOTE: CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
message(STATUS "NOTE: TORCH_CXX_FLAGS=${TORCH_CXX_FLAGS}")


# print debug info

message(STATUS "NOTE: PROJECT_ROOT=${PROJECT_ROOT}")
message(STATUS "NOTE: USE_CUDNN=${USE_CUDNN}")

#message(STATUS "paths=${CMAKE_PREFIX_PATH}")


#set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-checks=*")

# Define our library target
add_library(
        linodenet
        SHARED
        src/liblinodenet/singular_triplet.cpp
        src/liblinodenet/singular_triplet_debug.cpp
        src/liblinodenet/spectral_norm.cpp
        src/liblinodenet/spectral_norm_debug.cpp
)

# Enable C++14
target_compile_features(linodenet PRIVATE cxx_std_17)
# Link against LibTorch
target_link_libraries(linodenet "${TORCH_LIBRARIES}")
# # Link against OpenCV
# target_link_libraries(spectral_norm opencv_core opencv_imgproc)
