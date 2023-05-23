cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(liblinodenet)


set(ENV{USE_CUDNN} "1")
set(USE_CUDNN "1")

message(STATUS "USE_CUDNN=${USE_CUDNN}")

execute_process (
    COMMAND bash -c "git rev-parse --show-toplevel | xargs echo -n"
    OUTPUT_VARIABLE PROJECT_ROOT
)
message(STATUS "PROJECT_ROOT=${PROJECT_ROOT}")

list(APPEND CMAKE_PREFIX_PATH "${PROJECT_ROOT}/libtorch/share/cmake/Torch")

#message(STATUS "paths=${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-checks=*")

# Define our library target
add_library(linodenet SHARED spectral_norm.cpp singular_triplet.cpp)
# Enable C++14
target_compile_features(linodenet PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(linodenet "${TORCH_LIBRARIES}")
# # Link against OpenCV
# target_link_libraries(spectral_norm opencv_core opencv_imgproc)
