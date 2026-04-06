cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(linodenet_special VERSION 0.2.0 LANGUAGES CXX)


# OPTIONS
# Global settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_CLANG_TIDY clang-tidy)


message(STATUS "Configuring linodenet_special...")

# determine project root
execute_process (
    COMMAND bash -c "git rev-parse --show-toplevel | xargs echo -n"
    OUTPUT_VARIABLE PROJECT_ROOT
)
message(STATUS "NOTE: PROJECT_ROOT=${PROJECT_ROOT}")

set(LIBTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
set(LIBTORCH_CMAKE_DIR "${LIBTORCH_ROOT}/share/cmake")

set(DEFAULT_CUDA_ROOT "")
foreach(CUDA_ROOT_CANDIDATE IN ITEMS /usr/local/cuda /usr/local/cuda-13 /usr/local/cuda-13.0)
  if(EXISTS "${CUDA_ROOT_CANDIDATE}")
    set(DEFAULT_CUDA_ROOT "${CUDA_ROOT_CANDIDATE}")
    break()
  endif()
endforeach()
message(STATUS "NOTE: DEFAULT_CUDA_ROOT=${DEFAULT_CUDA_ROOT}")

# get dependencies
if(EXISTS "${LIBTORCH_CMAKE_DIR}")
  list(PREPEND CMAKE_PREFIX_PATH "${LIBTORCH_CMAKE_DIR}")
endif()



# set compiler flags
set(CAFFE2_USE_CUDNN true)
option(USE_CUDA "Use CUDA" ON)
option(USE_CUDNN "Use CUDNN" ON)
# option(CAFFE2_USE_CUDNN "Use CUDNN" ON)  # otherwise cudnn is not found

if(USE_CUDA AND DEFAULT_CUDA_ROOT)
  set(CMAKE_INCLUDE_PATH "${DEFAULT_CUDA_ROOT}/include")
  set(CMAKE_LIBRARY_PATH "${DEFAULT_CUDA_ROOT}/lib64")
  set(CUDAToolkit_ROOT "${DEFAULT_CUDA_ROOT}")
  set(CUDA_TOOLKIT_ROOT_DIR "${DEFAULT_CUDA_ROOT}")
endif()

if(USE_CUDA AND (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL ""))
  set(CMAKE_CUDA_ARCHITECTURES "all-major")
endif()

if(USE_CUDA AND DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(TORCH_CUDA_ARCH_LIST_FROM_CMAKE "")
  set(TORCH_CUDA_ARCH_TRANSLATION_OK ON)
  foreach(CUDA_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "-(real|virtual)$" "" CUDA_ARCH_TOKEN "${CUDA_ARCH}")
    if(CUDA_ARCH_TOKEN MATCHES "^([0-9]+)([0-9])([a-z]?)$")
      list(APPEND TORCH_CUDA_ARCH_LIST_FROM_CMAKE "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}${CMAKE_MATCH_3}")
    elseif(CUDA_ARCH_TOKEN MATCHES "^(all|all-major|native)$")
      set(TORCH_CUDA_ARCH_TRANSLATION_OK OFF)
      message(STATUS "NOTE: keeping TORCH_CUDA_ARCH_LIST unset for special CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH_TOKEN}")
      break()
    else()
      set(TORCH_CUDA_ARCH_TRANSLATION_OK OFF)
      message(STATUS "NOTE: ignoring unsupported CMAKE_CUDA_ARCHITECTURES entry: ${CUDA_ARCH}")
      break()
    endif()
  endforeach()

  if(TORCH_CUDA_ARCH_TRANSLATION_OK AND TORCH_CUDA_ARCH_LIST_FROM_CMAKE)
    list(JOIN TORCH_CUDA_ARCH_LIST_FROM_CMAKE ";" TORCH_CUDA_ARCH_LIST)
    message(STATUS "NOTE: TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}")
  endif()
endif()

if(USE_CUDA AND NOT DEFINED TORCH_CUDA_ARCH_LIST)
  set(TORCH_CUDA_ARCH_LIST "")
endif()

# Dependency discovery
find_package(Torch REQUIRED)
find_package(Python REQUIRED COMPONENTS Development)

if(USE_CUDA AND DEFAULT_CUDA_ROOT)
  find_package(CUDAToolkit REQUIRED)
endif()


# Targets
add_library(
        linodenet_special
        SHARED
        csrc/liblinodenet_special/singular_triplet.cpp
        csrc/liblinodenet_special/spectral_norm.cpp
        csrc/liblinodenet_special/ndtri_exp.cpp
        csrc/liblinodenet_special/hard_bend.cpp
        csrc/liblinodenet_special/gaussian_transport.cpp
)

# Target configuration
target_compile_features(linodenet_special PRIVATE cxx_std_17)
# Link against LibTorch
target_link_libraries(linodenet_special "${TORCH_LIBRARIES}")
