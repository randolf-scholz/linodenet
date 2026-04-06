cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(linodenet_special VERSION 0.4.2 LANGUAGES CXX)

# Section: configure user-facing build options.
option(LINODENET_SPECIAL_USE_CUDA "Enable CUDA-dependent build configuration" ON)
option(LINODENET_SPECIAL_ENABLE_CLANG_TIDY "Run clang-tidy during builds" ON)
option(LINODENET_SPECIAL_USE_CUDNN "Enable cuDNN support" ON)
option(LINODENET_SPECIAL_USE_CUSPARSELT "Enable cuSPARSELt support" OFF)
option(LINODENET_SPECIAL_USE_CUDSS "Enable cuDSS support" OFF)
option(LINODENET_SPECIAL_USE_CUFILE "Enable cuFile support" OFF)

# Section: declare the source files for the shared extension target.
set(linodenet_special_sources
        csrc/liblinodenet_special/singular_triplet.cpp
        csrc/liblinodenet_special/spectral_norm.cpp
        csrc/liblinodenet_special/ndtri_exp.cpp
        csrc/liblinodenet_special/hard_bend.cpp
        csrc/liblinodenet_special/gaussian_transport.cpp
)

# Section: discover third-party dependencies, preferring the vendored LibTorch package.
list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/share/cmake")

if (LINODENET_SPECIAL_USE_CUDA)
    if (NOT DEFINED TORCH_CUDA_ARCH_LIST)
        set(TORCH_CUDA_ARCH_LIST "Common")
    endif ()

    set(CAFFE2_USE_CUDNN "${LINODENET_SPECIAL_USE_CUDNN}" CACHE BOOL "Enable cuDNN support" FORCE)
    set(CAFFE2_USE_CUSPARSELT "${LINODENET_SPECIAL_USE_CUSPARSELT}" CACHE BOOL "Enable cuSPARSELt support" FORCE)
    set(USE_CUDSS "${LINODENET_SPECIAL_USE_CUDSS}" CACHE BOOL "Enable cuDSS support" FORCE)
    set(CAFFE2_USE_CUFILE "${LINODENET_SPECIAL_USE_CUFILE}" CACHE BOOL "Enable cuFile support" FORCE)
    find_package(CUDAToolkit REQUIRED)
endif ()

find_package(Torch REQUIRED)

# Section: define the shared library target built from the custom operator sources.
add_library(linodenet_special SHARED ${linodenet_special_sources})

# Section: apply target-local language, tooling, and linkage settings.
target_compile_features(linodenet_special PRIVATE cxx_std_17)
set_target_properties(
        linodenet_special
        PROPERTIES
        CXX_EXTENSIONS OFF
)

if (LINODENET_SPECIAL_ENABLE_CLANG_TIDY)
    set_target_properties(
            linodenet_special
            PROPERTIES
            CXX_CLANG_TIDY "clang-tidy"
    )
endif ()

target_link_libraries(linodenet_special PRIVATE "${TORCH_LIBRARIES}")
