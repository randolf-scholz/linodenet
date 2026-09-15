cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(linodenet_special VERSION 0.4.2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Section: configure user-facing build options.
option(LINODENET_SPECIAL_USE_CUDA "Enable CUDA-dependent build configuration" ON)
option(LINODENET_SPECIAL_ENABLE_CLANG_TIDY "Run clang-tidy during builds" OFF)
option(LINODENET_SPECIAL_USE_CUDNN "Enable cuDNN support" ON)
option(LINODENET_SPECIAL_USE_CUSPARSELT "Enable cuSPARSELt support" OFF)
option(LINODENET_SPECIAL_USE_CUDSS "Enable cuDSS support" OFF)
option(LINODENET_SPECIAL_USE_CUFILE "Enable cuFile support" OFF)
set(
        LINODENET_SPECIAL_TORCH_CXX11_ABI
        ""
        CACHE STRING
        "Value of torch.compiled_with_cxx11_abi() (0 or 1)"
)

find_package(Python REQUIRED COMPONENTS Interpreter)
execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE LINODENET_SPECIAL_TORCH_CMAKE_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
)
list(PREPEND CMAKE_PREFIX_PATH "${LINODENET_SPECIAL_TORCH_CMAKE_PREFIX}")

if (LINODENET_SPECIAL_TORCH_CXX11_ABI STREQUAL "")
    execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import torch; print(int(torch.compiled_with_cxx11_abi()))"
            OUTPUT_VARIABLE LINODENET_SPECIAL_TORCH_CXX11_ABI
            OUTPUT_STRIP_TRAILING_WHITESPACE
            COMMAND_ERROR_IS_FATAL ANY
    )
endif ()

# Section: declare the source files for the shared extension target.
set(linodenet_special_sources
        csrc/liblinodenet_special/singular_triplet.cpp
        csrc/liblinodenet_special/spectral_norm.cpp
        csrc/liblinodenet_special/ndtri_exp.cpp
        csrc/liblinodenet_special/hard_bend.cpp
        csrc/liblinodenet_special/gaussian_transport.cpp
)

# Section: discover third-party dependencies from the PyTorch build requirement.
unset(Caffe2_DIR CACHE)
unset(Torch_DIR CACHE)

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
target_compile_features(linodenet_special PRIVATE cxx_std_20)

# GCC 13 falsely reports an overflow in PyTorch's Function::apply vector<bool> bookkeeping.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
        AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13
        AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
    target_compile_options(linodenet_special PRIVATE -Wno-stringop-overflow)
endif ()

if (NOT LINODENET_SPECIAL_TORCH_CXX11_ABI STREQUAL "")
    if (NOT LINODENET_SPECIAL_TORCH_CXX11_ABI MATCHES "^[01]$")
        message(FATAL_ERROR "LINODENET_SPECIAL_TORCH_CXX11_ABI must be 0 or 1.")
    endif ()
    target_compile_definitions(
            linodenet_special
            PRIVATE "_GLIBCXX_USE_CXX11_ABI=${LINODENET_SPECIAL_TORCH_CXX11_ABI}"
    )
endif ()

if (LINODENET_SPECIAL_ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY NAMES clang-tidy-20 clang-tidy REQUIRED)
    set_property(TARGET linodenet_special PROPERTY CXX_CLANG_TIDY "${CLANG_TIDY}")
endif ()

target_link_libraries(linodenet_special PRIVATE torch)

install(TARGETS linodenet_special LIBRARY DESTINATION linodenet_special)
