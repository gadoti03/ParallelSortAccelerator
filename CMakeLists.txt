# --- Minimum CMake version and project name ---
cmake_minimum_required(VERSION 3.20)
project(ParallelSortAccelerator CXX)

# --- C++ standard ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Options to enable/disable backends ---
option(USE_SERIAL "Enable serial C++ backend" ON)
option(USE_SIMD "Enable SIMD CPU backend" ON)
option(USE_OPENMP_BACKEND "Enable OpenMP CPU backend" OFF)
option(USE_CUDA_BACKEND "Enable CUDA GPU backend" OFF)

# --- Initialize list of enabled backends ---
set(BACKENDS "")

# --- Find OpenMP if requested ---
if(USE_OPENMP_BACKEND)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found! Enabling OpenMP backend")
        set(USE_OPENMP TRUE)
    else()
        message(WARNING "OpenMP not found! OpenMP backend will be disabled.")
        set(USE_OPENMP FALSE)
    endif()
endif()

# --- Find CUDA if requested ---
if(USE_CUDA_BACKEND)
    find_package(CUDA)
    if(CUDA_FOUND)
        message(STATUS "CUDA found! Enabling CUDA backend")
        set(USE_CUDA TRUE)
    else()
        message(WARNING "CUDA not found! CUDA backend will be disabled.")
        set(USE_CUDA FALSE)
    endif()
endif()

# --- Add Serial backend ---
if(USE_SERIAL)
    add_subdirectory(src)
    list(APPEND BACKENDS serial_sort)
endif()

# --- Add SIMD backend ---
if(USE_SIMD)
    add_subdirectory(simd)
    list(APPEND BACKENDS simd_sort)
endif()

# --- Add OpenMP backend ---
if(USE_OPENMP_BACKEND AND USE_OPENMP)
    add_subdirectory(openmp)
    list(APPEND BACKENDS openmp_sort)
endif()

# --- Add CUDA backend ---
if(USE_CUDA_BACKEND AND USE_CUDA)
    add_subdirectory(cuda)
    list(APPEND BACKENDS cuda_sort)
endif()

# --- Add tests ---
add_subdirectory(tests)

# --- Add benchmarks ---
add_subdirectory(benchmarks)
