# --- Minimum CMake version and project name ---
cmake_minimum_required(VERSION 3.20)
project(ParallelSortAccelerator CXX) # add CUDA

# --- C++ standard ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Options to enable/disable backends ---
option(USE_SERIAL "Enable serial C++ backend" ON)
option(USE_SIMD "Enable SIMD CPU backend" OFF)
option(USE_CUDA "Enable CUDA GPU backend" OFF)

# --- List to store enabled backends ---
set(BACKENDS "")

# --- Add subdirectories for each backend if enabled ---
if (USE_SERIAL)
    add_subdirectory(src)
    list(APPEND BACKENDS serial_sort)
endif()

if (USE_SIMD)
    add_subdirectory(simd)
    list(APPEND BACKENDS simd_sort)
endif()

if (USE_CUDA)
    add_subdirectory(cuda)
    list(APPEND BACKENDS cuda_sort)
endif()

# --- Add tests ---
add_subdirectory(tests)

# --- Add benchmarks (optional) ---
add_subdirectory(benchmarks)
