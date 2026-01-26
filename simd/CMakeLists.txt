# Create a static library for the SIMD C++ backend
add_library(simd_sort STATIC simd_sort.cpp)

# Include headers from the include/ folder
target_include_directories(simd_sort PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Optional: compiler warnings
# target_compile_options(simd_sort PUBLIC -Wall -Wextra -Wpedantic)