# Create a static library for the CUDA C++ backend
add_library(cuda_sort STATIC cuda_sort.cpp)

# Include headers from the include/ folder
target_include_directories(cuda_sort PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Optional: compiler warnings
# target_compile_options(cuda_sort PUBLIC -Wall -Wextra -Wpedantic)