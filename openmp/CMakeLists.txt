# Create a static library for the SIMD C++ backend
add_library(openmp_sort STATIC openmp_sort.cpp)

# Include headers from the include/ folder
target_include_directories(openmp_sort PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Optional: compiler warnings
if(USE_OPENMP)
    target_link_libraries(openmp_sort PUBLIC OpenMP::OpenMP_CXX)
endif()