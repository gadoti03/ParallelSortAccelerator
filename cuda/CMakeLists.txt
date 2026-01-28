# Create a static library for the SIMD C++ backend
add_library(cuda_sort STATIC cuda_sort.cu)

# Include headers from the include/ folder
target_include_directories(cuda_sort PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Optional: compiler warnings
if(USE_CUDA)
    target_link_libraries(cuda_sort PUBLIC ${CUDA_LIBRARIES})
    target_compile_options(cuda_sort PUBLIC ${CUDA_NVCC_FLAGS})
endif()
