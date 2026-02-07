# Create a static library for the CUDA backend
add_library(cuda_sort_v1 STATIC cuda_sort_v1.cu)
target_include_directories(cuda_sort_v1 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_options(nvcc -o cuda_sort_v1 PRIVATE cuda_sort_v1.cu)

add_library(cuda_sort_v1 STATIC cuda_sort_v1.cu)
target_include_directories(cuda_sort_v1 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_options(nvcc -o cuda_sort_v1 PRIVATE cuda_sort_v1.cu)

add_library(cuda_sort_v1 STATIC cuda_sort_v1.cu)
target_include_directories(cuda_sort_v1 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_options(nvcc -o cuda_sort_v1 PRIVATE cuda_sort_v1.cu)