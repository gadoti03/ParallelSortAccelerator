# SIMD v1 → 128-bit
add_library(simd_sort_v1 STATIC simd_sort_v1.cpp)
target_include_directories(simd_sort_v1 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_options(simd_sort_v1 PUBLIC -O3 -msse4.2 -Wall -Wextra -Wpedantic)
target_compile_definitions(simd_sort_v1 PRIVATE SIMD_VERSION=1 SIMD_REGWIDTH=128)

# SIMD v2 → 256-bit
add_library(simd_sort_v2 STATIC simd_sort_v2.cpp)
target_include_directories(simd_sort_v2 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_options(simd_sort_v2 PUBLIC -O3 -mavx2 -Wall -Wextra -Wpedantic)
target_compile_definitions(simd_sort_v2 PRIVATE SIMD_VERSION=2 SIMD_REGWIDTH=256)

# SIMD v3 → 128-bit + 256-bit
add_library(simd_sort_v3 STATIC simd_sort_v3.cpp)
target_include_directories(simd_sort_v3 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_options(simd_sort_v3 PUBLIC -O3 -mavx2 -msse4.2 -Wall -Wextra -Wpedantic)
target_compile_definitions(simd_sort_v3 PRIVATE SIMD_VERSION=3)