# --- Add the test executable ---
add_executable(test_sort test_sort.cpp)

# --- Link the backends (from top-level CMake) ---
target_link_libraries(test_sort PRIVATE ${BACKENDS})

# --- Include directories (so the test can find headers) ---
target_include_directories(test_sort PRIVATE 
    ${CMAKE_SOURCE_DIR}/src # ??? potrebbe non servire
    ${CMAKE_SOURCE_DIR}/include
)