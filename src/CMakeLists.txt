# Create a static library for the serial C++ backend
add_library(serial_sort STATIC serial_sort.cpp)

# Include headers from the include/ folder
target_include_directories(serial_sort PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Optional: compiler warnings
# target_compile_options(serial_sort PUBLIC -Wall -Wextra -Wpedantic)