#pragma once
#include <random>
#include <algorithm>
#include <immintrin.h>
#include <functional>
#include <chrono>
#include <cstring>

#define DATA_LANE 32

namespace utils {

    // Generates a random integer array, aligned to DATA_LANE
    inline int* generate_random_array(int size, int seed = 42) {
        int* arr = static_cast<int*>(_mm_malloc(size*sizeof(int), DATA_LANE));
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, 2147483647);
        for(int i=0;i<size;i++) arr[i]=dist(rng);
        return arr; // caller must free with _mm_free(arr)
    }

    // Checks if the array is sorted
    inline bool is_sorted(const int* arr, int n) {
        for(int i=1;i<n;i++)
            if(arr[i-1]>arr[i]) return false;
        return true;
    }

    // Measures execution time of a function in milliseconds
    inline double measure_time(const std::function<void()>& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end-start).count();
    }

    // Copies an array
    inline void copy_array(const int* src, int* dst, int n) {
        std::memcpy(dst, src, n*sizeof(int));
    }

} // namespace utils