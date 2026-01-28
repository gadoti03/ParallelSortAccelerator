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
    inline unsigned int* generate_random_array(unsigned size, int seed = 42) {
        unsigned int* arr = static_cast<unsigned int*>(_mm_malloc(size*sizeof(unsigned int), DATA_LANE));
        std::mt19937 rng(seed);
        std::uniform_int_distribution<unsigned int> dist(0, 2147483647);
        for(unsigned i=0;i<size;i++) arr[i]=dist(rng);
        return arr; // caller must free with _mm_free(arr)
    }

    // Checks if the array is sorted
    inline bool is_sorted(const unsigned int* arr, unsigned n) {
        for(unsigned i=1;i<n;i++)
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
    inline void copy_array(const unsigned int* src, unsigned int* dst, unsigned n) {
        std::memcpy(dst, src, n*sizeof(unsigned int));
    }

    // Checks if two arrays are equal
    inline bool arrays_equal(const unsigned int* a, const unsigned int* b, unsigned n) {
        // byte-wise comparison is safe for unsigned int
        return std::memcmp(a, b, n * sizeof(unsigned int)) == 0;
    }
}