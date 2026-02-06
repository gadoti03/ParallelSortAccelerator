#include "sort.hpp"
#include <cstring>
#include <immintrin.h>
#include <cmath>

namespace simd { 

    // ------------------ Bitonic Sort ------------------
    /*
        Using SIMD instructions to accelerate the compare and swap phase of Bitonic Sort.
    */
    static void bitonic_merge(unsigned int* arr, unsigned int low, unsigned int cnt, bool dir) {
        if (cnt <= 1) return;

        unsigned int dim_regs = 16; // 128 bits / 16 Bytes
        unsigned int integers_per_reg = dim_regs / sizeof(unsigned int); // 4 integers per register

        unsigned int k = cnt >> 1;

        if (k >= integers_per_reg) { // if the half size is >= than the number of integers per register (4), use SIMD
            unsigned int i = low;

            unsigned int simd_end = low + k;

            for (; i < simd_end; i += integers_per_reg) {
                __m128i a = _mm_load_si128((__m128i*)&arr[i]); // load first half
                __m128i b = _mm_load_si128((__m128i*)&arr[i + k]); // load second half

                __m128i lo = _mm_min_epu32(a, b);
                __m128i hi = _mm_max_epu32(a, b);

                if (!dir) std::swap(lo, hi);

                _mm_store_si128((__m128i*)(arr + i), lo);
                _mm_store_si128((__m128i*)(arr + i + k), hi);
            }
        } else {
            for (unsigned int i = low; i < low + k; ++i) {
                if (dir == (arr[i] > arr[i + k])) {
                    std::swap(arr[i], arr[i + k]);
                }
            }
        }

        bitonic_merge(arr, low,     k, dir);
        bitonic_merge(arr, low + k, k, dir);
    }

    static void bitonic_sort_rec(unsigned int* arr, unsigned int low, unsigned int cnt, bool dir) {
        if (cnt <= 1) return;

        unsigned int k = cnt >> 1;

        bitonic_sort_rec(arr, low,     k, true);
        bitonic_sort_rec(arr, low + k, k, false);
        bitonic_merge(arr, low, cnt, dir);
    }

    void bitonic_sort(unsigned int* arr, unsigned int n) {
        bitonic_sort_rec(arr, 0, n, true);
    }

}
