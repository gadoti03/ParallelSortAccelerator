#include "sort.hpp"
#include <cstring>
#include <immintrin.h>
#include <cmath>

namespace simd { 

    // ------------------ Bitonic Sort ------------------
    static void bitonic_merge(unsigned int* arr, unsigned int low, unsigned int cnt, bool dir) {
        if (cnt <= 1) return;

        unsigned int k = cnt >> 1;

        if (k >= 8) { // Use AVX2
            unsigned int i = low;
            unsigned int integers_per_reg = 8;
            unsigned int simd_end = low + k;

            for (; i < simd_end; i += integers_per_reg) {
                __m256i a = _mm256_load_si256((__m256i*)&arr[i]);
                __m256i b = _mm256_load_si256((__m256i*)&arr[i + k]);

                __m256i lo = _mm256_min_epu32(a, b);
                __m256i hi = _mm256_max_epu32(a, b);

                if (!dir) std::swap(lo, hi);

                _mm256_store_si256((__m256i*)(arr + i), lo);
                _mm256_store_si256((__m256i*)(arr + i + k), hi);
            }
        }
        else if (k >= 4) { // Use SSE
            _mm256_zeroupper(); // avoid AVX->SSE penalty
            unsigned int i = low;
            unsigned int integers_per_reg = 4;
            unsigned int simd_end = low + k;

            for (; i < simd_end; i += integers_per_reg) {
                __m128i a = _mm_load_si128((__m128i*)&arr[i]);
                __m128i b = _mm_load_si128((__m128i*)&arr[i + k]);

                __m128i lo = _mm_min_epu32(a, b);
                __m128i hi = _mm_max_epu32(a, b);

                if (!dir) std::swap(lo, hi);

                _mm_store_si128((__m128i*)(arr + i), lo);
                _mm_store_si128((__m128i*)(arr + i + k), hi);
            }
        }
        else { // scalar fallback
            for (unsigned int i = low; i < low + k; ++i) {
                if (dir == (arr[i] > arr[i + k])) {
                    std::swap(arr[i], arr[i + k]);
                }
            }
        }

        bitonic_merge(arr, low, k, dir);
        bitonic_merge(arr, low + k, k, dir);
    }

    static void bitonic_sort_rec(unsigned int* arr, unsigned int low, unsigned int cnt, bool dir) {
        if (cnt <= 1) return;

        unsigned int k = cnt >> 1;

        bitonic_sort_rec(arr, low, k, true);
        bitonic_sort_rec(arr, low + k, k, false);
        bitonic_merge(arr, low, cnt, dir);
    }

    void bitonic_sort(unsigned int* arr, unsigned int n) {
        bitonic_sort_rec(arr, 0, n, true);
    }

}
