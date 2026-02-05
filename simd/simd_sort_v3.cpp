#include "sort.hpp"
#include <cstring>
#include <immintrin.h>
#include <cmath>

namespace simd { 

    // ------------------ Radix Sort ------------------
    void radix_sort(unsigned int* arr, unsigned int n) {
        unsigned int bits_per_pass = 4;
        unsigned int histogram_size = 1 << bits_per_pass;
        unsigned int mask = 0x0F;

        unsigned int dim_regs = 32; // 256-bit
        unsigned int integers_per_reg = dim_regs / sizeof(unsigned int);

        __m256i vec, max_vec, digits, digits_accumulator, cmp;
        unsigned int i, j, shift, index, mask_cmp, count_set_bits;

        max_vec = _mm256_setzero_si256();
        for (i= 0; i < n; i += integers_per_reg) {
            vec = _mm256_load_si256((__m256i*)&arr[i]);
            max_vec = _mm256_max_epi32(max_vec, vec);
        }

        alignas(32) unsigned int temp[8];
        _mm256_store_si256((__m256i*)temp, max_vec);

        unsigned int max_val = temp[0];
        for(unsigned int j = 1; j < 8; j++)
            if(temp[j] > max_val) max_val = temp[j];

        unsigned int max_bits = 32 - __builtin_clz(max_val);

        unsigned int* output = static_cast<unsigned int*>(_mm_malloc(n * sizeof(unsigned int), 32));
        unsigned int* count = static_cast<unsigned int*>(_mm_malloc(histogram_size * sizeof(unsigned int), 32));

        for (shift = 0; shift < max_bits; shift += bits_per_pass) {
            std::memset(count, 0, histogram_size * sizeof(unsigned int));

            for (i = 0; i < n; i+=integers_per_reg * 4) {
                digits_accumulator = _mm256_setzero_si256();
                for (j = 0; j < 4; j++) {
                    vec = _mm256_load_si256((__m256i*)&arr[i+j*integers_per_reg]);
                    digits = _mm256_and_si256(_mm256_srli_epi32(vec, shift), _mm256_set1_epi32(mask));
                    digits = _mm256_slli_epi32(digits, j * bits_per_pass * (8 / bits_per_pass));
                    digits_accumulator = _mm256_or_si256(digits_accumulator, digits);
                }

                for (index = 0; index < histogram_size; index++) {
                    cmp = _mm256_cmpeq_epi8(digits_accumulator, _mm256_set1_epi8(index));
                    mask_cmp = _mm256_movemask_epi8(cmp);
                    count_set_bits = __builtin_popcount(mask_cmp);
                    count[index] += count_set_bits;
                }
            }

            for (unsigned int i = 1; i < histogram_size; i++)
                count[i] += count[i - 1];

            for (unsigned int i = n; i-- > 0; ) {
                unsigned int digit = (arr[i] >> shift) & mask;
                output[count[digit] - 1] = arr[i];
                count[digit]--;
            }

            std::memcpy(arr, output, n * sizeof(unsigned int));
        }

        _mm_free(output);
        _mm_free(count);
    }

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
