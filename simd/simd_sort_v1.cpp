#include "sort.hpp"
#include <cstring>
#include <immintrin.h>
#include <cmath>

namespace simd { 

    // ------------------ Radix Sort ------------------
    /*
        Using SIMD instructions to accelerate the counting phase of Radix Sort.
        bits_per_pass is fixed to 4
    */
    void radix_sort(unsigned int* arr, unsigned int n) {

        // Parameters for radix sort
        unsigned int bits_per_pass = 4; // fixed to 4 bits
        unsigned int histogram_size = std::pow(2, bits_per_pass); // 2^4
        unsigned int mask = 0x0F; // mask for 4 bits
        unsigned int dim_regs = 16; // 128 bits / 16 Bytes
        unsigned int integers_per_reg = dim_regs / sizeof(unsigned int); // 4 integers per register
        
        // Varbiables
        __m128i vec, max_vec, digits, digits_accumulator, cmp;
        unsigned int i, j, shift, index, mask_cmp, count_set_bits;

        // Determine maximum value to know number of bits
        max_vec = _mm_setzero_si128();

        for (i= 0; i < n; i += integers_per_reg) {
            vec = _mm_load_si128((__m128i*)&arr[i]);
            max_vec = _mm_max_epu32(max_vec, vec);
        }

        alignas(16) unsigned int temp[4];
        _mm_store_si128((__m128i*)temp, max_vec);

        unsigned int max_val = temp[0];
        for(unsigned int j = 1; j < 4; j++)
            if(temp[j] > max_val) max_val = temp[j];

        // Determine number of bits in max_val
        unsigned int max_bits = 32 - __builtin_clz(max_val);

        // Allocate output array and count array
        unsigned int* output = static_cast<unsigned int*>(_mm_malloc(n * sizeof(unsigned int), 16)); // Dynamic aligned allocation
        unsigned int* count = static_cast<unsigned int*>(_mm_malloc(histogram_size * sizeof(unsigned int), 16)); // Dynamic aligned allocation

        for (shift = 0; shift < max_bits; shift += bits_per_pass) { // iterate over each 4 bits

            std::memset(count, 0, histogram_size * sizeof(unsigned int));

            for (i = 0; i < n; i+=integers_per_reg * 4) { // Scelgo 4 cosi da riempire il registro da 16 bytes (4*4=16 -> ogni intero è rappresentato da 4bit)
                // Zero the digits accumulator
                digits_accumulator = _mm_setzero_si128();
                for (j = 0; j < 4; j++) {
                    // Read 4 integers
                    vec = _mm_load_si128((__m128i*)&arr[i+j*integers_per_reg]);
                    // Extract the relevant 4 bits
                    digits = _mm_and_si128(_mm_srli_epi32(vec, shift), _mm_set1_epi32(mask));
                    // Shift digits to their position in the accumulator
                    digits = _mm_slli_epi32(digits, j * bits_per_pass * (4 / bits_per_pass));
                    // Accumulate digits
                    digits_accumulator = _mm_or_si128(digits_accumulator, digits);
                }

                // At this point, digits_accumulator contains 16 bytes, each holding the nibble (4 bits) of an integer
                // We need to count how many times each nibble value (0x0 to 0xF) appears in these 16 bytes

                // Update count array
                for (index = 0; index < histogram_size; index++) {
                    // Compare digits with index
                    cmp = _mm_cmpeq_epi8(digits_accumulator, _mm_set1_epi8(index));
                    // Take the most significant bit of each byte to form a mask (integer with 16 bits)
                    mask_cmp = _mm_movemask_epi8(cmp); // Compact comparison results into a 16-bit integer
                    // Count number of set bits in mask_cmp
                    count_set_bits = __builtin_popcount(mask_cmp); // POPCNT

                    count[index] += count_set_bits;
                }
            }            

            // Transform into cumulative position
            for (unsigned int i = 1; i < histogram_size; i++)
                count[i] += count[i - 1];

            // Fill the output array in reverse order for stability
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
