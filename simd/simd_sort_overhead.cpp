#include "sort.hpp"
#include <algorithm> // std::max_element
#include <cstring>   // std::memcpy
#include <immintrin.h>
#include <cmath>
#include <climits> 
#include <cstdint> 
#include <iostream>

namespace simd { 

    // ------------------ MergeSort ------------------
    /*
        Oprational Steps:
        1. Divide the array into two halves until each sub-array contains a single element
        2. Merge the sub-arrays back together in sorted order
    
        Time Complexity:
            - Average Case: O(n log n)
            - Worst Case: O(n log n)

                -> sorting two already sorted halves takes linear time O(n), and since we are dividing the array log n times, the overall time complexity is O(n log n)
    */
    static void merge(int* arr, int left, int mid, int right) {
        int n1 = mid - left + 1;
        int n2 = right - mid;
        int* L = new int[n1];
        int* R = new int[n2];
        // Fill L and R arrays
        for(int i=0;i<n1;i++) L[i]=arr[left+i];
        for(int i=0;i<n2;i++) R[i]=arr[mid+1+i];
        int i=0,j=0,k=left;
        // Merge the two halves
        while(i<n1 && j<n2) arr[k++] = (L[i]<=R[j]) ? L[i++] : R[j++];
        while(i<n1) arr[k++]=L[i++];
        while(j<n2) arr[k++]=R[j++];
        // Deallocate temporary arrays
        delete[] L; delete[] R;
    }

    void merge_sort(int* arr, int n) {
        if(n <= 1) return;
        int mid = n/2;
        merge_sort(arr, mid);
        merge_sort(arr+mid, n-mid);
        merge(arr, 0, mid-1, n-1); // arr, left, mid, right
    }

    // ------------------ Radix Sort ------------------
    /*
        Oprational Steps:
        1. Find the maximum number to determine the number of digits
        2. Perform counting sort for each digit, starting from the least significant digit to the most significant digit
            -> This is done using a stable sort (like counting sort) to maintain the relative order of elements with the same digit value
    
        Time Complexity:
            - Average Case: O(d*(n + k))
                -> where d is the number of digits in the maximum number, n is the number of elements in the array, and k is the range of the input (for base 10, k=10)
    */

    void radix_sort(unsigned int* arr, unsigned int n) {

        // Parameters for radix sort
        unsigned int bits_per_pass = 4; // fixed to 4 bits
        unsigned int histogram_size = std::pow(2, bits_per_pass); // 2^4
        unsigned int mask = 0xF; // mask for 4 bits
        unsigned int dim_regs = 32; // 256 bits / 32 Bytes
        unsigned int integers_per_reg = dim_regs / sizeof(unsigned int); // 8 integers per register
        
        // Varbiables
        __m256i vec, max_vec, digits, digits_accumulator, cmp, count_index, histogram_0, histogram_1;
        unsigned int i, j, shift, index, mask_cmp, count_set_bits;

        // Determine maximum value to know number of bits
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

        // Determine number of bits in max_val
        unsigned int max_bits = 32 - __builtin_clz(max_val);

        // Allocate output array
        unsigned int* output = new unsigned int[n];

        // Initialize histogram aligned array
        alignas(32) int count[16];

        for (shift = 0; shift < max_bits; shift += bits_per_pass) { // iterate over each 4 bits
            // Reset histogram in registers
            histogram_0 = _mm256_setzero_si256();
            histogram_1 = _mm256_setzero_si256();

            std::memset(count, 0, 16 * sizeof(unsigned int));

            for (i = 0; i < n; i+=integers_per_reg * 4) {
                // Zero the digits accumulator
                digits_accumulator = _mm256_setzero_si256();
                for (j = 0; j < 4; j++) {
                    // Read 8 integers
                    vec = _mm256_load_si256((__m256i*)&arr[i+j*integers_per_reg]);
                    // Extract the relevant 4 bits
                    digits = _mm256_and_si256(_mm256_srli_epi32(vec, shift), _mm256_set1_epi32(mask));
                    // Shift digits to their position in the accumulator
                    digits = _mm256_slli_epi32(digits, j * bits_per_pass * 2);
                    // Accumulate digits
                    digits_accumulator = _mm256_or_si256(digits_accumulator, digits);
                }

                // At this point, digits_accumulator contains 32 bytes, each holding the nibble (4 bits) of an integer
                // We need to count how many times each nibble value (0x0 to 0xF) appears in these 32 bytes
                // Finally, we update the registers that store the histogram

                // Update histogram in registers
                for (index = 0; index < histogram_size; index++) {
                    // Compare digits with index
                    cmp = _mm256_cmpeq_epi8(digits_accumulator, _mm256_set1_epi8(index));
                    // Take the most significant bit of each byte to form a mask (integer with 32 bits)
                    mask_cmp = _mm256_movemask_epi8(cmp);
                    // Count number of set bits in mask_cmp
                    count_set_bits = __builtin_popcount(mask_cmp);
                    
                    if (count_set_bits > 0) {
                        // Create a vector with count to add with only index position set
                        // I would like to use insert, but it needs immediate (constant) index
                        count_index = _mm256_blendv_epi8(
                            _mm256_setzero_si256(),
                            _mm256_set1_epi32(count_set_bits),
                            _mm256_cmpeq_epi32(
                                _mm256_set1_epi32(index % integers_per_reg),
                                _mm256_setr_epi32(0,1,2,3,4,5,6,7)
                            )
                        );
                        // Sum the count to the corresponding register
                        if(index < 8)
                            histogram_0 = _mm256_add_epi32(histogram_0, count_index);
                        else
                            histogram_1 = _mm256_add_epi32(histogram_1, count_index);
                    }
                }
            }

            /*

            __m256i histogram0_init = regs_for_histogram[0];
            __m256i histogram1_init = regs_for_histogram[1];

            // Accumulate counts from registers to main histogram
            for (unsigned int k = 1; k < histogram_size; k++) {
                regs_for_histogram[0] = _mm256_add_epi32(
                    _mm256_srli_si256(
                        histogram0_init, 
                        k * sizeof(unsigned int)
                    ),
                    regs_for_histogram[0]
                );

                regs_for_histogram[1] = _mm256_add_epi32(
                    _mm256_srli_si256(
                        histogram1_init, 
                        k * sizeof(unsigned int)
                    ),
                    regs_for_histogram[1]
                );    
                
                regs_for_histogram[1] = _mm256_add_epi32(
                    _mm256_slli_si256(
                        histogram0_init, 
                        (8-k) * sizeof(unsigned int)
                    ),
                    regs_for_histogram[1]
                );    
            }
            */

            // Update main histogram from registers
            // _mm256_store_si256((__m256i*)&count[0], histogram_0);                
            // _mm256_store_si256((__m256i*)&count[8], histogram_1);                

            // Transform into cumulative position
            for (unsigned int i = 1; i < 16; i++)
                count[i] += count[i - 1];

            // unsigned int* output = static_cast<unsigned int*>(_mm_malloc(n * sizeof(unsigned int), 32));

            // Fill the output array in reverse order for stability
            for (unsigned int i = n; i-- > 0; ) {
                unsigned int digit = (arr[i] >> shift) & mask;
                output[count[digit] - 1] = arr[i];
                count[digit]--;
            }

            std::memcpy(arr, output, n * sizeof(unsigned int));
        }
        delete[] output;
    }

    // ------------------ Bitonic Sort ------------------
    /*
        Oprational Steps:
        1. Create a bitonic sequence by recursively sorting the first half in ascending order (true) and the second half in descending order (false)
        2. Merge the bitonic sequence into a fully sorted sequence
    
        Time Complexity:
            - Average Case: O(n*log^2 n)
                -> The bitonic sort consists of log n stages, and each stage involves log n comparisons and swaps
    */

    static void bitonic_merge(unsigned int* arr, int low, int cnt, bool dir) {
        if(cnt > 1) {
            int k = cnt / 2;
            int vec_size = 8; // 8 uint32 per vettore

            // SIMD loop
            for(int i = low; i < low + k - vec_size + 1; i += vec_size) {
                __m256i a = _mm256_loadu_si256((__m256i*)(arr + i));
                __m256i b = _mm256_loadu_si256((__m256i*)(arr + i + k));

                __m256i lo = _mm256_min_epu32(a, b);
                __m256i hi = _mm256_max_epu32(a, b);

                if(!dir) std::swap(lo, hi); // invert if descending

                _mm256_storeu_si256((__m256i*)(arr + i), lo);
                _mm256_storeu_si256((__m256i*)(arr + i + k), hi);
            }

            // Scalar loop for remaining elements
            for(int i = low + (k / vec_size) * vec_size; i < low + k; i++) {
                if(dir == (arr[i] > arr[i + k])) std::swap(arr[i], arr[i + k]);
            }

            // Recursive merge
            bitonic_merge(arr, low, k, dir);
            bitonic_merge(arr, low + k, k, dir);
        }
    }

    static void bitonic_sort_rec(unsigned int* arr, int low, int cnt, bool dir) {
        if(cnt > 1) {
            int k = cnt / 2;
            bitonic_sort_rec(arr, low, k, true);        // sort ascending
            bitonic_sort_rec(arr, low + k, k, false);   // sort descending
            bitonic_merge(arr, low, cnt, dir);
        }
    }

    void bitonic_sort(unsigned int* arr, int n) {
        bitonic_sort_rec(arr, 0, n, true);
    }

}