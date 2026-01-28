#include "sort.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace serial {

    // ------------------ QuickSort ------------------
    /*
        Oprational Steps:
        1. Choose a pivot element from the array (here, I choose the last element)
        2. Partition the array into two halves: elements less than the pivot and elements greater
            -> I get a partition with:
                - on the left side, all elements < pivot
                - on the right side, all elements >= pivot
        3. Recursively apply the same logic to the left and right sub-arrays
    
        Time Complexity:
            - Average Case: O(n log n)
                -> the level of the partitioning tree becomes log n and at each level we do O(n) work to partition
            - Worst Case: O(n^2)
    */
    static void quicksort_rec(unsigned int* arr, int left, int right) {
        if(left >= right) return; // Base case (0 or 1 element)
        unsigned int pivot = arr[right];
        int i = left;
        for(int j = left; j < right; ++j) {
            if(arr[j] < pivot) {
                std::swap(arr[i], arr[j]);
                i++;
            }
        }
        std::swap(arr[i], arr[right]); // Place pivot in correct position
        quicksort_rec(arr, left, i-1);
        quicksort_rec(arr, i+1, right);
    }

    void quicksort(unsigned int* arr, int n) {
        quicksort_rec(arr, 0, n-1);
    }

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
    static void merge(unsigned int* arr, int left, int mid, int right) {
        int n1 = mid - left + 1;
        int n2 = right - mid;
        // Create temporary arrays
        unsigned int* L = new unsigned int[n1];
        unsigned int* R = new unsigned int[n2];
        // Fill L and R arrays
        for(int i=0;i<n1;i++) L[i]=arr[left+i];
        for(int i=0;i<n2;i++) R[i]=arr[mid+1+i];
        // Merge the two halves
        int i=0,j=0,k=left;
        while(i<n1 && j<n2) arr[k++] = (L[i]<=R[j]) ? L[i++] : R[j++];
        while(i<n1) arr[k++]=L[i++];
        while(j<n2) arr[k++]=R[j++];
        // Deallocate temporary arrays
        delete[] L; delete[] R;
    }

    void merge_sort(unsigned int* arr, int n) {
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
        2. Perform counting sort for each digit, starting from the least significant digit or byte to the most significant one  
            -> This is done using a stable sort (like counting sort) to maintain the relative order of elements with the same digit value
    
        Time Complexity:
            - Average Case: O(d*(n + k))
                -> where d is the number of digits or byte in the maximum number, n is the number of elements in the array, and k is the range of the input (for base 10, k=10; for byte, k=256)
    */
    void radix_sort_binary(unsigned int* arr, unsigned int n) {

        // Parameters for radix sort
        unsigned int bits_per_pass = 4; // number of bits per pass
        unsigned int bucket_size = 1u << bits_per_pass; // e.g., 2^bits_per_pass
        unsigned int mask = bucket_size - 1;            // mask to extract bits (e.g., for 8 bits, mask = 0xFF)

        // Determine maximum value to know number of bits
        unsigned int max_val = *std::max_element(arr, arr + n);

        // Determine number of bits in max_val
        unsigned int max_bits = 0;
        unsigned int tmp = max_val;
        while (tmp > 0) {
            tmp >>= 1;
            max_bits++;
        }

        // Variables
        unsigned int shift, i;

        // Allocate output array and count array
        unsigned int* output = new unsigned int[n];
        unsigned int* count  = new unsigned int[bucket_size];

        for (shift = 0; shift < max_bits; shift += bits_per_pass) {

            // Reset histogram
            std::memset(count, 0, bucket_size * sizeof(unsigned int));

            // Count occurrences of each value
            for (i = 0; i < n; i++) {
                unsigned int digit = (arr[i] >> shift) & mask;
                count[digit]++;
            }

            // Transform into cumulative position
            for (i = 1; i < bucket_size; i++)
                count[i] += count[i - 1];

            // Fill the output array in reverse order for stability
            for (i = n; i-- > 0; ) {
                unsigned int digit = (arr[i] >> shift) & mask;
                output[count[digit] - 1] = arr[i];
                count[digit]--;
            }

            // Copy back to arr
            std::memcpy(arr, output, n * sizeof(unsigned int));
        }

        delete[] output;
        delete[] count;
    }

    // ------------------ Bitonic Sort ------------------
    /*
        Oprational Steps:
        1. Create a bitonic sequence by recursively sorting the first half in ascending order (true) and the second half in descending order (false)
        2. Merge the bitonic sequence into a fully sorted sequence
    
        Time Complexity:
            - Average Case: O(n log^2 n)
                -> The bitonic sort consists of log n stages, and each stage involves log n comparisons and swaps
    */
    static void bitonic_merge(unsigned int* arr, unsigned int low, unsigned int cnt, bool dir) {
        // cnt: number of elements in the bitonic sequence
        if(cnt>1){
            unsigned int k = cnt/2;
            for(unsigned int i=low;i<low+k;i++)
                if(dir==(arr[i]>arr[i+k])) std::swap(arr[i],arr[i+k]);
            bitonic_merge(arr, low, k, dir);
            bitonic_merge(arr, low+k, k, dir);
        }
    }

    static void bitonic_sort_rec(unsigned int* arr, unsigned int low, unsigned int cnt, bool dir) {
        if(cnt>1){
            unsigned int k = cnt/2;
            bitonic_sort_rec(arr, low, k, true);        // sort ascending
            bitonic_sort_rec(arr, low+k, k, false);     // sort descending
            bitonic_merge(arr, low, cnt, dir);
        }
    }

    void bitonic_sort(unsigned int* arr, unsigned int n) {
        bitonic_sort_rec(arr, 0, n, true);
    }
}