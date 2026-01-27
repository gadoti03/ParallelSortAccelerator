#include "sort.hpp"
#include <algorithm> // std::max_element
#include <cstring>   // std::memcpy

namespace simd {

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
    static void quicksort_rec(int* arr, int left, int right) {
        if(left >= right) return; // Base case (0 or 1 element)
        int pivot = arr[right];
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

    void quicksort(int* arr, int n) {
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
    static void merge(int* arr, int left, int mid, int right) {
        int n1 = mid - left + 1;
        int n2 = right - mid;
        int* L = new int[n1];
        int* R = new int[n2];
        for(int i=0;i<n1;i++) L[i]=arr[left+i];
        for(int i=0;i<n2;i++) R[i]=arr[mid+1+i];
        int i=0,j=0,k=left;
        while(i<n1 && j<n2) arr[k++] = (L[i]<=R[j]) ? L[i++] : R[j++];
        while(i<n1) arr[k++]=L[i++];
        while(j<n2) arr[k++]=R[j++];
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
    void radix_sort(int* arr, int n) {
        int max_val = *std::max_element(arr, arr+n);
        for(int exp=1; max_val/exp > 0; exp*=10) { // iterate over each digit
            int* output = new int[n];
            int count[10] = {0}; // for digits 0-9
            for(int i=0;i<n;i++) count[(arr[i]/exp)%10]++; // count occurrences
            for(int i=1;i<10;i++) count[i]+=count[i-1];  // cumulative count: count[i] now contains actual position of this digit in output[]
            for(int i=n-1;i>=0;i--) { // preserve stability by iterating from end
                output[count[(arr[i]/exp)%10]-1]=arr[i]; // place in output in correct position
                count[(arr[i]/exp)%10]--; // decrease count for same digit
            }
            std::memcpy(arr, output, n*sizeof(int)); // copy back to arr
            delete[] output;
        }
    }

    // ------------------ Bitonic Sort ------------------
    /*
        Oprational Steps:
        1. Create a bitonic sequence by recursively sorting the first half in ascending order (true) and the second half in descending order (false)
        2. Merge the bitonic sequence into a fully sorted sequence
    
        Time Complexity:
            - Average Case: O(log^2 n)
                -> The bitonic sort consists of log n stages, and each stage involves log n comparisons and swaps
    */
    static void bitonic_merge(int* arr, int low, int cnt, bool dir) {
        // cnt: number of elements in the bitonic sequence
        if(cnt>1){
            int k = cnt/2;
            for(int i=low;i<low+k;i++)
                if(dir==(arr[i]>arr[i+k])) std::swap(arr[i],arr[i+k]);
            bitonic_merge(arr, low, k, dir);
            bitonic_merge(arr, low+k, k, dir);
        }
    }

    static void bitonic_sort_rec(int* arr, int low, int cnt, bool dir) {
        if(cnt>1){
            int k = cnt/2;
            bitonic_sort_rec(arr, low, k, true);
            bitonic_sort_rec(arr, low+k, k, false);
            bitonic_merge(arr, low, cnt, dir);
        }
    }

    void bitonic_sort(int* arr, int n) {
        bitonic_sort_rec(arr, 0, n, true);
    }
}