#pragma once

namespace serial {

    // QuickSort
    void quick_sort(unsigned int* arr, unsigned int n);

    // MergeSort
    void merge_sort(unsigned int* arr, unsigned int n);

    // Radix Sort Binary (for non-negative integers)
    void radix_sort(unsigned int* arr, unsigned int n);

    // Bitonic Sort (works best for n = power of 2)
    void bitonic_sort(unsigned int* arr, unsigned int n);
    
    // Even-Odd Sort
    // void even_odd_sort(unsigned int* arr, unsigned int n);
}

namespace simd {

    // QuickSort
    // void quicksort(int* arr, unsigned int n);

    // MergeSort
    // void merge_sort(unsigned int* arr, unsigned int n);
    
    // Radix Sort (for non-negative integers)
    void radix_sort(unsigned int* arr, unsigned int n);

    // Bitonic Sort (works best for n = power of 2)
    void bitonic_sort(unsigned int* arr, unsigned int n);

    // Even-Odd Sort
    // void even_odd_sort(unsigned int* arr, unsigned int n);
}

namespace openmp {

    // QuickSort
    // void quicksort(int* arr, unsigned int n);

    // MergeSort
    void merge_sort(unsigned int* arr, unsigned int n);
    
    // Radix Sort (for non-negative integers)
    void radix_sort(unsigned int* arr, unsigned int n);

    // Bitonic Sort (works best for n = power of 2)
    void bitonic_sort(unsigned int* arr, unsigned int n);

    // Even-Odd Sort
    // void even_odd_sort(unsigned int* arr, unsigned int n);
}
