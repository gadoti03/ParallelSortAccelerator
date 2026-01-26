#pragma once

namespace serial {

    // QuickSort
    void quicksort(int* arr, int n);

    // MergeSort
    void merge_sort(int* arr, int n);

    // Radix Sort (for non-negative integers)
    void radix_sort(int* arr, int n);

    // Bitonic Sort (works best for n = power of 2)
    void bitonic_sort(int* arr, int n);

    // Even-Odd Sort
    void even_odd_sort(int* arr, int n);

}
