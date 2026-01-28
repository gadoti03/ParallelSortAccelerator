#include "sort.hpp"
#include <algorithm> // std::max_element
#include <cstring>   // std::memcpy
#include <immintrin.h>
#include <cmath>
#include <climits> 
#include <cstdint> 
#include <iostream>

namespace openmp {

    void quick_sort(unsigned int* arr, unsigned int n) {
        #pragma omp parallel for
        for (unsigned int i = 0; i < n; ++i) {
            // codice quick sort parallelo
        }
    }

    void merge_sort(unsigned int* arr, unsigned int n) {
        #pragma omp parallel for
        for (unsigned int i = 0; i < n; ++i) {
            // codice merge sort parallelo
        }
    }

    void radix_sort(unsigned int* arr, unsigned int n) {
        #pragma omp parallel for
        for (unsigned int i = 0; i < n; ++i) {
            // codice radix sort parallelo
        }
    }

    void bitonic_sort(unsigned int* arr, unsigned int n) {
        #pragma omp parallel for
        for (unsigned int i = 0; i < n; ++i) {
            // codice bitonic sort parallelo
        }    
    }

}