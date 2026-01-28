#include "sort.hpp"
#include "utils.hpp"
#include <iostream>

int main() {
    
    const unsigned int n = 1 << 24;
    const int seed = 1337;

    // Generate original random array
    unsigned int* original = utils::generate_random_array(n, seed);

    // Arrays for each algorithm
    // unsigned int* arr_quick = utils::generate_random_array(n, seed);
    // unsigned int* arr_merge = utils::generate_random_array(n, seed);
    unsigned int* arr_radix = utils::generate_random_array(n, seed);
    unsigned int* arr_radix_binary = utils::generate_random_array(n, seed);
    unsigned int* arr_bitonic = utils::generate_random_array(n, seed);
    // unsigned int* arr_evenodd = utils::generate_random_array(n, seed);

    unsigned int* arr_radix_simd = utils::generate_random_array(n, seed);
    unsigned int* arr_bitonic_simd = utils::generate_random_array(n, seed);

    std::cout << "Benchmarking serial algorithms on " << n << " elements\n\n";

    if (utils::arrays_equal(original, arr_radix_binary, n))
        std::cout << "arr_radix_binary matches original\n";
    else
        std::cout << "arr_radix_binary does NOT match original\n";

    if (utils::arrays_equal(original, arr_radix_simd, n))
        std::cout << "arr_radix_simd matches original\n";
    else
        std::cout << "arr_radix_simd does NOT match original\n";
    std::cout << "\n";
 
    /*
    // QuickSort
    double t_quick = utils::measure_time([&]() {
        serial::quicksort(arr_quick, n);
    });
    std::cout << "QuickSort: " << t_quick << " ms, "
              << (utils::is_sorted(arr_quick, n) ? "sorted" : "NOT sorted") << "\n";

    // MergeSort
    double t_merge = utils::measure_time([&]() {
        serial::merge_sort(arr_merge, n);
    });
    std::cout << "MergeSort: " << t_merge << " ms, "
              << (utils::is_sorted(arr_merge, n) ? "sorted" : "NOT sorted") << "\n";
    
    // RadixSort
    double t_radix = utils::measure_time([&]() {
        serial::radix_sort(arr_radix, n);
    });
    std::cout << "RadixSort: " << t_radix << " ms, "
              << (utils::is_sorted(arr_radix, n) ? "sorted" : "NOT sorted") << "\n";
    
    // RadixSortBinary
    double t_radix_binary = utils::measure_time([&]() {
        serial::radix_sort_binary(arr_radix_binary, n);
    });
    std::cout << "RadixSort Binary: " << t_radix_binary << " ms, "
              << (utils::is_sorted(arr_radix_binary, n) ? "sorted" : "NOT sorted") << "\n";
    */
    // BitonicSort
    double t_bitonic = utils::measure_time([&]() {
        serial::bitonic_sort(arr_bitonic, n);
    });
    std::cout << "BitonicSort: " << t_bitonic << " ms, "
              << (utils::is_sorted(arr_bitonic, n) ? "sorted" : "NOT sorted") << "\n";
    /*
    std::cout << "\nBenchmarking SIMD algorithms on " << n << " elements\n\n";

    // BitonicSort
    double t_bitonic_simd = utils::measure_time([&]() {
        simd::bitonic_sort(arr_bitonic_simd, n);
    });
    std::cout << "BitonicSort: " << t_bitonic_simd << " ms, "
              << (utils::is_sorted(arr_bitonic_simd, n) ? "sorted" : "NOT sorted") << "\n";
    */

    // RadixSortBinary
    double t_radix_binary = utils::measure_time([&]() {
        serial::radix_sort_binary(arr_radix_binary, n);
    });
    std::cout << "RadixSort Binary: " << t_radix_binary << " ms, "
              << (utils::is_sorted(arr_radix_binary, n) ? "sorted" : "NOT sorted") << "\n";
    
    std::cout << "\nBenchmarking SIMD Radix Sort on " << n << " elements\n\n";
    // RadixSortBinary
    double t_radix_binary_simd = utils::measure_time([&]() {
        simd::radix_sort(arr_radix_simd, n);
    });
    std::cout << "RadixSort Binary: " << t_radix_binary_simd << " ms, "
              << (utils::is_sorted(arr_radix_simd, n) ? "sorted" : "NOT sorted") << "\n";

    // BitonicSort
    double t_bitonic_simd = utils::measure_time([&]() {
        simd::bitonic_sort(arr_bitonic_simd, n);
    });
    std::cout << "BitonicSort: " << t_bitonic_simd << " ms, "
              << (utils::is_sorted(arr_bitonic_simd, n) ? "sorted" : "NOT sorted") << "\n";

    // Free memory
    _mm_free(original);
    // _mm_free(arr_quick);
    // _mm_free(arr_merge);
    _mm_free(arr_radix);
    _mm_free(arr_radix_binary);
    _mm_free(arr_bitonic);
    // _mm_free(arr_evenodd);

    _mm_free(arr_bitonic_simd);
    _mm_free(arr_radix_simd);

    return 0;
}
