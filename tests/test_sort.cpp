#include "sort.hpp"
#include "utils.hpp"
#include <iostream>

int main() {
    const int n = 1 << 24; // 16777216 elements for test
    const int seed = 12345;

    // Generate original random array
    int* original = utils::generate_random_array(n, seed);

    // Arrays for each algorithm
    int* arr_quick = utils::generate_random_array(n, seed);
    int* arr_merge = utils::generate_random_array(n, seed);
    int* arr_radix = utils::generate_random_array(n, seed);
    int* arr_bitonic = utils::generate_random_array(n, seed);
    int* arr_evenodd = utils::generate_random_array(n, seed);

    std::cout << "Benchmarking serial algorithms on " << n << " elements\n\n";

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

    // BitonicSort
    double t_bitonic = utils::measure_time([&]() {
        serial::bitonic_sort(arr_bitonic, n);
    });
    std::cout << "BitonicSort: " << t_bitonic << " ms, "
              << (utils::is_sorted(arr_bitonic, n) ? "sorted" : "NOT sorted") << "\n";

    /*
    // Even-Odd Sort
    double t_evenodd = utils::measure_time([&]() {
        serial::even_odd_sort(arr_evenodd, n);
    });
    std::cout << "Even-Odd Sort: " << t_evenodd << " ms, "
              << (utils::is_sorted(arr_evenodd, n) ? "sorted" : "NOT sorted") << "\n";
    */

    // Free memory
    _mm_free(original);
    _mm_free(arr_quick);
    _mm_free(arr_merge);
    _mm_free(arr_radix);
    _mm_free(arr_bitonic);
    _mm_free(arr_evenodd);

    return 0;
}
