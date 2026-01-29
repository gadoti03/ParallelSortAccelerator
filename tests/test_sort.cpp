#include "sort.hpp"
#include "utils.hpp"
#include <iostream>
#include <cstring>

enum class Backend { Serial, SIMD, All };
enum class Algo { Quick, Merge, Radix, Bitonic, All };

// Helper per copiare l'array originale
unsigned int* copy_array(const unsigned int* original, unsigned int n) {
    unsigned int* arr = (unsigned int*) _mm_malloc(n * sizeof(unsigned int), 32);
    std::memcpy(arr, original, n * sizeof(unsigned int));
    return arr;
}

// Funzione generica di test
void run_test(const char* name,
              void (*sort_fn)(unsigned int*, unsigned int),
              const unsigned int* original,
              unsigned int n)
{
    unsigned int* arr = copy_array(original, n);

    double t = utils::measure_time([&]() {
        sort_fn(arr, n);
    });

    std::cout << name << ": " << t << " ms, "
              << (utils::is_sorted(arr, n) ? "sorted" : "NOT sorted")
              << "\n";

    _mm_free(arr);
}

// Test SERIAL
void test_serial(Algo algo, const unsigned int* original, unsigned int n) {
    std::cout << "\n[ SERIAL ]\n";

    if (algo == Algo::Quick || algo == Algo::All)
        run_test("Serial QuickSort", serial::quick_sort, original, n);

    if (algo == Algo::Merge || algo == Algo::All)
        run_test("Serial MergeSort", serial::merge_sort, original, n);

    if (algo == Algo::Radix || algo == Algo::All)
        run_test("Serial Radix", serial::radix_sort, original, n);

    if (algo == Algo::Bitonic || algo == Algo::All)
        run_test("Serial Bitonic", serial::bitonic_sort, original, n);
}

// Test SIMD
void test_simd(Algo algo, const unsigned int* original, unsigned int n) {
    std::cout << "\n[ SIMD ]\n";

    if (algo == Algo::Radix || algo == Algo::All)
        run_test("SIMD Radix", simd::radix_sort, original, n);

    if (algo == Algo::Bitonic || algo == Algo::All)
        run_test("SIMD Bitonic", simd::bitonic_sort, original, n);
}

// Parser argomenti
Backend parse_backend(const char* s) {
    if (!strcmp(s, "serial")) return Backend::Serial;
    if (!strcmp(s, "simd"))   return Backend::SIMD;
    if (!strcmp(s, "all"))    return Backend::All;
    std::cerr << "Invalid backend: " << s << "\n";
    std::exit(1);
}

Algo parse_algo(const char* s) {
    if (!strcmp(s, "quick"))   return Algo::Quick;
    if (!strcmp(s, "merge"))   return Algo::Merge;
    if (!strcmp(s, "radix"))   return Algo::Radix;
    if (!strcmp(s, "bitonic")) return Algo::Bitonic;
    if (!strcmp(s, "all"))     return Algo::All;
    std::cerr << "Invalid algorithm: " << s << "\n";
    std::exit(1);
}

int main(int argc, char** argv) {
    Backend backend = Backend::All;
    Algo algo = Algo::All;

    if (argc >= 2) backend = parse_backend(argv[1]);
    if (argc >= 3) algo = parse_algo(argv[2]);

    const unsigned int n = 1 << 24;
    const int seed = 1337;

    // Array originale
    unsigned int* original = utils::generate_random_array(n, seed);

    // Esecuzione test
    if (backend == Backend::Serial || backend == Backend::All)
        test_serial(algo, original, n);

    if (backend == Backend::SIMD || backend == Backend::All)
        test_simd(algo, original, n);

    _mm_free(original);
    return 0;
}
