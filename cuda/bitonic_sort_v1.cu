#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <climits>

// Macro to check CUDA errors
#define CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " \
                  << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
        exit(err); \
    } \
}

// Device function to swap two integers
__device__ void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

// CUDA kernel for a single compare-and-swap step in Bitonic Sort
__global__ void bitonicCompareSwap(int *data, int k, int j, int N) {
    /*
        k: current stage size
        j: current comparison distance
        N: total number of elements   
    */
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x; // global thread index

    unsigned int ixj = idx ^ j; // index of the element to compare with

    if (ixj > idx && ixj < N) { // prevent double swap and out-of-bounds
        if ((idx & k) == 0) { // ascending phase
            if (data[idx] > data[ixj])
                swap(data[idx], data[ixj]);
        } else { // descending phase
            if (data[idx] < data[ixj])
                swap(data[idx], data[ixj]);
        }
    }
}

// Host function to launch Bitonic Sort
void bitonicSort(int *d_data, int N) {
    int threadsPerBlock = 1024;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Number of blocks: " << numBlocks << ", Threads per block: " << threadsPerBlock << "\n";

    for (int k = 2; k <= N; k <<= 1) { // all possible sizes of subsequences (2, 4, 8, ..., N)
        for (int j = k >> 1; j > 0; j >>= 1) { // comparison distance (k/2, k/4, ..., 1)
            bitonicCompareSwap<<<numBlocks, threadsPerBlock>>>(d_data, k, j, N);
            CHECK(cudaPeekAtLastError());    // check kernel launch
            CHECK(cudaDeviceSynchronize());  // check kernel execution
        }
    }
}

int main(int argc, char *argv[]) {
    const int N = 1 << 20;
    int *h_data = new int[N];

    // Get seed from command line or use default
    unsigned int seed = (argc > 1) ? std::atoi(argv[1]) : 0;

    // Random number generator covering full int range
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(INT_MIN, INT_MAX);

    // Initialize array
    for (int i = 0; i < N; ++i)
        h_data[i] = dist(gen);

    // Allocate device memory
    int *d_data;
    CHECK(cudaMalloc((void**)&d_data, N * sizeof(int)));

    // Copy data from host to device
    CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

    // Perform Bitonic Sort on the device
    bitonicSort(d_data, N);

    // Copy sorted data back to host
    CHECK(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify results
    for (int i = 0; i < N - 1; ++i) {
        if (h_data[i] > h_data[i + 1]) {
            std::cout << "Sorting failed at index " << i << "\n";
            break;
        }
        if (i == N - 2) {
            std::cout << "Array sorted successfully.\n";
        }
    };

    // Free memory
    CHECK(cudaFree(d_data));
    delete[] h_data;

    return 0;
}