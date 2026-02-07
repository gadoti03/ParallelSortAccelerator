#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <climits>
#include <ctime>

#define THREADS_PER_BLOCK 256

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

// Device swap
__device__ void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

// Function to get current time in seconds
double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

// CUDA kernel for a single compare-and-swap step in Bitonic Sort
__global__ void bitonicCompareSwap(int *data, int k, int j, int N) {

    unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int ixj = gid ^ j;

    if (ixj > gid) { // prevent double swap
        if ((gid & k) == 0) { // ascending phase
            if (data[gid] > data[ixj])
                swap(data[gid], data[ixj]);
        } else { // descending phase
            if (data[gid] < data[ixj])
                swap(data[gid], data[ixj]);
        }
    }
}

// Host function to launch Bitonic Sort
void bitonicSort(int *d_data, int N) {
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
    dim3 gridDim(numBlocks, 1, 1);

    std::cout << "Number of blocks: " << numBlocks << ", Threads per block: " << THREADS_PER_BLOCK << "\n";
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicCompareSwap<<<gridDim, blockDim>>>(d_data, k, j, N);
            CHECK(cudaPeekAtLastError());    // check kernel launch
            CHECK(cudaDeviceSynchronize());  // check kernel execution
        }
    }
}

int main(int argc, char *argv[]) {

    int N = 1 << 28;

    if (argc > 1) {
        N = atoi(argv[1]);
    } else {
        std::cout << "Using default size N = " << N << "\n";
    }

    int *h_data = new int[N];

    // Get seed from command line or use default
    unsigned int seed = (argc > 1) ? std::atoi(argv[1]) : 1337;

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
    double start = cpuSecond();
    bitonicSort(d_data, N);
    double end = cpuSecond();
    std::cout << "Bitonic Sort completed in " << (end - start) << " seconds.\n";

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