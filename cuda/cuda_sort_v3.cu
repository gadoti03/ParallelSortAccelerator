#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <climits>
#include <ctime>

#define THREADS_PER_BLOCK 256
#define THREADS_PER_BLOCK_SHARED_MEMORY_init 128
#define THREADS_PER_BLOCK_SHARED_MEMORY 128

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

void printMemoryFromDevice(int *d_data, int N) {
    std::cout << "\n";
    int *h_temp = new int[N];
    CHECK(cudaMemcpy(h_temp, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        if ((i != 0) && (i % THREADS_PER_BLOCK == 0))
            std::cout << "\n";
        std::cout << h_temp[i] << " ";
    }
    std::cout << "\n";
    delete[] h_temp;
}

void printMemory(int *h_data, int N) {
    std::cout << "\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_data[i] << " ";
    std::cout << "\n";
}

// CUDA kernel for a single compare-and-swap step in Bitonic Sort
__global__ void bitonicCompareGlobal(int *data, int k, int j) {

    unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int ixj = gid ^ j;

    if (ixj > gid) { // single thread handles the pair

        int a = data[gid];   // register
        int b = data[ixj];   // register

        bool ascending = ((gid & k) == 0);

        if ((ascending && a > b) || (!ascending && a < b)) {
            data[gid] = b;
            data[ixj] = a;
        }
    }
}


// CUDA kernel for Bitonic Sort using shared memory (sort first phase, with direction)
__global__ void bitonicCompareLocal_init(int *data, int shift) {
    __shared__ int s[THREADS_PER_BLOCK_SHARED_MEMORY_init];

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;

    unsigned int direction = (blockIdx.x >> shift) & 1;

    // Load from global memory
    s[tid] = data[gid];
    __syncthreads();

    unsigned j, ixj, k, dir, a, b;

    for (k = 2; k <= THREADS_PER_BLOCK_SHARED_MEMORY_init; k <<= 1) {
        for (j = k >> 1; j > 0; j >>= 1) {
            ixj = tid ^ j;

            if (tid < ixj) {
                dir = ((tid & k) == 0) ? direction : 1 - direction;

                // Swap in shared memory, entrambi i thread leggono/scrivono su s[]
                a = s[tid];
                b = s[ixj];
                if ((dir == 0 && a > b) || (dir == 1 && a < b)) {
                    s[tid] = b;
                    s[ixj] = a;
                }
            }
            __syncthreads();
        }
    }

    // Store back to global memory
    data[gid] = s[tid];
}

// CUDA kernel for Bitonic Sort using shared memory (sort first phase, with direction)
__global__ void bitonicCompareLocal(int *data, int shift) {
    __shared__ int s[THREADS_PER_BLOCK_SHARED_MEMORY];

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;

    unsigned int direction = (blockIdx.x >> shift) & 1;

    unsigned int k, j, ixj, dir, a, b;

    // Load from global memory
    s[tid] = data[gid];
    __syncthreads();

    for (k = 2; k <= THREADS_PER_BLOCK_SHARED_MEMORY; k <<= 1) {
        for (j = k >> 1; j > 0; j >>= 1) {
            ixj = tid ^ j;

            if (tid < ixj) {
                dir = ((tid & k) == 0) ? direction : 1 - direction;

                // Swap in shared memory, entrambi i thread leggono/scrivono su s[]
                a = s[tid];
                b = s[ixj];
                if ((dir == 0 && a > b) || (dir == 1 && a < b)) {
                    s[tid] = b;
                    s[ixj] = a;
                }
            }
            __syncthreads();
        }
    }

    // Store back to global memory
    data[gid] = s[tid];
}

// Host function to launch Bitonic Sort
void bitonicSort(int *d_data, int N) {

    int k, j, shift, tmp_k;

    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int numBlocks_shared_init = (N + THREADS_PER_BLOCK_SHARED_MEMORY_init - 1) / THREADS_PER_BLOCK_SHARED_MEMORY_init;
    int numBlocks_shared = (N + THREADS_PER_BLOCK_SHARED_MEMORY - 1) / THREADS_PER_BLOCK_SHARED_MEMORY;

    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
    dim3 gridDim(numBlocks, 1, 1);
    dim3 blockDim_shared_init(THREADS_PER_BLOCK_SHARED_MEMORY_init, 1, 1);
    dim3 gridDim_shared_init(numBlocks_shared_init, 1, 1);
    dim3 blockDim_shared(THREADS_PER_BLOCK_SHARED_MEMORY, 1, 1);
    dim3 gridDim_shared(numBlocks_shared, 1, 1);

    std::cout << "Number of blocks: " << numBlocks << ", Threads per block: " << THREADS_PER_BLOCK << "\n";
    std::cout << "Number of shared memory blocks (init): " << numBlocks_shared_init << ", Threads per shared memory block (init): " << THREADS_PER_BLOCK_SHARED_MEMORY_init << "\n";
    std::cout << "Number of shared memory blocks: " << numBlocks_shared << ", Threads per shared memory block: " << THREADS_PER_BLOCK_SHARED_MEMORY << "\n";

    bitonicCompareLocal_init<<<gridDim_shared_init, blockDim_shared_init>>>(d_data, 0);
    CHECK(cudaPeekAtLastError());    // check kernel launch
    CHECK(cudaDeviceSynchronize());  // check kernel execution

    for (k = THREADS_PER_BLOCK_SHARED_MEMORY_init<<1; k <= N; k <<= 1) {
        for (j = k >> 1; j >= THREADS_PER_BLOCK_SHARED_MEMORY; j >>= 1) {
            bitonicCompareGlobal<<<gridDim, blockDim>>>(d_data, k, j);
            CHECK(cudaPeekAtLastError());    // check kernel launch
            CHECK(cudaDeviceSynchronize());  // check kernel execution

        }

        shift = 0;
        tmp_k = k;
        while (tmp_k > THREADS_PER_BLOCK_SHARED_MEMORY) { tmp_k >>= 1; shift++; }

        bitonicCompareLocal<<<gridDim_shared, blockDim_shared>>>(d_data, shift);
        CHECK(cudaPeekAtLastError());    // check kernel launch
        CHECK(cudaDeviceSynchronize());  // check kernel execution
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