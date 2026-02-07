#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <climits>
#include <ctime>

#define THREADS_PER_BLOCK 256
#define THREADS_PER_BLOCK_SHARED_MEMORY 1024

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
    int *h_temp = new int[N];
    CHECK(cudaMemcpy(h_temp, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        std::cout << h_temp[i] << " ";
        if (i % THREADS_PER_BLOCK == 0)
            std::cout << "\n";
    }
    std::cout << "\n";
    delete[] h_temp;
}

void printMemory(int *h_data, int N) {
    for (int i = 0; i < N; ++i)
        std::cout << h_data[i] << " ";
    std::cout << "\n";
}

// CUDA kernel for a single compare-and-swap step in Bitonic Sort
__global__ void bitonicCompareGlobal(int *data, int k, int j) {

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

    /*
int ascending = ((gid & k) == 0); // 1 se fase ascendente, 0 se discendente

// calcolo condizione di swap come 0 o 1
int cond = (data[gid] > data[ixj]) * ascending | (data[gid] < data[ixj]) * (1 - ascending);

// condizione in mask: 0xFFFFFFFF se vero, 0x0 se falso
int mask = -cond;

// swap usando bitwise
int tmp = data[gid];
data[gid] = (data[gid] & ~mask) | (data[ixj] & mask);
data[ixj] = (data[ixj] & ~mask) | (tmp & mask);
    */
}

// CUDA kernel for Bitonic Sort using shared memory (sort first phase, with direction)
__global__ void bitonicCompareLocal(int *data) {

    __shared__ int shared_memory[THREADS_PER_BLOCK_SHARED_MEMORY];

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int direction = blockIdx.x & 1; // 0: ascending, 1: descending

    // load of the block in shared memory
    shared_memory[tid] = data[gid];
    __syncthreads();

    // local Bitonic Sort in shared memory
    for (unsigned int k = 2; k <= THREADS_PER_BLOCK_SHARED_MEMORY; k <<= 1) {
        for (unsigned int j = k >> 1; j > 0; j >>= 1) {

            unsigned int ixj = tid ^ j;

            if (ixj > tid) {
                if (direction == 0) { // ascending
                    if ((tid & k) == 0 && shared_memory[tid] > shared_memory[ixj])
                        swap(shared_memory[tid], shared_memory[ixj]);
                    if ((tid & k) != 0 && shared_memory[tid] < shared_memory[ixj])
                        swap(shared_memory[tid], shared_memory[ixj]);
                } else { // descending
                    if ((tid & k) == 0 && shared_memory[tid] < shared_memory[ixj])
                        swap(shared_memory[tid], shared_memory[ixj]);
                    if ((tid & k) != 0 && shared_memory[tid] > shared_memory[ixj])
                        swap(shared_memory[tid], shared_memory[ixj]);
                }
            }

/*
int ascending = (direction == 0);        // 1 se fase ascendente, 0 se discendente
int tid_phase = (tid & k) == 0;         // 1 se "prima metà" del bit k, 0 se "seconda metà"

// condizione di swap: 1 se dobbiamo scambiare, 0 altrimenti
int cond = ((ascending && tid_phase && shared_memory[tid] > shared_memory[ixj]) ||
            (ascending && !tid_phase && shared_memory[tid] < shared_memory[ixj]) ||
            (!ascending && tid_phase && shared_memory[tid] < shared_memory[ixj]) ||
            (!ascending && !tid_phase && shared_memory[tid] > shared_memory[ixj]));

// trasformiamo cond in mask: 0xFFFFFFFF se vero, 0x0 se falso
int mask = -cond;

// swap branchless
int tmp = shared_memory[tid];
shared_memory[tid] = (shared_memory[tid] & ~mask) | (shared_memory[ixj] & mask);
shared_memory[ixj]  = (shared_memory[ixj]  & ~mask) | (tmp & mask);
*/

            __syncthreads();
        }
    }

    // store of the block back to global memory
    data[gid] = shared_memory[tid];
}

// Host function to launch Bitonic Sort
void bitonicSort(int *d_data, int N) {

    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int numBlocks_shared = (N + THREADS_PER_BLOCK_SHARED_MEMORY - 1) / THREADS_PER_BLOCK_SHARED_MEMORY;

    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
    dim3 gridDim(numBlocks, 1, 1);
    dim3 blockDim_shared(THREADS_PER_BLOCK_SHARED_MEMORY, 1, 1);
    dim3 gridDim_shared(numBlocks_shared, 1, 1);

    std::cout << "Number of blocks: " << numBlocks << ", Threads per block: " << THREADS_PER_BLOCK << "\n";
    std::cout << "Number of shared memory blocks: " << numBlocks_shared << ", Threads per shared memory block: " << THREADS_PER_BLOCK_SHARED_MEMORY << "\n";

    bitonicCompareLocal<<<gridDim_shared, blockDim_shared>>>(d_data);
    CHECK(cudaPeekAtLastError());    // check kernel launch
    CHECK(cudaDeviceSynchronize());  // check kernel execution

    for (int k = THREADS_PER_BLOCK_SHARED_MEMORY<<1; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicCompareGlobal<<<gridDim, blockDim>>>(d_data, k, j);
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