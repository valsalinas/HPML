#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 64            // Can increase for testing larger arrays
#define BLOCK_SIZE 16   // Threads per block (adjustable)

// Warmup kernel to remove first-launch overhead
__global__ void warmup() {}

// ====================== Serial Odd-Even Sort ======================
void oddEvenSortSerial(int *arr, int n) {
    for (int phase = 0; phase < n; ++phase) {
        int start = phase % 2;
        for (int i = start; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                int tmp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = tmp;
            }
        }
    }
}

// ====================== CUDA Global Memory Kernel ======================
__global__ void oddEvenGlobal(int *d_arr, int n, int phase) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 2 * tid + (phase % 2);
    if (i + 1 < n) {
        if (d_arr[i] > d_arr[i + 1]) {
            int tmp = d_arr[i];
            d_arr[i] = d_arr[i + 1];
            d_arr[i + 1] = tmp;
        }
    }
}

// ====================== CUDA Shared Memory Kernel (single-block) ======================
__global__ void oddEvenShared(int *d_arr, int n) {
    __shared__ int s_arr[BLOCK_SIZE];

    int tid = threadIdx.x;

    if (tid < n) s_arr[tid] = d_arr[tid];
    __syncthreads();

    for (int phase = 0; phase < n; ++phase) {
        int i = 2 * tid + (phase % 2);
        if (i + 1 < n) {
            if (s_arr[i] > s_arr[i + 1]) {
                int tmp = s_arr[i];
                s_arr[i] = s_arr[i + 1];
                s_arr[i + 1] = tmp;
            }
        }
        __syncthreads();
    }

    if (tid < n) d_arr[tid] = s_arr[tid];
}

// ====================== CUDA Multi-Block Shared Memory Kernel ======================
__global__ void oddEvenSharedMultiBlock(int *arr, int n, int phase) {
    __shared__ int s_arr[BLOCK_SIZE + 1];  // +1 for boundary element

    int start = blockIdx.x * blockDim.x * 2 + (phase % 2); // start index for this block
    int tid = threadIdx.x;

    // Load data into shared memory
    if (start + tid < n)
        s_arr[tid] = arr[start + tid];

    // Load the boundary element for cross-block swap
    if (tid == BLOCK_SIZE && start + tid < n)
        s_arr[BLOCK_SIZE] = arr[start + BLOCK_SIZE];

    __syncthreads();

    // Odd-even swaps inside shared memory
    for (int i = tid; i < BLOCK_SIZE; i += blockDim.x) {
        if (i + 1 <= BLOCK_SIZE && i + start + 1 < n) {
            if (s_arr[i] > s_arr[i + 1]) {
                int tmp = s_arr[i];
                s_arr[i] = s_arr[i + 1];
                s_arr[i + 1] = tmp;
            }
        }
    }
    __syncthreads();

    // Write back results
    if (start + tid < n)
        arr[start + tid] = s_arr[tid];
}

// ====================== Main ======================
int main() {
    int *h_arr = (int*)malloc(N * sizeof(int));
    int *h_serial = (int*)malloc(N * sizeof(int));

    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        h_arr[i] = rand() % 100;
        h_serial[i] = h_arr[i];
    }

    printf("Original array first 16 elements:\n");
    for (int i = 0; i < N && i < 16; i++) printf("%d ", h_arr[i]);
    printf("\n");

    // Warmup
    warmup<<<1,1>>>();
    cudaDeviceSynchronize();

    // ------------------- Serial -------------------
    clock_t start = clock();
    oddEvenSortSerial(h_serial, N);
    clock_t end = clock();
    double serialTime = 1000.0 * (end - start) / CLOCKS_PER_SEC;

    printf("Serial sorted first 16 elements:\n");
    for (int i = 0; i < N && i < 16; i++) printf("%d ", h_serial[i]);
    printf("\nSerial execution time: %f ms\n\n", serialTime);

    // ------------------- CUDA Multi-Block Shared Memory -------------------
    int *d_arr;
    cudaMalloc(&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (N + BLOCK_SIZE*2 - 1) / (BLOCK_SIZE*2);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent); cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);

    for (int phase = 0; phase < N; ++phase) {
        oddEvenSharedMultiBlock<<<numBlocks, BLOCK_SIZE>>>(d_arr, N, phase);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    float timeMulti = 0;
    cudaEventElapsedTime(&timeMulti, startEvent, stopEvent);

    int *h_multi = (int*)malloc(N * sizeof(int));
    cudaMemcpy(h_multi, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("CUDA Multi-block Shared sorted first 16 elements:\n");
    for (int i = 0; i < N && i < 16; i++) printf("%d ", h_multi[i]);
    printf("\nCUDA Multi-block execution time: %f ms\n\n", timeMulti);

    // ------------------- Verification -------------------
    int pass = 1;
    for (int i = 0; i < N; i++) {
        if (h_serial[i] != h_multi[i]) {
            pass = 0;
            break;
        }
    }
    printf("Correctness check: %s\n", pass ? "PASS" : "FAIL");

    // Cleanup
    cudaFree(d_arr);
    free(h_arr); free(h_serial); free(h_multi);
    cudaEventDestroy(startEvent); cudaEventDestroy(stopEvent);

    return 0;
}
