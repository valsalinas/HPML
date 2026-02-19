#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 32            // Number of elements (for testing small array)
#define BLOCK_SIZE 32   // Threads per block (multiple of 32)

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

    // Copy global to shared memory
    if (tid < n) s_arr[tid] = d_arr[tid];
    __syncthreads();

    // Odd-even sort in shared memory
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

    // Copy back to global memory
    if (tid < n) d_arr[tid] = s_arr[tid];
}

// ====================== Multi-block Kernel ======================
__global__ void oddEvenMultiBlock(int *d_arr, int n, int phase) {
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

// ====================== Main ======================
int main() {
    int *h_arr = (int*)malloc(N * sizeof(int));
    int *h_serial = (int*)malloc(N * sizeof(int));

    // Initialize array
    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        h_arr[i] = rand() % 100;  // small numbers for easy debugging
        h_serial[i] = h_arr[i];
    }

    // Print original array
    printf("Original array first 16 elements:\n");
    for (int i = 0; i < N && i < 16; i++) printf("%d ", h_arr[i]);
    printf("\n");

    // ------------------- Warmup -------------------
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

    // ------------------- CUDA Global Memory -------------------
    int *d_arr;
    cudaMalloc(&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t startEvent, stopEvent;
    float timeGlobal;
    cudaEventCreate(&startEvent); cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);

    for (int phase = 0; phase < N; ++phase) {
        oddEvenGlobal<<<numBlocks, BLOCK_SIZE>>>(d_arr, N, phase);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&timeGlobal, startEvent, stopEvent);

    int *h_global = (int*)malloc(N * sizeof(int));
    cudaMemcpy(h_global, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("CUDA Global sorted first 16 elements:\n");
    for (int i = 0; i < N && i < 16; i++) printf("%d ", h_global[i]);
    printf("\nCUDA Global execution time: %f ms\n\n", timeGlobal);

    // ------------------- CUDA Shared Memory -------------------
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    float timeShared;
    cudaEventRecord(startEvent);

    oddEvenShared<<<1, BLOCK_SIZE>>>(d_arr, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&timeShared, startEvent, stopEvent);

    int *h_shared = (int*)malloc(N * sizeof(int));
    cudaMemcpy(h_shared, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("CUDA Shared (single-block) sorted first 16 elements:\n");
    for (int i = 0; i < N && i < 16; i++) printf("%d ", h_shared[i]);
    printf("\nCUDA Shared execution time: %f ms\n\n", timeShared);

    // ------------------- CUDA Multi-block -------------------
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    float timeMulti;
    cudaEventRecord(startEvent);

    for (int phase = 0; phase < N; ++phase) {
        oddEvenMultiBlock<<<numBlocks, BLOCK_SIZE>>>(d_arr, N, phase);
        cudaDeviceSynchronize(); // inter-block sync
    }

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&timeMulti, startEvent, stopEvent);

    int *h_multi = (int*)malloc(N * sizeof(int));
    cudaMemcpy(h_multi, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("CUDA Multi-block sorted first 16 elements:\n");
    for (int i = 0; i < N && i < 16; i++) printf("%d ", h_multi[i]);
    printf("\nCUDA Multi-block execution time: %f ms\n\n", timeMulti);

    // ------------------- Cleanup -------------------
    cudaFree(d_arr);
    free(h_arr); free(h_serial); free(h_global); free(h_shared); free(h_multi);
    cudaEventDestroy(startEvent); cudaEventDestroy(stopEvent);

    return 0;
}