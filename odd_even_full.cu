#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1024          // Number of elements to sort
#define BLOCK_SIZE 128  // Threads per block (multiple of 32)

// Warmup kernel to remove first-launch overhead
__global__ void warmup() {}

// ====================== Serial Odd-Even Sort ======================
void oddEvenSortSerial(int *arr, int n) {
    for (int phase = 0; phase < n; phase++) {
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

// ====================== CUDA Shared Memory Kernel (Single Block) ======================
__global__ void oddEvenShared(int *arr, int n) {
    __shared__ int s_arr[BLOCK_SIZE * 8]; // Adjust size if needed
    int tid = threadIdx.x;

    // Copy from global to shared memory
    if (tid < n)
        s_arr[tid] = arr[tid];
    __syncthreads();  // Barrier

    // Odd-even sort in shared memory
    for (int phase = 0; phase < n; ++phase) {
        int i = 2 * tid + (phase % 2);
        if (i + 1 < n) {
            if (s_arr[i] > s_arr[i+1]) {
                int tmp = s_arr[i];
                s_arr[i] = s_arr[i+1];
                s_arr[i+1] = tmp;
            }
        }
        __syncthreads();  // Barrier after each phase
    }

    // Copy back to global memory
    if (tid < n)
        arr[tid] = s_arr[tid];
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

// ====================== Helper: Print first k elements ======================
void printArray(const char* title, int *arr, int k) {
    printf("%s", title);
    for (int i = 0; i < k; i++) printf("%d ", arr[i]);
    printf("\n");
}

// ====================== Main ======================
int main() {
    int *h_arr = (int*)malloc(N * sizeof(int));
    int *h_serial = (int*)malloc(N * sizeof(int));

    srand(time(NULL)); // Seed once
    for (int i = 0; i < N; i++) {
        h_arr[i] = rand() % 100 + 1; // Random values 1-100
        h_serial[i] = h_arr[i];
    }

    // Warmup kernel
    warmup<<<1,1>>>();
    cudaDeviceSynchronize();

    // ------------------- Serial -------------------
    clock_t start = clock();
    oddEvenSortSerial(h_serial, N);
    clock_t end = clock();
    double serialTime = 1000.0 * (end - start) / CLOCKS_PER_SEC;

    printArray("Serial sorted first 16 elements:\n", h_serial, 16);
    printf("Serial execution time: %f ms\n\n", serialTime);

    // ------------------- CUDA Global Memory -------------------
    int *d_arr;
    cudaMalloc(&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t startEvent, stopEvent;
    float timeGlobal;
    cudaEventCreate(&startEvent); cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);

    for (int phase = 0; phase < N; phase++) {
        oddEvenGlobal<<<numBlocks, BLOCK_SIZE>>>(d_arr, N, phase);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&timeGlobal, startEvent, stopEvent);

    int *h_global = (int*)malloc(N * sizeof(int));
    cudaMemcpy(h_global, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    printArray("CUDA Global sorted first 16 elements:\n", h_global, 16);
    printf("CUDA Global execution time: %f ms\n\n", timeGlobal);

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

    printArray("CUDA Shared (single-block) sorted first 16 elements:\n", h_shared, 16);
    printf("CUDA Shared execution time: %f ms\n\n", timeShared);

    // ------------------- CUDA Multi-block -------------------
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    float timeMulti;
    cudaEventRecord(startEvent);

    for (int phase = 0; phase < N; phase++) {
        oddEvenMultiBlock<<<numBlocks, BLOCK_SIZE>>>(d_arr, N, phase);
        cudaDeviceSynchronize(); // inter-block sync
    }

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&timeMulti, startEvent, stopEvent);

    int *h_multi = (int*)malloc(N * sizeof(int));
    cudaMemcpy(h_multi, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    printArray("CUDA Multi-block sorted first 16 elements:\n", h_multi, 16);
    printf("CUDA Multi-block execution time: %f ms\n\n", timeMulti);

    // Cleanup
    cudaFree(d_arr);
    free(h_arr); free(h_serial);
    free(h_global); free(h_shared); free(h_multi);

    cudaEventDestroy(startEvent); cudaEventDestroy(stopEvent);

    return 0;
}
