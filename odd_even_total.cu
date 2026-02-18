#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

#define N 1024  // Number of elements
#define BLOCK_SIZE 256

// ------------------------
// Serial Odd-Even Transposition Sort
// ------------------------
void oddEvenSortSerial(int *arr, int n) {
    int sorted = 0;
    while (!sorted) {
        sorted = 1;

        // Even phase
        for (int i = 0; i <= n - 2; i += 2) {
            if (arr[i] > arr[i + 1]) {
                int tmp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = tmp;
                sorted = 0;
            }
        }

        // Odd phase
        for (int i = 1; i <= n - 2; i += 2) {
            if (arr[i] > arr[i + 1]) {
                int tmp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = tmp;
                sorted = 0;
            }
        }
    }
}

// ------------------------
// CUDA Kernel: Single-block Global Memory
// ------------------------
__global__ void oddEvenGlobal(int *d_arr, int n) {
    int tid = threadIdx.x;
    for (int phase = 0; phase < n; phase++) {
        int i = tid * 2 + (phase % 2); // even or odd phase
        if (i + 1 < n) {
            if (d_arr[i] > d_arr[i + 1]) {
                int tmp = d_arr[i];
                d_arr[i] = d_arr[i + 1];
                d_arr[i + 1] = tmp;
            }
        }
        __syncthreads();
    }
}

// ------------------------
// CUDA Kernel: Single-block Shared Memory
// ------------------------
__global__ void oddEvenShared(int *d_arr, int n) {
    extern __shared__ int s_arr[];
    int tid = threadIdx.x;

    // Load into shared memory
    if (tid < n) s_arr[tid] = d_arr[tid];
    __syncthreads();

    for (int phase = 0; phase < n; phase++) {
        int i = tid * 2 + (phase % 2);
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

// ------------------------
// CUDA Kernel: Multi-block Global Memory
// ------------------------
__global__ void oddEvenMultiBlock(int *d_arr, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int phase = 0; phase < n; phase++) {
        int i = tid * 2 + (phase % 2);
        if (i + 1 < n) {
            if (d_arr[i] > d_arr[i + 1]) {
                int tmp = d_arr[i];
                d_arr[i] = d_arr[i + 1];
                d_arr[i + 1] = tmp;
            }
        }
        // inter-block sync: terminate kernel and launch next phase
        __syncthreads();
    }
}

// ------------------------
// Utility Functions
// ------------------------
void printArray(int *arr, int n) {
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}

void copyArray(int *dst, int *src, int n) {
    for (int i = 0; i < n; i++) dst[i] = src[i];
}

// ------------------------
// Main
// ------------------------
int main() {
    int *h_arr = (int *)malloc(N * sizeof(int));
    int *h_serial = (int *)malloc(N * sizeof(int));

    srand(time(NULL));
    for (int i = 0; i < N; i++) h_arr[i] = rand() % 1000;

    copyArray(h_serial, h_arr, N);

    // ---------------- Serial ----------------
    clock_t start = clock();
    oddEvenSortSerial(h_serial, N);
    clock_t end = clock();
    printf("Serial sorted first 10 elements: ");
    for (int i = 0; i < 10; i++) printf("%d ", h_serial[i]);
    printf("\n");
    printf("Serial execution time: %f ms\n", 1000.0 * (end - start) / CLOCKS_PER_SEC);

    // ---------------- GPU: Single-block Global Memory ----------------
    int *d_arr;
    cudaMalloc(&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t g_start, g_stop;
    float g_time;
    cudaEventCreate(&g_start); cudaEventCreate(&g_stop);
    cudaEventRecord(g_start);

    oddEvenGlobal<<<1, (N+1)/2>>>(d_arr, N); // single block
    cudaDeviceSynchronize();

    cudaEventRecord(g_stop); cudaEventSynchronize(g_stop);
    cudaEventElapsedTime(&g_time, g_start, g_stop);

    int *h_global = (int *)malloc(N * sizeof(int));
    cudaMemcpy(h_global, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("CUDA Global sorted first 10 elements: ");
    for (int i = 0; i < 10; i++) printf("%d ", h_global[i]);
    printf("\n");
    printf("CUDA Global execution time: %f ms\n", g_time);

    // ---------------- GPU: Single-block Shared Memory ----------------
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(g_start);
    oddEvenShared<<<1, (N+1)/2, N * sizeof(int)>>>(d_arr, N);
    cudaDeviceSynchronize();
    cudaEventRecord(g_stop); cudaEventSynchronize(g_stop);
    cudaEventElapsedTime(&g_time, g_start, g_stop);

    int *h_shared = (int *)malloc(N * sizeof(int));
    cudaMemcpy(h_shared, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("CUDA Shared sorted first 10 elements: ");
    for (int i = 0; i < 10; i++) printf("%d ", h_shared[i]);
    printf("\n");
    printf("CUDA Shared execution time: %f ms\n", g_time);

    // ---------------- GPU: Multi-block Global Memory ----------------
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEventRecord(g_start);
    for (int phase = 0; phase < N; phase++) {
        oddEvenMultiBlock<<<numBlocks, BLOCK_SIZE>>>(d_arr, N);
        cudaDeviceSynchronize(); // inter-block synchronization
    }
    cudaEventRecord(g_stop); cudaEventSynchronize(g_stop);
    cudaEventElapsedTime(&g_time, g_start, g_stop);

    int *h_multiblock = (int *)malloc(N * sizeof(int));
    cudaMemcpy(h_multiblock, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("CUDA Multi-block sorted first 10 elements: ");
    for (int i = 0; i < 10; i++) printf("%d ", h_multiblock[i]);
    printf("\n");
    printf("CUDA Multi-block execution time: %f ms\n", g_time);

    // Cleanup
    cudaFree(d_arr);
    free(h_arr); free(h_serial); free(h_global); free(h_shared); free(h_multiblock);
    return 0;
}
