/*
Odd-Even Transposition Sort in CUDA
Includes:
1. Serial
2. CUDA Global Memory (Single Block)
3. CUDA Shared Memory (Single Block)
4. CUDA Multi-Block Version
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

#define N 1024  // Array size
#define BLOCK_SIZE 128

// ---------------- Serial Odd-Even Sort ----------------
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

// ---------------- CUDA Global Memory Kernel ----------------
__global__ void oddEvenGlobal(int *d_arr, int n, int phase) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid * 2 + (phase % 2); // even or odd
    if (i + 1 < n) {
        if (d_arr[i] > d_arr[i + 1]) {
            int tmp = d_arr[i];
            d_arr[i] = d_arr[i + 1];
            d_arr[i + 1] = tmp;
        }
    }
}

// ---------------- CUDA Shared Memory Kernel ----------------
__global__ void oddEvenShared(int *d_arr, int n) {
    extern __shared__ int s_arr[];
    int tid = threadIdx.x;

    if (tid < n)
        s_arr[tid] = d_arr[tid];
    __syncthreads();

    for (int phase = 0; phase < n; phase++) {
        int i = tid + (phase % 2);
        if (i + 1 < n && tid < n/2) {
            if (s_arr[i] > s_arr[i + 1]) {
                int tmp = s_arr[i];
                s_arr[i] = s_arr[i + 1];
                s_arr[i + 1] = tmp;
            }
        }
        __syncthreads();
    }

    if (tid < n)
        d_arr[tid] = s_arr[tid];
}

// ---------------- CUDA Multi-Block Kernel ----------------
__global__ void oddEvenMultiBlock(int *d_arr, int n, int phase) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid * 2 + (phase % 2);
    if (i + 1 < n) {
        if (d_arr[i] > d_arr[i + 1]) {
            int tmp = d_arr[i];
            d_arr[i] = d_arr[i + 1];
            d_arr[i + 1] = tmp;
        }
    }
}

// ---------------- Utility: Print first few elements ----------------
void printArray(int *arr, int n, int maxPrint) {
    for (int i = 0; i < n && i < maxPrint; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

// ---------------- Main ----------------
int main() {
    int *h_arr = (int*)malloc(N * sizeof(int));
    int *h_serial = (int*)malloc(N * sizeof(int));
    int *h_cuda = (int*)malloc(N * sizeof(int));
    int *d_arr;

    // Initialize random array
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_arr[i] = rand() % 1000;
        h_serial[i] = h_arr[i];
        h_cuda[i] = h_arr[i];
    }

    // ---------------- Serial ----------------
    clock_t start = clock();
    oddEvenSortSerial(h_serial, N);
    clock_t end = clock();
    printf("Serial sorted first 16 elements:\n");
    printArray(h_serial, N, 16);
    double serialTime = (double)(end - start) / CLOCKS_PER_SEC * 1000;
    printf("Serial execution time: %f ms\n\n", serialTime);

    // ---------------- CUDA Setup ----------------
    cudaMalloc(&d_arr, N * sizeof(int));

    // ---------------- Global Memory Single Block ----------------
    cudaMemcpy(d_arr, h_cuda, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t gStart, gStop;
    cudaEventCreate(&gStart); cudaEventCreate(&gStop);
    cudaEventRecord(gStart);

    int numThreads = (N + 1)/2;
    int numBlocks = (numThreads + BLOCK_SIZE -1)/BLOCK_SIZE;
    for (int phase = 0; phase < N; phase++)
        oddEvenGlobal<<<numBlocks, BLOCK_SIZE>>>(d_arr, N, phase);
    cudaDeviceSynchronize();

    cudaEventRecord(gStop);
    cudaEventSynchronize(gStop);
    float gTime;
    cudaEventElapsedTime(&gTime, gStart, gStop);

    cudaMemcpy(h_cuda, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("CUDA Global sorted first 16 elements:\n");
    printArray(h_cuda, N, 16);
    printf("CUDA Global execution time: %f ms\n\n", gTime);

    // ---------------- Shared Memory Single Block ----------------
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t sStart, sStop;
    cudaEventCreate(&sStart); cudaEventCreate(&sStop);
    cudaEventRecord(sStart);

    oddEvenShared<<<1, N, N * sizeof(int)>>>(d_arr, N);
    cudaDeviceSynchronize();

    cudaEventRecord(sStop);
    cudaEventSynchronize(sStop);
    float sTime;
    cudaEventElapsedTime(&sTime, sStart, sStop);

    cudaMemcpy(h_cuda, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("CUDA Shared sorted first 16 elements:\n");
    printArray(h_cuda, N, 16);
    printf("CUDA Shared execution time: %f ms\n\n", sTime);

    // ---------------- Multi-Block Version ----------------
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t mStart, mStop;
    cudaEventCreate(&mStart); cudaEventCreate(&mStop);
    cudaEventRecord(mStart);

    numThreads = (N + 1)/2;
    numBlocks = (numThreads + BLOCK_SIZE -1)/BLOCK_SIZE;
    for (int phase = 0; phase < N; phase++) {
        oddEvenMultiBlock<<<numBlocks, BLOCK_SIZE>>>(d_arr, N, phase);
        cudaDeviceSynchronize(); // inter-block sync
    }

    cudaEventRecord(mStop);
    cudaEventSynchronize(mStop);
    float mTime;
    cudaEventElapsedTime(&mTime, mStart, mStop);

    cudaMemcpy(h_cuda, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("CUDA Multi-block sorted first 16 elements:\n");
    printArray(h_cuda, N, 16);
    printf("CUDA Multi-block execution time: %f ms\n\n", mTime);

    // ---------------- Cleanup ----------------
    cudaFree(d_arr);
    free(h_arr); free(h_serial); free(h_cuda);

    return 0;
}
