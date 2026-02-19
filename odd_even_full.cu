#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define N 1024  // array size

// --- Serial Odd-Even Sort ---
void serialOddEvenSort(int *arr, int n) {
    for(int phase = 0; phase < n; phase++) {
        int start = phase % 2;
        for(int i = start; i < n-1; i+=2) {
            if(arr[i] > arr[i+1]) {
                int tmp = arr[i];
                arr[i] = arr[i+1];
                arr[i+1] = tmp;
            }
        }
    }
}

// --- CUDA: Global Memory Kernel ---
__global__ void oddEvenGlobal(int *arr, int n, int phase) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int start = phase % 2;
    i += start;
    if(i < n-1) {
        if(arr[i] > arr[i+1]) {
            int tmp = arr[i];
            arr[i] = arr[i+1];
            arr[i+1] = tmp;
        }
    }
}

// --- CUDA: Shared Memory Kernel (Single Block) ---
__global__ void oddEvenSharedSingleBlock(int *arr, int n, int phase) {
    __shared__ int tile[1024]; // max array size for single block
    int tid = threadIdx.x;
    if(tid < n) tile[tid] = arr[tid];
    __syncthreads();

    int start = phase % 2;
    for(int i = start + tid; i < n-1; i += blockDim.x * 2) {
        if(tile[i] > tile[i+1]) {
            int tmp = tile[i];
            tile[i] = tile[i+1];
            tile[i+1] = tmp;
        }
    }
    __syncthreads();

    if(tid < n) arr[tid] = tile[tid];
}

// --- CUDA: Shared Memory Kernel (Multi-Block Tiling) ---
__global__ void oddEvenSharedMultiBlock(int *arr, int n, int phase) {
    __shared__ int tile[256]; // tile size = BLOCK_SIZE
    int blockStart = blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int globalIdx = blockStart + tid;

    if(globalIdx < n) tile[tid] = arr[globalIdx];
    __syncthreads();

    int start = phase % 2;
    for(int i = start + tid; i < blockDim.x - 1 && (blockStart + i + 1) < n; i += blockDim.x * 2) {
        if(tile[i] > tile[i+1]) {
            int tmp = tile[i];
            tile[i] = tile[i+1];
            tile[i+1] = tmp;
        }
    }
    __syncthreads();

    if(globalIdx < n) arr[globalIdx] = tile[tid];
}

// --- Helper: Print first 16 elements ---
void print16(int *arr) {
    for(int i=0;i<16;i++) printf("%d ", arr[i]);
    printf("\n");
}

// --- Main ---
int main() {
    int h_arr[N], h_arr_serial[N];
    int *d_arr;
    srand(time(NULL));
    for(int i = 0; i < N; i++) {
        h_arr[i] = rand() % 100;
        h_arr_serial[i] = h_arr[i];
    }

    printf("Original array first 16 elements:\n");
    print16(h_arr);

    // --- Serial Sort ---
    clock_t start = clock();
    serialOddEvenSort(h_arr_serial, N);
    clock_t end = clock();
    printf("Serial sorted first 16 elements:\n");
    print16(h_arr_serial);
    printf("Serial execution time: %f ms\n\n", 1000.0 * (end-start)/CLOCKS_PER_SEC);

    // Allocate GPU memory
    cudaMalloc((void**)&d_arr, N*sizeof(int));

    int blockSizes[] = {64, 128, 256};
    for(int b = 0; b < 3; b++) {
        int BLOCK_SIZE = blockSizes[b];
        int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        printf("=== Testing BLOCK_SIZE = %d ===\n", BLOCK_SIZE);

        // Copy original array to device
        cudaMemcpy(d_arr, h_arr, N*sizeof(int), cudaMemcpyHostToDevice);

        // --- CUDA Global Memory ---
        start = clock();
        for(int phase=0; phase<N; phase++) {
            oddEvenGlobal<<<numBlocks, BLOCK_SIZE>>>(d_arr, N, phase);
            cudaDeviceSynchronize();
        }
        end = clock();
        cudaMemcpy(h_arr, d_arr, N*sizeof(int), cudaMemcpyDeviceToHost);
        printf("CUDA Global sorted first 16 elements:\n");
        print16(h_arr);
        printf("CUDA Global execution time: %f ms\n\n", 1000.0 * (end-start)/CLOCKS_PER_SEC);

        // --- CUDA Shared Memory Single Block (only works if N <= BLOCK_SIZE) ---
        if(N <= BLOCK_SIZE) {
            cudaMemcpy(d_arr, h_arr, N*sizeof(int), cudaMemcpyHostToDevice);
            start = clock();
            for(int phase=0; phase<N; phase++) {
                oddEvenSharedSingleBlock<<<1, BLOCK_SIZE>>>(d_arr, N, phase);
                cudaDeviceSynchronize();
            }
            end = clock();
            cudaMemcpy(h_arr, d_arr, N*sizeof(int), cudaMemcpyDeviceToHost);
            printf("CUDA Shared (single-block) sorted first 16 elements:\n");
            print16(h_arr);
            printf("CUDA Shared execution time: %f ms\n\n", 1000.0 * (end-start)/CLOCKS_PER_SEC);
        }

        // --- CUDA Shared Memory Multi-Block ---
        cudaMemcpy(d_arr, h_arr, N*sizeof(int), cudaMemcpyHostToDevice);
        start = clock();
        for(int phase=0; phase<N; phase++) {
            oddEvenSharedMultiBlock<<<numBlocks, BLOCK_SIZE>>>(d_arr, N, phase);
            cudaDeviceSynchronize();
        }
        end = clock();
        cudaMemcpy(h_arr, d_arr, N*sizeof(int), cudaMemcpyDeviceToHost);
        printf("CUDA Shared Multi-block sorted first 16 elements:\n");
        print16(h_arr);
        printf("CUDA Shared Multi-block execution time: %f ms\n\n", 1000.0 * (end-start)/CLOCKS_PER_SEC);
    }

    cudaFree(d_arr);
    free(h_arr);
    free(h_arr_serial);

    return 0;
}
