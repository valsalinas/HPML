#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define N 1024  // Array size
#define WARP_SIZE 32

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

// --- Warmup Kernel ---
__global__ void warmupKernel(int *arr) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N) arr[idx] += 0;  // do nothing, just touch memory
}

// --- CUDA Global Memory Kernel ---
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

// --- Shared Memory Single Block Kernel (with padding to reduce bank conflicts) ---
__global__ void oddEvenSharedSingleBlock(int *arr, int n, int phase) {
    __shared__ int tile[N + WARP_SIZE];  // padding for bank conflicts
    int tid = threadIdx.x;

    if(tid < n) tile[tid + (tid/WARP_SIZE)] = arr[tid]; // apply padding
    __syncthreads();

    int start = phase % 2;
    for(int i = start + tid; i < n-1; i += blockDim.x*2) {
        int idx_i = i + (i/WARP_SIZE);     // padded index
        int idx_i1 = i+1 + ((i+1)/WARP_SIZE);
        if(tile[idx_i] > tile[idx_i1]) {
            int tmp = tile[idx_i];
            tile[idx_i] = tile[idx_i1];
            tile[idx_i1] = tmp;
        }
    }
    __syncthreads();

    if(tid < n) arr[tid] = tile[tid + (tid/WARP_SIZE)];
}

// --- Shared Memory Multi-Block Kernel (with boundary handling and padding) ---
__global__ void oddEvenSharedMultiBlock(int *arr, int n, int phase) {
    __shared__ int tile[256 + WARP_SIZE];  // block tile + padding
    int blockStart = blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int globalIdx = blockStart + tid;

    // load to shared memory with padding
    if(globalIdx < n) tile[tid + (tid/WARP_SIZE)] = arr[globalIdx];
    __syncthreads();

    int start = phase % 2;
    for(int i = start + tid; i < blockDim.x-1 && (blockStart + i + 1) < n; i += blockDim.x*2) {
        int idx_i = i + (i/WARP_SIZE);
        int idx_i1 = i+1 + ((i+1)/WARP_SIZE);
        if(tile[idx_i] > tile[idx_i1]) {
            int tmp = tile[idx_i];
            tile[idx_i] = tile[idx_i1];
            tile[idx_i1] = tmp;
        }
    }
    __syncthreads();

    // store back to global memory
    if(globalIdx < n) arr[globalIdx] = tile[tid + (tid/WARP_SIZE)];
}

// --- Helper to print first 16 elements ---
void print16(int *arr) {
    for(int i = 0; i < 16; i++) printf("%d ", arr[i]);
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

    // --- Serial ---
    clock_t start = clock();
    serialOddEvenSort(h_arr_serial, N);
    clock_t end = clock();
    printf("Serial sorted first 16 elements:\n");
    print16(h_arr_serial);
    printf("Serial execution time: %f ms\n\n", 1000.0*(end-start)/CLOCKS_PER_SEC);

    // --- GPU ---
    cudaMalloc((void**)&d_arr, N*sizeof(int));

    // Warmup
    warmupKernel<<<(N+127)/128, 128>>>(d_arr);
    cudaDeviceSynchronize();

    int blockSizes[] = {64, 128, 256};
    for(int b = 0; b < 3; b++) {
        int BLOCK_SIZE = blockSizes[b];
        int numBlocks = (N + BLOCK_SIZE - 1)/BLOCK_SIZE;
        printf("=== Testing BLOCK_SIZE = %d ===\n", BLOCK_SIZE);

        // --- Global Memory ---
        cudaMemcpy(d_arr, h_arr, N*sizeof(int), cudaMemcpyHostToDevice);
        start = clock();
        for(int phase = 0; phase < N; phase++) {
            oddEvenGlobal<<<numBlocks, BLOCK_SIZE>>>(d_arr, N, phase);
            cudaDeviceSynchronize();
        }
        end = clock();
        cudaMemcpy(h_arr, d_arr, N*sizeof(int), cudaMemcpyDeviceToHost);
        printf("CUDA Global sorted first 16 elements:\n");
        print16(h_arr);
        printf("CUDA Global execution time: %f ms\n\n", 1000.0*(end-start)/CLOCKS_PER_SEC);

        // --- Shared Memory Single Block ---
        if(N <= BLOCK_SIZE) {
            cudaMemcpy(d_arr, h_arr, N*sizeof(int), cudaMemcpyHostToDevice);
            start = clock();
            for(int phase = 0; phase < N; phase++) {
                oddEvenSharedSingleBlock<<<1, BLOCK_SIZE>>>(d_arr, N, phase);
                cudaDeviceSynchronize();
            }
            end = clock();
            cudaMemcpy(h_arr, d_arr, N*sizeof(int), cudaMemcpyDeviceToHost);
            printf("CUDA Shared (single-block) sorted first 16 elements:\n");
            print16(h_arr);
            printf("CUDA Shared execution time: %f ms\n\n", 1000.0*(end-start)/CLOCKS_PER_SEC);
        }

        // --- Shared Memory Multi-Block ---
        cudaMemcpy(d_arr, h_arr, N*sizeof(int), cudaMemcpyHostToDevice);
        start = clock();
        for(int phase = 0; phase < N; phase++) {
            oddEvenSharedMultiBlock<<<numBlocks, BLOCK_SIZE>>>(d_arr, N, phase);
            cudaDeviceSynchronize();
        }
        end = clock();
        cudaMemcpy(h_arr, d_arr, N*sizeof(int), cudaMemcpyDeviceToHost);
        printf("CUDA Shared Multi-block sorted first 16 elements:\n");
        print16(h_arr);
        printf("CUDA Shared Multi-block execution time: %f ms\n\n", 1000.0*(end-start)/CLOCKS_PER_SEC);
    }

    cudaFree(d_arr);
    return 0;
}
