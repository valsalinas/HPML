#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 32          // Start small for debugging
#define BLOCK_SIZE 16 // Threads per block (must be ≤ N)

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

// ====================== CUDA Shared Memory Kernel (Debug) ======================
__global__ void oddEvenSharedDebug(int *arr, int n) {
    __shared__ int s_arr[BLOCK_SIZE * 2]; // Make large enough for your block slice
    int tid = threadIdx.x;
    int globalStart = blockIdx.x * blockDim.x * 2; // Each block handles 2*BLOCK_SIZE elements

    // Copy block slice from global to shared memory
    if (tid * 2 < n - globalStart) s_arr[tid * 2] = arr[globalStart + tid * 2];
    if (tid * 2 + 1 < n - globalStart) s_arr[tid * 2 + 1] = arr[globalStart + tid * 2 + 1];
    __syncthreads();

    // Odd-even sort in shared memory (block-level)
    for (int phase = 0; phase < BLOCK_SIZE * 2; phase++) {
        int i = 2 * tid + (phase % 2);
        if (i + 1 < BLOCK_SIZE * 2 && i + globalStart + 1 < n) {
            if (s_arr[i] > s_arr[i + 1]) {
                int tmp = s_arr[i];
                s_arr[i] = s_arr[i + 1];
                s_arr[i + 1] = tmp;
            }
        }
        __syncthreads(); // Barrier to avoid race condition

        // Debug: print the block after each phase
        if (tid == 0) {
            printf("Block %d after phase %d: ", blockIdx.x, phase);
            for (int j = 0; j < BLOCK_SIZE * 2 && globalStart + j < n; j++) {
                printf("%d ", s_arr[j]);
            }
            printf("\n");
        }
        __syncthreads();
    }

    // Copy back to global memory
    if (tid * 2 < n - globalStart) arr[globalStart + tid * 2] = s_arr[tid * 2];
    if (tid * 2 + 1 < n - globalStart) arr[globalStart + tid * 2 + 1] = s_arr[tid * 2 + 1];
}

// ====================== Main ======================
int main() {
    int *h_arr = (int*)malloc(N * sizeof(int));
    int *h_serial = (int*)malloc(N * sizeof(int));

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_arr[i] = rand() % 100; // small numbers for easy reading
        h_serial[i] = h_arr[i];
    }

    printf("Warmup kernel...\n");
    warmup<<<1,1>>>();
    cudaDeviceSynchronize();

    // ------------------- Serial -------------------
    oddEvenSortSerial(h_serial, N);
    printf("Serial sorted array:\n");
    for (int i = 0; i < N; i++) printf("%d ", h_serial[i]);
    printf("\n\n");

    // ------------------- CUDA Shared Memory Debug -------------------
    int *d_arr;
    cudaMalloc(&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    oddEvenSharedDebug<<<1, BLOCK_SIZE>>>(d_arr, N);
    cudaDeviceSynchronize();

    int *h_shared = (int*)malloc(N * sizeof(int));
    cudaMemcpy(h_shared, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("CUDA Shared sorted array:\n");
    for (int i = 0; i < N; i++) printf("%d ", h_shared[i]);
    printf("\n");

    // Cleanup
    cudaFree(d_arr);
    free(h_arr);
    free(h_serial);
    free(h_shared);

    return 0;
}
