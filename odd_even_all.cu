#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define MAX_BLOCK_SIZE 256  // maximum threads per block

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

// ====================== CUDA Shared Memory Kernel (single-block) ======================
__global__ void oddEvenShared(int *d_arr, int n) {
    __shared__ int s_arr[MAX_BLOCK_SIZE];
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

// ====================== Main ======================
int main() {
    int sizes[] = {1024, 2048, 4096};
    int blockSizes[] = {64, 128, 256};

    int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    int numBlockSizes = sizeof(blockSizes) / sizeof(blockSizes[0]);

    printf("N\tBLOCK\tSerial(ms)\tGlobal(ms)\tShared(ms)\tSpeedup_Global\tSpeedup_Shared\n");

    for (int s = 0; s < numSizes; s++) {
        int N = sizes[s];

        // Allocate host memory
        int *h_arr = (int*)malloc(N * sizeof(int));
        int *h_serial = (int*)malloc(N * sizeof(int));

        srand(time(NULL));
        for (int i = 0; i < N; ++i) {
            h_arr[i] = rand() % 1000;
            h_serial[i] = h_arr[i];
        }

        // -------- Serial --------
        clock_t start = clock();
        oddEvenSortSerial(h_serial, N);
        clock_t end = clock();
        double serialTime = 1000.0 * (end - start) / CLOCKS_PER_SEC;

        for (int b = 0; b < numBlockSizes; b++) {
            int BLOCK_SIZE = blockSizes[b];
            int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // Allocate device memory
            int *d_arr;
            cudaMalloc(&d_arr, N * sizeof(int));
            cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

            // -------- Global Multi-block --------
            cudaEvent_t startEvent, stopEvent;
            cudaEventCreate(&startEvent); cudaEventCreate(&stopEvent);
            cudaEventRecord(startEvent);

            for (int phase = 0; phase < N; ++phase) {
                oddEvenMultiBlock<<<numBlocks, BLOCK_SIZE>>>(d_arr, N, phase);
                cudaDeviceSynchronize();
            }

            cudaEventRecord(stopEvent);
            cudaEventSynchronize(stopEvent);
            float timeGlobal;
            cudaEventElapsedTime(&timeGlobal, startEvent, stopEvent);

            int *h_global = (int*)malloc(N * sizeof(int));
            cudaMemcpy(h_global, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

            // Correctness check
            int passGlobal = 1;
            for (int i = 0; i < N; i++) {
                if (h_serial[i] != h_global[i]) passGlobal = 0;
            }

            // -------- Shared single-block --------
            cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

            cudaEventRecord(startEvent);
            oddEvenShared<<<1, BLOCK_SIZE>>>(d_arr, N);
            cudaDeviceSynchronize();
            cudaEventRecord(stopEvent);
            cudaEventSynchronize(stopEvent);
            float timeShared;
            cudaEventElapsedTime(&timeShared, startEvent, stopEvent);

            int *h_shared = (int*)malloc(N * sizeof(int));
            cudaMemcpy(h_shared, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

            int passShared = 1;
            for (int i = 0; i < N; i++) {
                if (h_serial[i] != h_shared[i]) passShared = 0;
            }

            // -------- Print results --------
            printf("%d\t%d\t%.3f\t\t%.3f\t\t%.3f\t\t%.2f\t\t%.2f\t%s\t%s\n",
                N, BLOCK_SIZE,
                serialTime,
                timeGlobal,
                timeShared,
                serialTime / timeGlobal,
                serialTime / timeShared,
                passGlobal ? "" : "FAIL",
                passShared ? "" : "FAIL");

            cudaFree(d_arr);
            free(h_global); free(h_shared);
            cudaEventDestroy(startEvent); cudaEventDestroy(stopEvent);
        }

        free(h_arr); free(h_serial);
    }

    return 0;
}
