#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

#define MAX_N 4096
#define MAX_BLOCK 256

// Warmup kernel
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
    extern __shared__ int s_arr[];

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

// ====================== Correctness Check ======================
int checkSorted(int *arr, int *ref, int n) {
    for (int i = 0; i < n; ++i) {
        if (arr[i] != ref[i]) return 0;
    }
    return 1;
}

// ====================== Main ======================
int main() {
    int test_N[] = {32, 64, 128, 1024, 2048, 4096};
    int block_sizes[] = {32, 64, 128, 256};

    printf("N,BLOCK,Serial(ms),Global(ms),Shared(ms),Speedup_Global,Speedup_Shared,GlobalCheck,SharedCheck\n");

    for (int ni = 0; ni < sizeof(test_N)/sizeof(int); ni++) {
        int N = test_N[ni];

        for (int bi = 0; bi < sizeof(block_sizes)/sizeof(int); bi++) {
            int BLOCK_SIZE = block_sizes[bi];
            int shared_valid = (N <= BLOCK_SIZE) ? 1 : 0;

            int *h_arr = (int*)malloc(N*sizeof(int));
            int *h_serial = (int*)malloc(N*sizeof(int));

            srand(time(NULL));
            for (int i = 0; i < N; i++) {
                h_arr[i] = rand() % 100;
                h_serial[i] = h_arr[i];
            }

            // ------------------- Print small arrays for verification -------------------
            if (N <= 32) {
                printf("\nOriginal array (N=%d): ", N);
                for (int i=0;i<N;i++) printf("%d ", h_arr[i]);
                printf("\n");
            }

            // ------------------- Serial -------------------
            clock_t start = clock();
            oddEvenSortSerial(h_serial, N);
            clock_t end = clock();
            double serialTime = 1000.0*(end - start)/CLOCKS_PER_SEC;

            if (N <= 32) {
                printf("Serial sorted: ");
                for (int i=0;i<N;i++) printf("%d ", h_serial[i]);
                printf("\n");
            }

            // ------------------- CUDA Global Multi-block -------------------
            int *d_arr;
            cudaMalloc(&d_arr, N*sizeof(int));
            cudaMemcpy(d_arr, h_arr, N*sizeof(int), cudaMemcpyHostToDevice);

            int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

            cudaEvent_t startEvent, stopEvent;
            float timeGlobal = 0;
            cudaEventCreate(&startEvent); cudaEventCreate(&stopEvent);
            cudaEventRecord(startEvent);

            for (int phase = 0; phase < N; phase++) {
                oddEvenMultiBlock<<<numBlocks, BLOCK_SIZE>>>(d_arr, N, phase);
                cudaDeviceSynchronize();
            }

            cudaEventRecord(stopEvent);
            cudaEventSynchronize(stopEvent);
            cudaEventElapsedTime(&timeGlobal, startEvent, stopEvent);

            int *h_global = (int*)malloc(N*sizeof(int));
            cudaMemcpy(h_global, d_arr, N*sizeof(int), cudaMemcpyDeviceToHost);
            int globalCheck = checkSorted(h_global, h_serial, N);

            // ------------------- CUDA Shared Memory -------------------
            float timeShared = -1.0f;
            int sharedCheck = 0;
            int *h_shared = NULL;

            if (shared_valid) {
                cudaMemcpy(d_arr, h_arr, N*sizeof(int), cudaMemcpyHostToDevice);

                cudaEventRecord(startEvent);
                oddEvenShared<<<1, BLOCK_SIZE, BLOCK_SIZE*sizeof(int)>>>(d_arr, N);
                cudaDeviceSynchronize();
                cudaEventRecord(stopEvent);
                cudaEventSynchronize(stopEvent);
                cudaEventElapsedTime(&timeShared, startEvent, stopEvent);

                h_shared = (int*)malloc(N*sizeof(int));
                cudaMemcpy(h_shared, d_arr, N*sizeof(int), cudaMemcpyDeviceToHost);
                sharedCheck = checkSorted(h_shared, h_serial, N);

                if (N <= 32) {
                    printf("Shared sorted: ");
                    for (int i=0;i<N;i++) printf("%d ", h_shared[i]);
                    printf("\n");
                }
            }

            // ------------------- Output CSV -------------------
            printf("%d,%d,%.3f,%.3f,", N, BLOCK_SIZE, serialTime, timeGlobal);
            if (shared_valid) printf("%.3f,", timeShared);
            else printf("N/A,");
            printf("%.2f,", serialTime/timeGlobal);
            if (shared_valid) printf("%.2f,", serialTime/timeShared);
            else printf("N/A,");
            printf("%s,", globalCheck ? "PASS" : "FAIL");
            if (shared_valid) printf("%s", sharedCheck ? "PASS" : "FAIL");
            else printf("N/A");
            printf("\n");

            // ------------------- Cleanup -------------------
            free(h_arr); free(h_serial); free(h_global);
            if (shared_valid) free(h_shared);
            cudaFree(d_arr);
            cudaEventDestroy(startEvent); cudaEventDestroy(stopEvent);
        }
    }

    return 0;
}
