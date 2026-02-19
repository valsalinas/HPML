#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define MAX_N 4096
#define MAX_BLOCK 256

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

// ====================== CUDA Kernels ======================
__global__ void oddEvenGlobal(int *d_arr, int n, int phase) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 2 * tid + (phase % 2);
    if (i + 1 < n && d_arr[i] > d_arr[i + 1]) {
        int tmp = d_arr[i];
        d_arr[i] = d_arr[i + 1];
        d_arr[i + 1] = tmp;
    }
}

__global__ void oddEvenShared(int *d_arr, int n) {
    extern __shared__ int s_arr[];
    int tid = threadIdx.x;
    if (tid < n) s_arr[tid] = d_arr[tid];
    __syncthreads();

    for (int phase = 0; phase < n; ++phase) {
        int i = 2 * tid + (phase % 2);
        if (i + 1 < n && s_arr[i] > s_arr[i + 1]) {
            int tmp = s_arr[i];
            s_arr[i] = s_arr[i + 1];
            s_arr[i + 1] = tmp;
        }
        __syncthreads();
    }

    if (tid < n) d_arr[tid] = s_arr[tid];
}

__global__ void oddEvenMultiBlock(int *d_arr, int n, int phase) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 2 * tid + (phase % 2);
    if (i + 1 < n && d_arr[i] > d_arr[i + 1]) {
        int tmp = d_arr[i];
        d_arr[i] = d_arr[i + 1];
        d_arr[i + 1] = tmp;
    }
}

// ====================== Correctness Check ======================
int checkSorted(int *arr, int *ref, int n) {
    for (int i = 0; i < n; ++i)
        if (arr[i] != ref[i]) return 0;
    return 1;
}

// ====================== Main ======================
int main() {
    int test_N[] = {32, 64, 128, 1024, 2048, 4096};
    int block_sizes[] = {32, 64, 128, 256};

    printf("N,BLOCK,Serial(ms),GlobalSingle(ms),Shared(ms),GlobalMulti(ms),");
    printf("Speedup_GlobalSingle,Speedup_Shared,Speedup_GlobalMulti,");
    printf("GlobalSingleCheck,SharedCheck,GlobalMultiCheck\n");

    for (int ni = 0; ni < sizeof(test_N)/sizeof(int); ni++) {
        int N = test_N[ni];

        for (int bi = 0; bi < sizeof(block_sizes)/sizeof(int); bi++) {
            int BLOCK_SIZE = block_sizes[bi];
            int single_block_valid = (N <= BLOCK_SIZE) ? 1 : 0;
            int shared_valid = (N <= BLOCK_SIZE) ? 1 : 0;

            // Allocate host arrays
            int *h_arr = (int*)malloc(N*sizeof(int));
            int *h_serial = (int*)malloc(N*sizeof(int));

            // Initialize array
            srand(time(NULL));
            for (int i = 0; i < N; i++) {
                h_arr[i] = rand() % 100;
                h_serial[i] = h_arr[i];
            }

            // ------------------- Serial -------------------
            clock_t start = clock();
            oddEvenSortSerial(h_serial, N);
            clock_t end = clock();
            double serialTime = 1000.0*(end - start)/CLOCKS_PER_SEC;

            // ------------------- Allocate device -------------------
            int *d_arr;
            cudaMalloc(&d_arr, N*sizeof(int));
            cudaEvent_t startEvent, stopEvent;
            cudaEventCreate(&startEvent); cudaEventCreate(&stopEvent);

            // ------------------- Global Memory Single-block -------------------
            float timeGlobalSingle = -1.0f;
            int globalSingleCheck = 0;
            if (single_block_valid) {
                cudaMemcpy(d_arr, h_arr, N*sizeof(int), cudaMemcpyHostToDevice);
                cudaEventRecord(startEvent);
                for (int phase = 0; phase < N; phase++)
                    oddEvenGlobal<<<1, BLOCK_SIZE>>>(d_arr, N, phase);
                cudaDeviceSynchronize();
                cudaEventRecord(stopEvent);
                cudaEventSynchronize(stopEvent);
                cudaEventElapsedTime(&timeGlobalSingle, startEvent, stopEvent);

                int *h_globalSingle = (int*)malloc(N*sizeof(int));
                cudaMemcpy(h_globalSingle, d_arr, N*sizeof(int), cudaMemcpyDeviceToHost);
                globalSingleCheck = checkSorted(h_globalSingle, h_serial, N);
                free(h_globalSingle);
            }

            // ------------------- Shared Memory Single-block -------------------
            float timeShared = -1.0f;
            int sharedCheck = 0;
            if (shared_valid) {
                cudaMemcpy(d_arr, h_arr, N*sizeof(int), cudaMemcpyHostToDevice);
                cudaEventRecord(startEvent);
                oddEvenShared<<<1, BLOCK_SIZE, BLOCK_SIZE*sizeof(int)>>>(d_arr, N);
                cudaDeviceSynchronize();
                cudaEventRecord(stopEvent);
                cudaEventSynchronize(stopEvent);
                cudaEventElapsedTime(&timeShared, startEvent, stopEvent);

                int *h_shared = (int*)malloc(N*sizeof(int));
                cudaMemcpy(h_shared, d_arr, N*sizeof(int), cudaMemcpyDeviceToHost);
                sharedCheck = checkSorted(h_shared, h_serial, N);
                free(h_shared);
            }

            // ------------------- Global Memory Multi-block -------------------
            int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            float timeGlobalMulti = -1.0f;
            int globalMultiCheck = 0;
            cudaMemcpy(d_arr, h_arr, N*sizeof(int), cudaMemcpyHostToDevice);
            cudaEventRecord(startEvent);
            for (int phase = 0; phase < N; phase++)
                oddEvenMultiBlock<<<numBlocks, BLOCK_SIZE>>>(d_arr, N, phase);
            cudaDeviceSynchronize();
            cudaEventRecord(stopEvent);
            cudaEventSynchronize(stopEvent);
            cudaEventElapsedTime(&timeGlobalMulti, startEvent, stopEvent);

            int *h_globalMulti = (int*)malloc(N*sizeof(int));
            cudaMemcpy(h_globalMulti, d_arr, N*sizeof(int), cudaMemcpyDeviceToHost);
            globalMultiCheck = checkSorted(h_globalMulti, h_serial, N);
            free(h_globalMulti);

            // ------------------- Print CSV -------------------
            printf("%d,%d,%.3f,", N, BLOCK_SIZE, serialTime);
            printf(single_block_valid ? "%.3f," : "N/A,", timeGlobalSingle);
            printf(shared_valid ? "%.3f," : "N/A,", timeShared);
            printf("%.3f,", timeGlobalMulti);
            printf(single_block_valid ? "%.2f," : "N/A,", serialTime/timeGlobalSingle);
            printf(shared_valid ? "%.2f," : "N/A,", serialTime/timeShared);
            printf("%.2f,", serialTime/timeGlobalMulti);
            printf(single_block_valid ? "%s," : "N/A,", globalSingleCheck ? "PASS" : "FAIL");
            printf(shared_valid ? "%s," : "N/A,", sharedCheck ? "PASS" : "FAIL");
            printf("%s\n", globalMultiCheck ? "PASS" : "FAIL");

            // ------------------- Cleanup -------------------
            free(h_arr); free(h_serial);
            cudaFree(d_arr);
            cudaEventDestroy(startEvent); cudaEventDestroy(stopEvent);
        }
    }

    return 0;
}
