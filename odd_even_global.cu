#include <stdio.h>
#include <cuda_runtime.h>
#define N 9
#define BLOCK_SIZE 8

__global__ void oddEvenGlobal(int *arr, int n, int phase) {
    int tid = threadIdx.x;
    int i = tid + (phase % 2);  // switch between even/odd phase
    if (i < n-1) {
        if (arr[i] > arr[i+1]) {
            int tmp = arr[i]; arr[i] = arr[i+1]; arr[i+1] = tmp;
        }
    }
    __syncthreads();
}

int main() {
    int h_arr[N] = {5,2,8,3,1,6,4,7,0};
    int *d_arr;

    cudaMalloc((void**)&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    for (int phase = 0; phase < N; phase++) {
        oddEvenGlobal<<<1, BLOCK_SIZE>>>(d_arr, N, phase);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("After CUDA sorting (global memory): ");
    for (int i = 0; i < N; i++) printf("%d ", h_arr[i]);
    printf("\n");

    cudaFree(d_arr);
    return 0;
}
