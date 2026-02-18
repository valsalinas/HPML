#include <stdio.h>
#include <cuda_runtime.h>

#define N 8
#define BLOCK_SIZE 8

__global__ void oddEvenGlobal(int *d_arr, int n, int phase){
    int tid = threadIdx.x;  // Only one block
    int i = tid + (phase % 2); // even or odd phase

    if(i < n-1){
        if(d_arr[i] > d_arr[i+1]){
            int tmp = d_arr[i];
            d_arr[i] = d_arr[i+1];
            d_arr[i+1] = tmp;
        }
    }

    __syncthreads(); // synchronize threads within block
}

int main() {
    int h_arr[N] = {5,2,8,3,1,6,4,7};
    int *d_arr;

    cudaMalloc((void**)&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel for N phases
    for(int phase=0; phase<N; phase++){
        oddEvenGlobal<<<1, BLOCK_SIZE>>>(d_arr, N, phase);
        cudaDeviceSynchronize(); // ensure all threads finish
    }

    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("After CUDA sorting (global memory): ");
    for(int i=0;i<N;i++) printf("%d ", h_arr[i]);
    printf("\n");

    cudaFree(d_arr);
    return 0;
}
