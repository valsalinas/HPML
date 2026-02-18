#include <stdio.h>
#include <cuda_runtime.h>

#define N 8  // Small test array
#define BLOCK_SIZE 4 // Number of threads in the block

__global__ void oddEvenKernelGlobal(int *arr, int n) {
    int tid = threadIdx.x;

    for(int phase=0; phase<n; phase++){
        int i = tid*2 + (phase % 2); // Even or odd phase
        if(i+1 < n){
            if(arr[i] > arr[i+1]){
                int tmp = arr[i];
                arr[i] = arr[i+1];
                arr[i+1] = tmp;
            }
        }
        __syncthreads(); // Wait for all threads in block
    }
}

int main() {
    int h_arr[N] = {5,3,8,1,4,7,2,6};
    int *d_arr;

    cudaMalloc(&d_arr, N*sizeof(int));
    cudaMemcpy(d_arr, h_arr, N*sizeof(int), cudaMemcpyHostToDevice);

    oddEvenKernelGlobal<<<1, BLOCK_SIZE>>>(d_arr, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_arr, d_arr, N*sizeof(int), cudaMemcpyDeviceToHost);

    printf("After CUDA sorting:\n");
    for(int i=0;i<N;i++) printf("%d ", h_arr[i]);
    printf("\n");

    cudaFree(d_arr);
    return 0;
}
