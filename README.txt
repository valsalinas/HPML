Build Instructions:

To compile the program:

nvcc -o odd_even.cu odd_even

This creates an executable file called odd_even.

Run Instructions:

To run the program:

./odd_even

The program automatically runs the odd_even sort for different:

Array sizes (N)

Thread block sizes (BLOCK)

It prints the results in the following CSV format:

Serial execution time (CPU)

Global memory (single block)

Shared memory (single block)

Global memory (multi block)

Speedups are compared to the serial version.

Correctness has a PASS/FAIL check.

To save the output to a file:

./odd_even > results.csv

Sync Design:

The odd_even sort algorithm runs in multiple phases. Each phase must complete before the next one starts, so synchronization is very important.

Global Memory (Single Block):

In this version, each phase of the algorithm is launched as a separate kernel. Since CUDA kernels complete before the next one starts, this automatically provides synchronization between phases.

Shared Memory (Single Block):

In the shared memory version, I used __syncthreads() inside the kernel. This ensures that all threads finish their comparisons and swaps before moving to the next phase. This prevents race conditions when threads are modifying nearby elements.

Global Memory (Multi Block):

For the multi block version, each phase is also launched as a separate kernel. The kernel launch acts as a global synchronization point across all blocks.

Memory Usage (LMM):

For device memory management:

-cudaMalloc to allocate memory on the GPU.

-cudaMemcpy to copy data between the host (CPU) and device (GPU).

In the global memory versions, sorting is done directly in device global memory.

In the shared memory version, data is first loaded into shared memory inside each block. This reduces global memory accesses and improves performance for smaller input sizes.

The shared memory size is specified dynamically during kernel launch.

This program compares the performance of different CUDA implementations of odd_even sort and measures their speedup relative to the serial CPU version.