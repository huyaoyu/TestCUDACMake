#include <stdio.h>

#include "CUDARoutines/CUDACommon.hpp"
#include "CUDARoutines/Add.hpp"

// function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i+=stride)
        y[i] = x[i] + y[i];
}

int crAdd(void)
{
    int N = 1 << 20; // 1M elements.

    float *x, *y;

    // Allocate Unified Memory - accessible from both CPU and GPU.
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host.
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU.
    const int nBlocks = ( N + N_SP_PER_SMM - 1 ) / N_SP_PER_SMM;
    add<<<nBlocks, N_SP_PER_SMM>>>(N, x, y);

    // Wait for the GPU.
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    printf("Max error: %f\n", maxError);

    // Free memory
    cudaFree(x); x = NULL;
    cudaFree(y); y = NULL;

    return 0;
}
