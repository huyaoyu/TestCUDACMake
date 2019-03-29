#include <assert.h>
#include <math.h>
#include <stdio.h>

// Need this file to be included before any Eigen headers.
#include "CUDARoutines/CUDACommon.hpp"

// CUDA runtime
#include <cuda_runtime.h>

// // helper functions and utilities to work with CUDA
// #include <helper_functions.h>
// #include <helper_cuda.h>

#include <eigen3/Eigen/Dense>

#include "CUDARoutines/KernelEigen.hpp"

__device__ void convert_index_2_row_col(int idx, int rows, int cols, int* r, int* c)
{
    *r = idx / cols;
    *c = idx % cols;
}

__device__ void find_window_indices(int r, int c, int rows, int cols, int half, int *rIdx, int *cIdx)
{
    rIdx[0] = r - half >=    0 ? r - half : 0;
    rIdx[1] = r + half <  rows ? r + half : rows - 1;

    cIdx[0] = c - half >=    0 ? c - half : 0;
    cIdx[1] = c + half <  cols ? c + half : cols - 1;
}

__global__
void exponent(int n, const CRReal* input, CRReal* output, int rows, int cols, int cStep, int w = 39)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    index *= cStep;
    int stride = blockDim.x * gridDim.x;
    stride *= cStep;

    // Convert the input memory into Eigen matrix.
    const auto m = Eigen::Map< 
        const Eigen::Matrix< CRReal, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor >, 
        Eigen::Unaligned, 
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> 
    >( input, rows, cols, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cols*cStep, cStep)  );

    int r = 0, c = 0; // Row and column indices.
    int half = ( w - 1 ) / 2;
    int rIdx[2] = {0, 0}; // The actual row indices of the window.
    int cIdx[2] = {0, 0}; // The actual col indices of the window.

    for ( int i = index; i < n; i += stride )
    {
        // Convert the index into row and column indices.
        convert_index_2_row_col( i/cStep, rows, cols, &r, &c );

        // Check if the index is in side the range.
        if ( r >= rows || c >= cols )
        {
            break;
        }

        // Figure out the actual indices of the window.
        find_window_indices( r, c, rows, cols, half, rIdx, cIdx );

        // Take out the window.
        const Eigen::Matrix<CRReal, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            win = m.block( rIdx[0], cIdx[0], rIdx[1] - rIdx[0] + 1, cIdx[1] - cIdx[0] + 1 );

        // Calculate the exponent and sum.
        output[i] = win.array().exp().matrix().sum();
    }
}

int crExponent(const CRReal* input, int rows, int cols, int cStep, CRReal* output)
{
    if ( NULL == output )
    {
        printf("crExponent: output == NULL.");
        return -1;
    }

    CRReal* dmIn  = NULL; // Device memeory pointer.
    CRReal* dmOut = NULL; // Device memeory pointer.

    // Allocate and copy the memory from CPU to GPU.
    const int size = rows*cols*cStep;
    cudaMallocManaged(&dmIn,  size*sizeof( CRReal ));
    cudaMallocManaged(&dmOut, size*sizeof( CRReal ));

    printf("dmIn is of size %d.\n", size*sizeof(CRReal));

    for ( int i = 0; i < size; ++i )
    {
        dmIn[i]  = input[i];
        dmOut[i] = 0;
    }

    // Run kernel on 1M elements on the GPU.
    const int nBlocks = ( size/cStep + N_SP_PER_SMM - 1 ) / N_SP_PER_SMM;
    printf("Start exponent() with nBlocks = %d, threads per block = %d.\n", nBlocks, N_SP_PER_SMM);
    exponent<<<nBlocks, N_SP_PER_SMM>>>(size, dmIn, dmOut, rows, cols, cStep, 7);

    // Wait for the GPU.
    cudaDeviceSynchronize();

    // Copy the value back.
    for ( int i = 0; i < size; ++i )
    {
        output[i] = dmOut[i];
    }

    // Free the memory.
    cudaFree(dmOut); dmOut = NULL;
    cudaFree(dmIn); dmIn = NULL;

    return 0;
}