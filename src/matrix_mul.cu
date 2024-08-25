// matrix_mul.cu
#include <cuda_runtime.h>
#include <iostream>

// Kernel function to perform matrix multiplication
#define TILE_SIZE 32

__global__ void matrixMulShared(float* C, const float* A, const float* B, int N) {
    // Shared memory for tiles of A and B
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    // Calculate row and column for each thread
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float result = 0.0f;

    // Loop over tiles
    for (int tile = 0; tile < N / TILE_SIZE; ++tile) {
        // Load tiles into shared memory
        sharedA[threadIdx.y][threadIdx.x] = A[row * N + (tile * TILE_SIZE + threadIdx.x)];
        sharedB[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];

        __syncthreads();  // Synchronize threads in the block

        // Compute partial result
        for (int k = 0; k < TILE_SIZE; ++k) {
            result += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
        }

        __syncthreads();  // Synchronize threads before loading new tiles
    }

    // Write result to global memory
    if (row < N && col < N) {
        C[row * N + col] = result;
    }
}

void matrixMul(float* C, const float* A, const float* B, int N) {
    // Device pointers
    float *d_A, *d_B, *d_C;

    size_t bytes = N * N * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data to device
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    // Set grid and block dimensions
    int blockSize = 32;  // set threads per block which will be blockSizexblockSize
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);

    // Launch the kernel
    matrixMulShared<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, N);

    // Copy result back to host
    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
