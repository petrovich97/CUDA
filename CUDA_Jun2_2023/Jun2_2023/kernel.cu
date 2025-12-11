
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "book.h"

#include <stdio.h>

#define BLOCK_DIM 4
#define GRID_DIM 4
#define N 32



__global__ void calculate (int *nizA, int* nizB, int n)
{
    __shared__ int sh[BLOCK_DIM + 2];

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    while (tid < N - 2)
    {
        sh[threadIdx.x] = nizA[tid];

        if (threadIdx.x == blockDim.x - 1)
        {
            sh[threadIdx.x + 1] = nizA[tid + 1];
            sh[threadIdx.x + 2] = nizA[tid + 2];
        }
        __syncthreads();

        nizB[tid] = sh[threadIdx.x] * sh[threadIdx.x + 1] * sh[threadIdx.x + 2] / (sh[threadIdx.x] + sh[threadIdx.x + 1] + sh[threadIdx.x + 2]);
        tid += gridDim.x * blockDim.x;
    }

}


int main()
{
    int nizA[N], nizB[N - 2];
    int* devA, * devB;

    cudaMalloc((void**)&devA, N * sizeof(int));
    cudaMalloc((void**)&devB, (N - 2) * sizeof(int));

    for (int i = 0; i < N; i++)
        nizA[i] = i + 1;

    cudaMemcpy(devA, nizA, N * sizeof(int), cudaMemcpyHostToDevice);
    
    calculate << <GRID_DIM, BLOCK_DIM >> > (devA, devB, N);

    cudaMemcpy(nizB, devB, (N - 2) * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\n");

    for (int i = 0; i < N - 2; i++)
        printf("%d, ", nizB[i]);
        
}
