
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 4
#define N 8


__global__ void kernelMin(int *devMat, int*rez)
{
    __shared__ int partMin[BLOCK_DIM];
    int indexX, indexY;

    if (blockIdx.x == blockIdx.y && threadIdx.x == threadIdx.y)
    {
        indexX = blockIdx.x * blockDim.x + threadIdx.x;
        indexY = blockIdx.y * blockDim.y + threadIdx.y;

        partMin[threadIdx.x] = devMat[indexY*N+indexX];

        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == blockIdx.y)
        {
            // printf("Glavna dijagonala: ");
            for (int i = 0; i < BLOCK_DIM; i++)
                printf("%d ", partMin[i]);
        }

        for (int stride = blockDim.x / 2; stride > 0; stride = stride / 2)
        {
            if (threadIdx.x < stride)
            {
                if (partMin[threadIdx.x + stride] < partMin[threadIdx.x])
                    partMin[threadIdx.x] = partMin[threadIdx.x + stride];
                __syncthreads();
            }
        }
        if (threadIdx.x == 0)
            rez[blockIdx.x] = partMin[0];
    }

    
   


}

__global__ void redKernel(int* n1, int* n2)
{
    __shared__ int sh[N/BLOCK_DIM];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    sh[threadIdx.x] = n1[tid];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride = stride / 2)
    {
        if (threadIdx.x < stride)
        {
            if (sh[threadIdx.x + stride] < sh[threadIdx.x])
                sh[threadIdx.x] = sh[threadIdx.x + stride];
            __syncthreads();
        }
    }
    if (threadIdx.x == 0)
        n2[blockIdx.x] = sh[0];

}


int main()
{
    int mat[N][N],rezEnd[BLOCK_DIM];
    int* devMat, *rez;


    srand(22);
    for(int i=0; i<N;i++)
        for (int j = 0; j < N; j++)
        {
            mat[i][j] = rand() % 10+2;
        }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    cudaMalloc((void**)&devMat, N * N * sizeof(int));
    cudaMalloc((void**)&rez, N/BLOCK_DIM * sizeof(int));

    cudaMemcpy(devMat, mat, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid(N / BLOCK_DIM, N / BLOCK_DIM);

    kernelMin << <dimGrid, dimBlock >> > (devMat, rez);
    redKernel<<<1,N/BLOCK_DIM>>>(rez, rez);
    cudaMemcpy(rezEnd, rez, N/BLOCK_DIM * sizeof(int), cudaMemcpyDeviceToHost);


    printf("\n");
    
        printf("Rezultat: %d ", rezEnd[0]);

    printf("\n");

}
