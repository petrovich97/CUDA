
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define N 8

__global__ void vrstaKernel(int* mA, int* mB)
{
	__shared__ int shVrstaA[N];
	__shared__ int shVrstaB[N];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	shVrstaA[threadIdx.x] = mA[tid];
	shVrstaB[threadIdx.x] = mB[tid];

	__syncthreads();
	
	//if (threadIdx.x < blockDim.x / 2)
	//{
	if (shVrstaA[threadIdx.x] < shVrstaB[threadIdx.x])
	{
		mA[tid] = shVrstaB[threadIdx.x];
		shVrstaA[threadIdx.x] = shVrstaB[threadIdx.x];
	}
	//	if (shVrstaA[threadIdx.x + blockDim.x / 2] < shVrstaB[threadIdx.x + blockDim.x / 2])
	//		mA[tid + blockDim.x / 2] = shVrstaB[threadIdx.x + blockDim.x / 2];
	//}
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride = stride / 2)
	{
		if (threadIdx.x < stride)
		{
			if (shVrstaA[threadIdx.x + stride] < shVrstaA[threadIdx.x])
				shVrstaA[threadIdx.x] = shVrstaA[threadIdx.x + stride];

		__syncthreads();
		}
	}
	if (threadIdx.x == 0)
		mB[blockIdx.x] = shVrstaA[threadIdx.x];

}


int main()
{
	int matA[N][N], matB[N][N], matC[N][N], vekC[N];
	int* devA, * devB;

	srand(22);

	for(int i =0; i<N;i++)
		for (int j = 0; j < N; j++)
		{
			matA[i][j] = rand() % 10 + 24;
			matB[i][j] = rand() % 100 * 1.5;
		}
 //stampa radi provere
	printf("Mat A:\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%d ", matA[i][j]);
		}
		printf("\n");
	}

	printf("Mat B:\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%d ", matB[i][j]);
		}
		printf("\n");
	}


	cudaMalloc((void**)&devA, N * N * sizeof(int));
	cudaMalloc((void**)&devB, N * N * sizeof(int));

	cudaMemcpy(devA, matA, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devB, matB, N * N * sizeof(int), cudaMemcpyHostToDevice);

	vrstaKernel << <N, N >> > (devA, devB);

	cudaMemcpy(matC, devA, N * N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(vekC, devB, N * sizeof(int), cudaMemcpyDeviceToHost);

	printf("Mat C:\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%d ", matC[i][j]);
		}
		printf("\n");
	}

	printf("Vektor C: ");
	for (int i = 0; i < N; i++)
		printf("%d ", vekC[i]);

	cudaFree(devA);
	cudaFree(devB);

}
