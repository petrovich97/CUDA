
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define N 8


__global__ void addKernel(int *devA, int* devB, int* rez)
{
	__shared__ int partSum[N];

	int tid = blockIdx.y * N + threadIdx.x;

	partSum[threadIdx.x] = devA[tid] + devB[tid];

	__syncthreads();

	for(int s=blockDim.x/2;s>0;s=s/2)
		if (threadIdx.x < s)
		{
			if (partSum[threadIdx.x + s] < partSum[threadIdx.x])
				partSum[threadIdx.x] = partSum[threadIdx.x + s];
			__syncthreads();
		}
	if (threadIdx.x == 0)
		rez[blockIdx.y] = partSum[0];

}

int main()
{
	int matA[N][N], matB[N][N], rezultat[N];
	int* devA, * devB, *rez;

	srand(22);
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			matA[i][j] = rand() % 10 ;
		}

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%d ", matA[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	srand(333);
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			matB[i][j] = rand() % 10 + 2;
		}
/// printam matrice zbog provere // // // // // // // // // // //
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%d ", matB[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%d ", matA[i][j]+matB[i][j]);
		}
		printf("\n");
	}
	printf("\n");
// // // // // // // // // // // // // // // // // // // // // //

	cudaMalloc((void**)&devA, N * N * sizeof(int));
	cudaMalloc((void**)&devB, N * N * sizeof(int));
	cudaMalloc((void**)&rez, N * sizeof(int));

	cudaMemcpy(devA, matA, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devB, matB, N * N * sizeof(int), cudaMemcpyHostToDevice);

	dim3 gridDim(1, N);

	addKernel << <gridDim, N >> > (devA, devB,rez);


	cudaMemcpy(rezultat, rez, N * sizeof(int), cudaMemcpyDeviceToHost);

	printf("REZULTAT.............................................................\n");

	for (int i = 0; i < N; i++)
		printf("%d ", rezultat[i]);


}
