
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define N 32
#define GRID_DIM 4

__global__ void addKernel(int *devA,int *devB)
{
	__shared__ int temp[N / GRID_DIM];

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	temp[threadIdx.x] = devA[tid] + devB[tid];
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s = s / 2)
	{
		if (threadIdx.x < s)
		{
			if (temp[threadIdx.x + s] > temp[threadIdx.x])
				temp[threadIdx.x] = temp[threadIdx.x + s];
			__syncthreads();
		}
	}
	if (threadIdx.x == 0)
		devA[blockIdx.x] = temp[0];  
}

__global__ void redKernel(int* devA)
{
	__shared__ int temp[N / GRID_DIM];

	temp[threadIdx.x] = devA[threadIdx.x];
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s = s / 2)
	{
		if (threadIdx.x < s)
		{
			if (temp[threadIdx.x + s] > temp[threadIdx.x])
				temp[threadIdx.x] = temp[threadIdx.x + s];
			__syncthreads();
		}
	}
	if (threadIdx.x == 0)
		devA[0] = temp[0];
}

int main()
{
    int nizA[N], nizB[N], rez[1];
	int* devA, * devB;// *devR;


	srand(333);
	for (int i = 0; i < N; i++)
		nizA[i] = rand() % 10 + 2;
		
	srand(232);
	for (int i = 0; i < N; i++)
		nizB[i] = rand() % 10 ;


	// printam nizove zbog provere 
	printf("Niz A: \n");
	for (int i = 0; i < N; i++)
		printf("%d ", nizA[i]);

	printf("\n Niz B:\n");
	for (int i = 0; i < N; i++)
		printf("%d ", nizB[i]);

	printf("\n A+B:\n");
	for (int i = 0; i < N; i++)
		printf("%d ", nizB[i] + nizA[i]);
	// // // // // // // // // // // // // // 

	cudaMalloc((void**)&devA, N * sizeof(int));
	cudaMalloc((void**)&devB, N * sizeof(int));
	//cudaMalloc((void**)&devR, GRID_DIM * sizeof(int));

	cudaMemcpy(devA, nizA, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devB, nizB, N * sizeof(int), cudaMemcpyHostToDevice);

	addKernel << <GRID_DIM, N / GRID_DIM >> > (devA, devB);
	redKernel << < 1, N / GRID_DIM >> > (devA);

	cudaMemcpy(rez, devA, sizeof(int), cudaMemcpyDeviceToHost);

	printf("\n MAX rezultat: %d ", rez[0]);


		
}
