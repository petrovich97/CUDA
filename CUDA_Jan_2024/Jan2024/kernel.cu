#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

#define N 1024
#define BLOCK_DIM 128
#define GRID_DIM 8


struct tacka { int x; int y; double rastojanje; };

__device__ double udaljenost(tacka a, tacka b)
{
	return sqrt(pow(abs((a.x - b.x)), 2) + pow(abs((a.y - b.y)), 2));
}

__global__ void nadji(tacka* devNiz, int indeks)
{
	__shared__ tacka sh[BLOCK_DIM];
	__shared__ tacka test;
	test.x = devNiz[indeks].x;
	test.y = devNiz[indeks].y;
	test.rastojanje = devNiz[indeks].rastojanje;

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	sh[threadIdx.x].x = devNiz[tid].x;
	sh[threadIdx.x].y = devNiz[tid].y;
	sh[threadIdx.x].rastojanje = devNiz[tid].rastojanje;
	__syncthreads();

	for (int s = blockDim.x / 2; s >= 8; s = s / 2)
	{
		if (threadIdx.x < s)
		{
			sh[threadIdx.x].rastojanje = udaljenost(sh[threadIdx.x], test);
			sh[threadIdx.x + s].rastojanje = udaljenost(sh[threadIdx.x + s], test);

			if (sh[threadIdx.x + s].rastojanje < sh[threadIdx.x].rastojanje)
			{
				sh[threadIdx.x].x = sh[threadIdx.x + s].x;
				sh[threadIdx.x].y = sh[threadIdx.x + s].y;
				sh[threadIdx.x].rastojanje = sh[threadIdx.x + s].rastojanje;
			}
			__syncthreads();
		}
	}
	if (threadIdx.x < 10)
		devNiz[blockIdx.x * 10 + threadIdx.x] = sh[threadIdx.x];


}



__global__ void sortiraj(tacka* devNiz)
{
	__shared__ tacka sh[BLOCK_DIM];
	int tid = threadIdx.x;

	sh[threadIdx.x].x = devNiz[tid].x;
	sh[threadIdx.x].y = devNiz[tid].y;
	sh[threadIdx.x].rastojanje = devNiz[tid].rastojanje;
	__syncthreads();

	for (int s = blockDim.x; s >= 8; s = s / 2)
	{
		if (threadIdx.x < s);
		{
			if (sh[threadIdx.x + s].rastojanje < sh[threadIdx.x].rastojanje)
			{
				sh[threadIdx.x].x = sh[threadIdx.x + s].x;
				sh[threadIdx.x].y = sh[threadIdx.x + s].y;
				sh[threadIdx.x].rastojanje = sh[threadIdx.x + s].rastojanje;
			}
			__syncthreads();
		}
	}
	if (threadIdx.x < 10)
	{
		devNiz[threadIdx.x].x = sh[threadIdx.x].x;
		devNiz[threadIdx.x].y = sh[threadIdx.x].y;
		devNiz[threadIdx.x].rastojanje = sh[threadIdx.x].rastojanje;
	}
}

int main()
{
	tacka niz[N];
	tacka* devNiz, * devNiz2;
	tacka rezNiz1[80];

	int indeks;

	printf("Unesite poziciju tacke u nizu: ");
	scanf("%d", &indeks);



	cudaMalloc((void**)&devNiz, N * sizeof(tacka));
	cudaMalloc((void**)&devNiz2, 80 * sizeof(tacka));

	for (int i = 0; i < N; i++)
	{
		niz[i].x = 0;
		niz[i].y = i;
		niz[i].rastojanje = 0;
	}

	for (int i = 0; i < 10; i++)
	{
		rezNiz1[i].x = -5;
		rezNiz1[i].y = -5;
		rezNiz1[i].rastojanje = -5;
	}

	//	for (int i = 0; i < N; i++)
	//		printf("Tacka %d: koordinate (%d, %d), rastojanje: %f \n", i, niz[i].x, niz[i].y, niz[i].rastojanje);

	cudaMemcpy(devNiz, niz, N * sizeof(tacka), cudaMemcpyHostToDevice);

	nadji << <GRID_DIM, BLOCK_DIM >> > (devNiz, indeks);

	cudaMemcpy(rezNiz1, devNiz, 80 * sizeof(tacka), cudaMemcpyDeviceToHost);

	//for (int i = 0; i < 100; i++)
	//	printf("Tacka %d: koordinate (%d, %d),  rastojanje od tacke (%d, %d): %f \n", i, rezNiz1[i].x, rezNiz1[i].y, niz[indeks].x, niz[indeks].y, rezNiz1[i].rastojanje);

	cudaMemcpy(devNiz2, rezNiz1, 80 * sizeof(tacka), cudaMemcpyHostToDevice);

	sortiraj << <1, 80 >> > (devNiz2);

	cudaMemcpy(rezNiz1, devNiz2, 10 * sizeof(tacka), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 10; i++)
		printf("Tacka %d: koordinate (%d, %d),  rastojanje od tacke (%d, %d): %f \n", i, rezNiz1[i].x, rezNiz1[i].y, niz[indeks].x, niz[indeks].y, rezNiz1[i].rastojanje);

	cudaFree(devNiz);


}
