#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

#define N 1024
#define BLOCK_DIM 128
#define GRID_DIM 8


struct kruznica { int x; int y; int r;  double povrsina; };

__device__ double povrsina(kruznica a)
{
	return (a.r*a.r*3.14);
}

__global__ void nadji(kruznica  *devNiz)
{
	__shared__ kruznica sh[BLOCK_DIM];
	
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	sh[threadIdx.x] = devNiz[tid];
	__syncthreads();

	for (int s = blockDim.x / 2; s >= 1; s = s / 2) //  for (int s = blockDim.x / 2; s >= 2; s = s / 2)
	{
		if (threadIdx.x < s)
		{
			sh[threadIdx.x].povrsina = povrsina(sh[threadIdx.x]);
			sh[threadIdx.x + s].povrsina = povrsina(sh[threadIdx.x + s]);

			if (sh[threadIdx.x + s].povrsina > sh[threadIdx.x].povrsina)
			{
				sh[threadIdx.x] = sh[threadIdx.x + s];
			}
			__syncthreads();
		}
	}
	if (threadIdx.x ==0) // <2
		devNiz[threadIdx.x] = sh[threadIdx.x]; // devNiz[blockIdx.x*blockDim.x+threadIdx.x]=sh[threadIdx.x];


}



__global__ void sortiraj(kruznica *devNiz)
{
	__shared__ kruznica sh[BLOCK_DIM];
	sh[threadIdx.x] = devNiz[threadIdx.x];
	__syncthreads();

	for (int s = blockDim.x; s >= 1; s = s / 2) // for (int s = blockDim.x; s >= 2; s = s / 2)
	{
		if (threadIdx.x < s);
		{
			if (sh[threadIdx.x + s].povrsina > sh[threadIdx.x].povrsina)
			{
				sh[threadIdx.x] = sh[threadIdx.x + s];
			}
			__syncthreads();
		}
	}
	if (threadIdx.x ==0) // < 2
	{
		devNiz[threadIdx.x] = sh[threadIdx.x];
	}
}

int main()
{
	kruznica niz[N];
	kruznica* devNiz;
	kruznica *rez; // rez[2];

	int indeks;

	printf("Unesite poziciju tacke u nizu: ");
	scanf("%d", &indeks);



	cudaMalloc((void**)&devNiz, N * sizeof(kruznica));
	rez = new kruznica;

	for (int i = 0; i < N; i++)
	{
		niz[i].x = 0;
		niz[i].y = i;
		niz[i].povrsina = 0;
	}

	cudaMemcpy(devNiz, niz, N * sizeof(kruznica), cudaMemcpyHostToDevice);

	nadji << <GRID_DIM, BLOCK_DIM >> > (devNiz);
	sortiraj << <1, BLOCK_DIM >> > (devNiz);

	cudaMemcpy(rez, devNiz, 1 * sizeof(kruznica), cudaMemcpyDeviceToHost); //2*sizeof(kruznica)

	printf("Povrsina minimalne: %d", rez->povrsina );

	cudaFree(devNiz);


}
