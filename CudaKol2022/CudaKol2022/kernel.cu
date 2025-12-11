
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define n 32

#include <stdio.h>

#ifdef __CUDACC__
#define cuda_SYNCTHREADS() __syncthreads()
#else
#define cuda_SYNCTHREADS()
#endif

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);



__global__ void kernelK1(float* a, float* b, float p)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	__shared__ float shA[4 + 2];
	__shared__ float shB[4 + 2];

	while (tid < n)
	{
		shA[threadIdx.x] = a[tid];
		shB[threadIdx.x] = b[tid];

		if (threadIdx.x == blockDim.x - 1)
		{
			shA[threadIdx.x + 1] = a[tid + 1];
			shB[threadIdx.x + 1] = b[tid + 1];
			shA[threadIdx.x + 2] = a[tid + 2];
			shB[threadIdx.x + 2] = b[tid + 2];
		}
		__syncthreads();


		
		a[tid] = (shA[threadIdx.x] + shA[threadIdx.x + 1] + shA[threadIdx.x + 2]) * p + (shB[threadIdx.x] + shB[threadIdx.x + 1] + shB[threadIdx.x + 2]) * (1 - p);
		tid += gridDim.x * blockDim.x;
	}

}

int main()
{
	float A[n + 2], B[n + 2], C[n];
	float* a, * b;

	for (int i = 0; i < n + 2; i++)
	{
		A[i] = i;
		B[i] = i;
	}

	cudaMalloc((void**)&a, (n + 2) * sizeof(float));
	cudaMalloc((void**)&b, (n + 2) * sizeof(float));
	//cudaMalloc((void**)&c, n * sizeof(float));

	cudaMemcpy(a, A, (n + 2) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b, B, (n + 2) * sizeof(float), cudaMemcpyHostToDevice);

	kernelK1 << <4, 4 >> > (a, b, 0.5);

	cudaMemcpy(C, a, n * sizeof(float), cudaMemcpyDeviceToHost);

	printf("Niz C: \n");
	for (int i = 0; i < n; i++)
	{
		printf("%2f\n", C[i]);
	}

	cudaFree(a);
	cudaFree(b);
	//cudaFree(c);
}



