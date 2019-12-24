#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <chrono>

void checkCUDAError(const char *msg)
{
 cudaError_t err = cudaGetLastError();
 if( cudaSuccess != err)
 {
  fprintf(stderr, "CUDA Error: %s: %s.\n", msg, cudaGetErrorString(err) );
  exit(EXIT_FAILURE);
 }
}

#define BLOCKSIZE 1024

__global__ void reduce(unsigned int* dVec, unsigned int* dAux, size_t N)
{
 __shared__ unsigned int sdata[BLOCKSIZE];

 size_t tid = threadIdx.x;
 size_t i = blockIdx.x*blockDim.x + threadIdx.x;

 sdata[tid] = dVec[i];
 __syncthreads();

 for(size_t s = 1; s < blockDim.x; s *= 2)
 {
  if (tid % (s*2) == 0) sdata[tid] += sdata[tid + s];
  __syncthreads();
 }

 if (tid == 0) dAux[blockIdx.x] = sdata[0];
}

int main(int argc, char** argv)
{
	unsigned int *vec;
	unsigned int *dVec, *dAux;

  size_t N0 = 32768;
  size_t N = N0*N0;

  vec = (unsigned int*) malloc (sizeof(unsigned int)*N);

  for (size_t i = 0; i < N; i++) vec[i] = i;

  cudaMalloc(&dVec,  sizeof(unsigned int)*N); checkCUDAError("Error allocating dVec");
  cudaMalloc(&dAux, sizeof(unsigned int)*N); checkCUDAError("Error allocating dAux");
  cudaMemcpy(dVec, vec, sizeof(unsigned int)*N, cudaMemcpyHostToDevice); checkCUDAError("Error copying vec");
  
  auto startTime = std::chrono::system_clock::now();

  for (size_t n = N; n > 1; n = n / BLOCKSIZE)
  {
  	size_t bSize = BLOCKSIZE;           if (bSize > n) bSize = n;
  	size_t gSize = ceil((double)n / (double)BLOCKSIZE); if (bSize > n) gSize = 1;
  	printf("bSize: %lu - gSize: %lu\n", bSize, gSize);
    reduce<<<gSize, bSize>>>(dVec, dAux, n); checkCUDAError("Failed Kernel Launch");
    unsigned int *tmp = dVec; dVec = dAux; dAux = tmp;
  }

  cudaDeviceSynchronize();

  auto endTime = std::chrono::system_clock::now();
  
  unsigned int result = 0.0;
  cudaMemcpy(&result, dVec, sizeof(unsigned int), cudaMemcpyDeviceToHost); checkCUDAError("Error getting result");

  printf("[GPU] Result: %u - Elapsed Time: %fs\n", result, std::chrono::duration<double>(endTime-startTime).count());

  return 0;
}

