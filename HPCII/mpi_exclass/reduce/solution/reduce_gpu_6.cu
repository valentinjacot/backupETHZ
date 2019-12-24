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

template <unsigned int blockSize> __device__ void warpReduce(volatile unsigned int* sdata, int tid) {
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize> __global__ void reduce(unsigned int* dVec, unsigned int* dAux, size_t N)
{
 __shared__ unsigned int sdata[BLOCKSIZE];

 size_t tid = threadIdx.x;
 size_t i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

 sdata[tid] = dVec[i] + dVec[i+blockDim.x];

 __syncthreads();

 if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
 if (blockSize >= 512)  { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
 if (blockSize >= 256)  { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
 if (blockSize >= 128)  { if (tid < 64)  { sdata[tid] += sdata[tid + 64];  } __syncthreads(); }
 if (tid < 32) warpReduce<blockSize>(sdata, tid);

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
    size_t bSize = BLOCKSIZE;
    size_t gSize = floor((double)n / (2.0*(double)BLOCKSIZE));
    if (gSize == 0) { gSize = 2; bSize = n/4; }
    printf("bSize: %lu - gSize: %lu\n", bSize, gSize);

    switch (bSize)
    {
     case 1024:  reduce<1024><<< gSize, bSize>>>(dVec, dAux, N); break;
     case 512:   reduce<512><<<  gSize, bSize>>>(dVec, dAux, N); break;
     case 256:   reduce<256><<<  gSize, bSize>>>(dVec, dAux, N); break;
     case 128:   reduce<128><<<  gSize, bSize>>>(dVec, dAux, N); break;
     case 64:    reduce< 64><<<  gSize, bSize>>>(dVec, dAux, N); break;
     case 32:    reduce< 32><<<  gSize, bSize>>>(dVec, dAux, N); break;
     case 16:    reduce< 16><<<  gSize, bSize>>>(dVec, dAux, N); break;
     case 8:     reduce< 8><<<   gSize, bSize>>>(dVec, dAux, N); break;
     case 4:     reduce< 4><<<   gSize, bSize>>>(dVec, dAux, N); break;
     case 2:     reduce< 2><<<   gSize, bSize>>>(dVec, dAux, N); break;
     case 1:     reduce< 1><<<   gSize, bSize>>>(dVec, dAux, N); break;
/*    size_t bSize = BLOCKSIZE;
    size_t gSize = floor((double)n / (2.0*(double)BLOCKSIZE));
    if (gSize == 0) { gSize = 2; bSize = n/4; }
    printf("bSize: %lu - gSize: %lu\n", bSize, gSize);

    switch (bSize)
    {
     case 1024:  reduce<1024><<< gSize, bSize>>>(dRescpy, dtemp, n); break;
     case 512:   reduce<512><<<  gSize, bSize>>>(dRescpy, dtemp, n); break;
     case 256:   reduce<256><<<  gSize, bSize>>>(dRescpy, dtemp, n); break;
     case 128:   reduce<128><<<  gSize, bSize>>>(dRescpy, dtemp, n); break;
     case 64:    reduce< 64><<<  gSize, bSize>>>(dRescpy, dtemp, n); break;
     case 32:    reduce< 32><<<  gSize, bSize>>>(dRescpy, dtemp, n); break;
     case 16:    reduce< 16><<<  gSize, bSize>>>(dRescpy, dtemp, n);break;
     case 8:     reduce< 8><<<   gSize, bSize>>>(dRescpy, dtemp, n);break;
     case 4:     reduce< 4><<<   gSize, bSize>>>(dRescpy, dtemp, n);break;
     case 2:     reduce< 2><<<   gSize, bSize>>>(dRescpy, dtemp, n);break;
     case 1:     reduce< 1><<<   gSize, bSize>>>(dRescpy, dtemp, n);break;
    }
    double *tmp = dRescpy; dRescpy = dtemp; dtemp = tmp;
    */
}

    unsigned int *tmp = dVec; dVec = dAux; dAux = tmp;
  }

  cudaDeviceSynchronize();

  auto endTime = std::chrono::system_clock::now();
  
  unsigned int result = 0.0;
  cudaMemcpy(&result, dVec, sizeof(unsigned int), cudaMemcpyDeviceToHost); checkCUDAError("Error getting result");

  printf("[GPU] Result: %u - Elapsed Time: %fs\n", result, std::chrono::duration<double>(endTime-startTime).count());

  return 0;
}

