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

#define BLOCKSIZE 32

__global__ void dgemm(double* A, double* B, double* C, int N)
{
 size_t myRow = blockIdx.y*blockDim.y+threadIdx.y;
 size_t myCol = blockIdx.x*blockDim.x+threadIdx.x;

 size_t localRow = threadIdx.y;
 size_t localCol = threadIdx.x;

 double c = 0.0;

 if (myRow >= N || myCol >= N) return;

 for (size_t m = 0; m < N; m += BLOCKSIZE)
 {
  __shared__ double shrA[BLOCKSIZE][BLOCKSIZE];
	__shared__ double shrB[BLOCKSIZE][BLOCKSIZE];

	shrA[localRow][localCol] = A[(myRow)*N        + (m + localCol)];
	shrB[localRow][localCol] = B[(m + localRow)*N + (myCol)];

	__syncthreads();

   #pragma unroll 32
   for (size_t i = 0; i < BLOCKSIZE; i++)
    c += shrA[localRow][i] * shrB[i][localCol];

  __syncthreads();
 }

 C[myRow * N + myCol] = c;
}

int main(int argc, char** argv)
{
  double *A, *B, *C;
  double *dA, *dB, *dC;
  size_t N = 12288;
  
  A = (double*) malloc (sizeof(double)*N*N);
  B = (double*) malloc (sizeof(double)*N*N);
  C = (double*) malloc (sizeof(double)*N*N);

  for (size_t i = 0; i < N; i++)
  for (size_t j = 0; j < N; j++)
  {
   A[i*N + j] = sin(i);
   B[i*N + j] = cos(j);
  }
     
  cudaMalloc(&dA, sizeof(double)*N*N); checkCUDAError("Error allocating dA");
  cudaMalloc(&dB, sizeof(double)*N*N); checkCUDAError("Error allocating dB");
  cudaMalloc(&dC, sizeof(double)*N*N); checkCUDAError("Error allocating dC");

  cudaMemcpy(dA, A, sizeof(double)*N*N, cudaMemcpyHostToDevice); checkCUDAError("Error copying A");
  cudaMemcpy(dB, B, sizeof(double)*N*N, cudaMemcpyHostToDevice); checkCUDAError("Error copying B");

  dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
  dim3 blocksPerGrid(N/BLOCKSIZE, N/BLOCKSIZE);

  auto startTime = std::chrono::system_clock::now();

  dgemm<<<blocksPerGrid,threadsPerBlock>>>(dA, dB, dC, N); checkCUDAError("Failed Kernel Launch");

  cudaDeviceSynchronize();
  auto endTime = std::chrono::system_clock::now();

  cudaMemcpy(C, dC, sizeof(double)*N*N, cudaMemcpyDeviceToHost);

  double checkSum = 0.0;
  for (size_t i = 0; i < N; i++)
  for (size_t j = 0; j < N; j++)
   checkSum += C[i*N + j];
  
  printf("[GPU] Checksum: %f - Elapsed Time: %fs\n", checkSum, std::chrono::duration<double>(endTime-startTime).count());

  return 0;
}

