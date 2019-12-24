#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <chrono>

void checkCUDAError(const char *msg)
{
 cudaError_t err = cudaGetLastError();
 if( cudaSuccess !=err )
 {
   fprintf(stderr," CUDA Error: %s: %s. \n",msg, cudaGetErrorString(err));
	exit(-1);	
 }
}

__global__ void dgemm(double *A, double *B, double *C, size_t N)
{
	size_t myRow = blockIdx.y*blockDim.y + threadIdx.y;
	size_t myCol = blockIdx.x*blockDim.x + threadIdx.x;

//if(myRow < N %% myCol < N)
	C[myRow*N + myCol]=0;
	for (size_t i = 0; i < N; i++)
		C[myRow*N + myCol] += A[myRow*N +i] * B[i*N + myCol];
}


//pw: lo48cylrt08
int main(int argc, char** argv)
{
  double *A, *B, *C;
  double *dA, *dB, *dC;
 
  size_t N = 2048;
  
  A = (double*) malloc (sizeof(double)*N*N);
  B = (double*) malloc (sizeof(double)*N*N);
  C = (double*) malloc (sizeof(double)*N*N);

  for (size_t i = 0; i < N; i++)
  for (size_t j = 0; j < N; j++)
  {
   A[i*N + j] = sin(i);
   B[i*N + j] = cos(j);
  }
	cudaSetDevice(0);

	cudaMalloc(&dA, sizeof(double)*N*N);	
	checkCUDAError("Error allocating dA \n");
	cudaMalloc(&dB, sizeof(double)*N*N);	
	checkCUDAError("Error allocating dB \n");
	cudaMalloc(&dC, sizeof(double)*N*N);	
	checkCUDAError("Error allocating dC \n");
     
    cudaMemcpy(dA,A, sizeof(double)*N*N,cudaMemcpyHostToDevice);checkCUDAError("Error coping A \n");
    cudaMemcpy(dB,B, sizeof(double)*N*N,cudaMemcpyHostToDevice);checkCUDAError("Error coping B \n");
    //cudaMemset(dC, size, 0);
  auto startTime = std::chrono::system_clock::now();
  
  dim3 threadsPerBlock(32,32);
  dim3 blocksPerGrid(N/32,N/32);//depends on the size of the problem

  dgemm<<< blocksPerGrid, threadsPerBlock >>>(dA,dB, dC,N);checkCUDAError("Error executing kernel \n");
  
  cudaMemcpy(C,dC, sizeof(double)*N*N,cudaMemcpyDeviceToHost);checkCUDAError("Error coping C \n");

  cudaDeviceSynchronize();
/*  
  for (size_t i = 0; i < N; i++)
   for (size_t j = 0; j < N; j++)
   {
    C[i * N + j] = 0;
    for (size_t k = 0; k < N; k++)
      C[i * N + j] += A[i * N + k] * B[k * N + j];
   } 
*/   
  auto endTime = std::chrono::system_clock::now();
  
  double checkSum = 0.0;
  for (size_t i = 0; i < N; i++)
  for (size_t j = 0; j < N; j++)
   checkSum += C[i*N + j];
  
  printf("[CPU] Checksum: %f - Elapsed Time: %fs\n", checkSum, std::chrono::duration<double>(endTime-startTime).count());

  return 0;
}

 
