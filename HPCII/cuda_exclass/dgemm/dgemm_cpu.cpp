#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

int main(int argc, char** argv)
{
  double *A, *B, *C;
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
     
  auto startTime = std::chrono::system_clock::now();
  
  for (size_t i = 0; i < N; i++)
   for (size_t j = 0; j < N; j++)
   {
    C[i * N + j] = 0;
    for (size_t k = 0; k < N; k++)
      C[i * N + j] += A[i * N + k] * B[k * N + j];
   } 
   
  auto endTime = std::chrono::system_clock::now();
  
  double checkSum = 0.0;
  for (size_t i = 0; i < N; i++)
  for (size_t j = 0; j < N; j++)
   checkSum += C[i*N + j];
  
  printf("[CPU] Checksum: %f - Elapsed Time: %fs\n", checkSum, std::chrono::duration<double>(endTime-startTime).count());

  return 0;
}

 
