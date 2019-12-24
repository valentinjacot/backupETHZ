#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

int main(int argc, char** argv)
{
	unsigned int *vec;

  size_t N0 = 32768;
  size_t N = N0*N0;
  
  vec = (unsigned int*) malloc (sizeof(unsigned int)*N);

  for (size_t i = 0; i < N; i++) vec[i] = i;

  auto startTime = std::chrono::system_clock::now();

  unsigned int result = 0.0;
  for (size_t i = 0; i < N; i++) result += vec[i];

  auto endTime = std::chrono::system_clock::now();

  printf("[CPU] Result: %u - Elapsed Time: %fs\n", result, std::chrono::duration<double>(endTime-startTime).count());

  return 0;
}

