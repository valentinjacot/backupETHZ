#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "sampler/sampler.hpp"

size_t nSamples;
size_t nParameters;

#define NSAMPLES 240
#define NPARAMETERS 2

int main(int argc, char* argv[])
{
	nSamples = NSAMPLES;
	nParameters = NPARAMETERS;

	double* sampleArray = initializeSampler(nSamples, nParameters);
	double* resultsArray = (double*) calloc (nSamples, sizeof(double));

	printf("Processing %ld Samples each with %ld Parameter(s)...\n", nSamples, nParameters);

  auto start = std::chrono::system_clock::now();
  for (size_t i = 0; i < nSamples; i++) resultsArray[i] = evaluateSample(&sampleArray[i*nParameters]);
  auto end = std::chrono::system_clock::now();

  checkResults(resultsArray);

  double totalTime = std::chrono::duration<double>(end-start).count();
  printf("Total Running Time: %.3fs\n", totalTime);

	return 0;
}
