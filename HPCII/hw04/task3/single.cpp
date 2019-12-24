#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "sampler/sampler.hpp"

size_t nSamples;
size_t nParameters;
size_t nInitialSamples;

#define NSAMPLES 240
#define NPARAMETERS 2

void processSample(size_t sampleId)
{
  double sample[nParameters];
  getSample(sampleId, sample);
  double eval = evaluateSample(sample);
  updateEvaluation(sampleId, eval);
}

int main(int argc, char* argv[])
{
	nSamples = NSAMPLES;
	nParameters = NPARAMETERS;

	printf("Processing %ld Samples (24 initially available), each with %ld Parameter(s)...\n", nSamples, nParameters);

	initializeSampler(nSamples, nParameters);

  auto start = std::chrono::system_clock::now();
  for (size_t sampleId = 0; sampleId < nSamples; sampleId++) processSample(sampleId);
  auto end = std::chrono::system_clock::now();

  checkResults();

  double totalTime = std::chrono::duration<double>(end-start).count();
  printf("Total Running Time: %.3fs\n", totalTime);

	return 0;
}
