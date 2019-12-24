#include <stdio.h>
#include <chrono>
#include <queue>
#include <upcxx/upcxx.hpp>
#include "sampler/sampler.hpp"

#define NSAMPLES 240
#define NPARAMETERS 2

int rankId;
int rankCount;
size_t nSamples;
size_t nParameters;
upcxx::global_ptr<double> sampleArray;

int main(int argc, char* argv[])
{
	upcxx::init();
	rankId    = upcxx::rank_me();
	rankCount = upcxx::rank_n();
	nSamples = NSAMPLES;
	nParameters = NPARAMETERS;

  if (rankId == 0) printf("Processing %ld Samples (24 initially available), each with %ld Parameter(s)...\n", nSamples, nParameters);
  if (rankId == 0) initializeSampler(nSamples, nParameters);

  auto t0 = std::chrono::system_clock::now();

	auto t1 = std::chrono::system_clock::now();

	if (rankId == 0)
	{
	checkResults(/* This will FAIL */ ); // Make sure you check results!
	 double evalTime = std::chrono::duration<double>(t1-t0).count();
	 printf("Total Running Time: %.3fs\n", evalTime);
	}

  upcxx::finalize();
	return 0;
}


