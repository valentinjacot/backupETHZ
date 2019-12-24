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
int sampleId;
std::queue<int> actives;
bool finished = false;

/*
void processSample(size_t sampleId)
{
  double sample[nParameters];
  getSample(sampleId, sample);
  double eval = evaluateSample(sample);
  updateEvaluation(sampleId, eval);
}
*/
int main(int argc, char* argv[])
{
	upcxx::init();
	rankId    = upcxx::rank_me();
	rankCount = upcxx::rank_n();
	nSamples = NSAMPLES;
	nParameters = NPARAMETERS;
	
	if (rankId == 0) {sampleId=0;}
	//if (rankId == 0) gptr = upcxx::new_array<double>(nSamples);
	if(rankId == 0)for (int i=1; i < rankCount;i++)actives.push(i);
	
	if (rankId == 0) printf("Processing %ld Samples (24 initially available), each with %ld Parameter(s)...\n", nSamples, nParameters);
	if (rankId == 0) sampleArray = upcxx::new_array<double>(nSamples*nParameters);
	if (rankId == 0) initializeSampler(nSamples, nParameters);
//	else  sampleArray = (double*)calloc(nSamples*nParameters,(sizeof(double)));
	upcxx::broadcast(&sampleArray, 1, 0).wait();//only the pointer 
	upcxx::barrier();

	auto t0 = std::chrono::system_clock::now();
	if (rankId == 0){
	//printf("step3\n");
		upcxx::progress();
		while(sampleId < nSamples){
			while(actives.empty()){upcxx::progress();}
			if(!actives.empty()){
				int temp = actives.front(); actives.pop();
				getSample(sampleId, sampleArray.local()+sampleId*nParameters);
				upcxx::rpc_ff(temp, [](int sampleId){
					double eval = evaluateSample(sampleArray.local() + sampleId*nParameters);
					printf("%d  \t %d \t %f \n",rankId, sampleId, eval);
					upcxx::rpc_ff(0,[sampleId,eval](int idx){updateEvaluation(sampleId,eval);actives.push(idx);}, rankId);},
				sampleId);
			}
			sampleId++;
			upcxx::progress();
			}
			for (int i=1;i<rankCount;i++) upcxx::rpc(i,[](){finished=true;});//for all workers!
		}
	if(rankId>0){while(!finished){upcxx::progress();}}	
	auto t1 = std::chrono::system_clock::now();
	
	upcxx::barrier();
	
	if (rankId == 0)
	{
		checkResults();
		double evalTime = std::chrono::duration<double>(t1-t0).count();
		printf("Total Running Time: %.3fs\n", evalTime);
	}

	upcxx::finalize();
	return 0;
}



