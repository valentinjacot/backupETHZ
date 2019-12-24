#include <stdio.h>
#include <chrono>
#include <queue>
#include <upcxx/upcxx.hpp>
#include "sampler/sampler.hpp"
#include <queue>
#define NSAMPLES 240
#define NPARAMETERS 2

int rankId;
int rankCount;
size_t nSamples;
size_t nParameters;
upcxx::global_ptr<double> gptr;
double* sampleArray; 
int lSamples;
bool finished = false;
int processed;
std::queue<int> actives;

int main(int argc, char* argv[])
{

	upcxx::init();
	rankId    = upcxx::rank_me();
	rankCount = upcxx::rank_n();
	nSamples = NSAMPLES;
	nParameters = NPARAMETERS;

	if (rankId == 0) printf("Processing %ld Samples each with %ld Parameter(s)...\n", nSamples, nParameters);
	
	if(rankId == 0)for (int i=1; i < rankCount;i++) actives.push(i);// master initializes the queue
	if(rankId > 0)bool l_finished=false;
	if (rankId == 0) processed=0;
	if (rankId == 0) gptr = upcxx::new_array<double>(nSamples);
	if (rankId == 0) printf("step1");
	
	auto t0 = std::chrono::system_clock::now();
	if (rankId == 0) sampleArray = initializeSampler(nSamples, nParameters);
	else  sampleArray = (double*)calloc(nSamples*nParameters,(sizeof(double)));
	double* resultsArray = (double*) calloc (nSamples, sizeof(double));
	upcxx::broadcast(sampleArray, nSamples*nParameters, 0).wait();
	upcxx::broadcast(&gptr, 1 , 0).wait(); //just the address
	upcxx::barrier();
		
	if(rankId==0){
	printf("step3\n");
		while(processed < nSamples){
			while(actives.empty()){upcxx::progress();}
			if(!actives.empty()){
				int idx =actives.front();actives.pop();
				upcxx::rpc_ff(idx, [](int processed){
					double result;
					result = evaluateSample(&sampleArray[processed*nParameters]);
					upcxx::rput(&result,gptr+processed,1);
					//printf("%d  \t %d \t %f \n",rankId, processed, result);
					upcxx::rpc_ff(0, [](int idx){
						actives.push(idx);
					},rankId);				
				 },processed);
			 }			 
			 processed++;
//			 upcxx::progress();
		 }
		 finished=true;
//		 upcxx::progress();
		for (int i=1;i<rankCount;i++) upcxx::rpc(i,[](){finished=true;});//for all workers!
	 }
	if (rankId>0){while(!finished){upcxx::progress();}} //everycode
	auto t1 = std::chrono::system_clock::now();
	upcxx::barrier();

	if (rankId == 0)
	{
	 resultsArray = gptr.local();
	 checkResults(resultsArray); // Make sure you check results!
	 double evalTime = std::chrono::duration<double>(t1-t0).count();
	 printf("Total Running Time: %.3fs\n", evalTime);
	}

	upcxx::finalize();
	return 0;
}
/*if (rankId == 0) {
  for (size_t i = 0; i < nSamples; i++){
	 
	 auto fut = upcxx::rpc(1, eval_wrapper, upcxx::rget(sampleArray + i * nSamples)+i*nParameters);
	 double res = fut.wait();
} else {
  upcxx::progress();
}
*/
