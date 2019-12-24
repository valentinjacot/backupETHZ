#include <stdio.h>
#include <chrono>
#include <upcxx/upcxx.hpp>
#include "sampler/sampler.hpp"

size_t nSamples;
size_t nParameters;
int rankId;
int rankCount;
upcxx::global_ptr<double> gptr;
double* sampleArray; 
int lSamples;
bool finished = false;
#define NSAMPLES 240
#define NPARAMETERS 2

int main(int argc, char* argv[])
{
	upcxx::init();
	rankId    = upcxx::rank_me();
	rankCount = upcxx::rank_n();
	
	nSamples = NSAMPLES;
	lSamples = NSAMPLES/rankCount;
	nParameters = NPARAMETERS;
	//n_local = nSamples/rankCount;
	
//	upcxx::dist_object<upcxx::global_ptr<double>> gptr(upcxx::new_array<double>(lSamples));
//	upcxx::dist_object<upcxx::global_ptr<double>> gsmpl_ptr(upcxx::new_array<double>(nSamples));
//	double *local_ptr = gptr->local();
//	double *lsmpl_ptr = gsmpl_ptr->local();
//	upcxx::global_ptr<double> lsmpl_ptr = nullptr;
	
//	upcxx::global_ptr<double> gsmpl_ptr;
	if (rankId == 0) gptr = upcxx::new_array<double>(nSamples);
//	if (rankId == 0) gsmpl_ptr = upcxx::new_array<double>(lSamples * nParameters);
//	upcxx::broadcast(&gptr, 1, 0).wait();
//	double *local_ptr = gptr.local();
//	double *lsmpl_ptr = gsmpl_ptr.local();
	auto start = std::chrono::system_clock::now();

	if(rankId==0) sampleArray = initializeSampler(lSamples, nParameters);
	else  sampleArray = (double*)calloc(nSamples*nParameters,(sizeof(double)));
	
	double* resultsArray = (double*) calloc (nSamples, sizeof(double));
	
//	resultsArray = (double*)calloc(nSamples*nParameters,(sizeof(double)));
//	if(rankId==0) lsmpl_ptr = initializeSampler(lSamples, nParameters);
	upcxx::broadcast(sampleArray, nSamples*nParameters, 0).wait();
	upcxx::broadcast(&gptr, 1 , 0).wait(); //just the address
	upcxx::barrier();
	
	if(rankId ==0){ for(int rank=1;rank<rankCount; rank++){
		upcxx::rpc_ff( rank, [](){
				double result;
				for (size_t i = (rankId)*lSamples; i < lSamples * (rankId+1); i++){
					 result = evaluateSample(&sampleArray[i*nParameters]);
					 upcxx::rput(&result,gptr+i,1);
//					 printf("%d  \t %d \t %f \n",rankId, i, result);
				 }
			finished=true;	 				
			});
		}
		
	for (size_t i = 0; i < lSamples; i++){
		 double result = evaluateSample(&sampleArray[i*nParameters]);
		 upcxx::rput(&result,gptr+i,1);
//		 printf("%d  \t %d \t %f \n",rankId, i, result);

	}
}	
	if (rankId>0){while(!finished){upcxx::progress();}} //everycode
	auto end = std::chrono::system_clock::now();
	upcxx::barrier();

	if(rankId==0){ 
		resultsArray = gptr.local();
		checkResults(resultsArray);
	}

/*	if (rankId==0){
		double* gResultsArray = (double*) calloc (NSAMPLES, sizeof(double));
		std::memcpy(&gResultsArray[0], &local_ptr[0], nSamples);
		checkResults(gResultsArray);
	}	
*/	
	double totalTime = std::chrono::duration<double>(end-start).count();

	printf("Rank: %d Total Running Time: %.3fs\n", rankId , totalTime);
	
	//upcxx::delete_array(gptr);
	//upcxx::delete_array(gsmpl_ptr);
	upcxx::finalize();
	return 0;
}

//	upcxx::dist_object<upcxx::global_ptr<double>> u_g(upcxx::new_array<double>(n_local)):
//	double *u = u_g.local();
//	printf("Rank %d: ", rankId);
//	if(rankId==0){
//		double* sampleArray = initializeSampler(nSamples, nParameters);
//		double* resultsArray = (double*) calloc (nSamples, sizeof(double));
//	}
	
