#include <stdio.h>
#include <chrono>
#include "sampler/sampler.hpp"
#include <mpi.h>
#include <cstdlib>
#include <cstring>
#include <queue>

size_t nSamples;
size_t nParameters;

#define NSAMPLES 240
#define NPARAMETERS 2

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	int rank, size; //local for each process
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); //initalized with the command mpiexec
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	nSamples = NSAMPLES;	//
	nParameters = NPARAMETERS; //fixed 
	double* sampleArray =(double*) calloc (nSamples*nParameters, sizeof(double));
	int sampleId=0;
	bool finished = false;
	std::queue<int> actives;
	double* resultsArray = (double*) calloc (nSamples, sizeof(double));
	double result;
	MPI_Status status[size];

	auto start = std::chrono::system_clock::now();
	if(rank==0)	printf("Processing %ld Samples each with %ld Parameter(s)...\n", nSamples, nParameters);

	if(rank==0){
		for (int i=1; i < rank;i++) actives.push(i);
		double* gSampleArray = initializeSampler(NSAMPLES, nParameters);
		double* gResultsArray = (double*) calloc (NSAMPLES, sizeof(double));
		while(sampleId < NSAMPLES){
			while(actives.empty()){}
			if(!actives.empty()){
				printf(" %ld SampleID \n", sampleId);
				int idx =actives.front();actives.pop();
				MPI_Send(&gSampleArray[sampleId*nParameters], 1, MPI_DOUBLE, idx, 1,MPI_COMM_WORLD);
				MPI_Recv(&gResultsArray[sampleId],1, MPI_DOUBLE, idx, 1, MPI_COMM_WORLD, &status[idx]);
				actives.push(idx);
				sampleId++;
			}
		finished=true;
		}
	}else{
		while(!finished==true){
			printf(" %ld rank \n", rank);
			MPI_Status status_;
			MPI_Recv(&sampleArray[0], 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status_);
			result=evaluateSample(&sampleArray[0]);
			MPI_Send(&result, 1,MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
		}	
	}
			
	auto end = std::chrono::system_clock::now();

	if(rank==0){
		double totalTime = std::chrono::duration<double>(end-start).count();
		printf("%f s \n", totalTime);
	}

/*/-_______________-------_--------------------------------------------------
	if(rank==0){
		//printf("%d \n", size);
		//printf("%d \n", nSamples);
		double* gSampleArray = initializeSampler(NSAMPLES, nParameters);
		//for(int i=0;i<NSAMPLES;i++)	printf("%f \t",gSampleArray[i] );
		std::memcpy(&sampleArray[0], &gSampleArray[0], nSamples*nParameters*sizeof(double));
		for (int i = 1; i < size; i++){
			MPI_Isend(&gSampleArray[i*nSamples*nParameters],nSamples*nParameters,MPI_DOUBLE,i, 1,MPI_COMM_WORLD,&req);
		 }
	}else{
		//MPI_Status status;
		MPI_Irecv(&sampleArray[0],nSamples*nParameters, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &req);
	}
	MPI_Wait(&req, &status);

	//MPI_Barrier(MPI_COMM_WORLD);
	double* resultsArray = (double*) calloc (nSamples, sizeof(double));

	if(rank==0)	printf("Processing %ld Samples each with %ld Parameter(s)...\n", nSamples, nParameters);

	auto start = std::chrono::system_clock::now();
	for (size_t i = 0; i < nSamples; i++) resultsArray[i] = evaluateSample(&sampleArray[i*nParameters]);//printf("%f \n ",resultsArray[i]);}
	auto end = std::chrono::system_clock::now();
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Request req2;
	MPI_Status status2;
		//req[0] = MPI_REQUEST_NULL;
		//req[1] = MPI_REQUEST_NULL;
	if(rank==0){
		double* gResultsArray = (double*) calloc (NSAMPLES, sizeof(double));
		std::memcpy(&gResultsArray[0], &resultsArray[0], nSamples*sizeof(double));
		
		MPI_Status status[size-1];
		for (int i = 1; i < size; i++){
			MPI_Irecv(&gResultsArray[i*nSamples],nSamples, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &req2);	
		 }
		 //for(int i=0;i<NSAMPLES;i++)	printf("%f \t",gResultsArray[i] );
		 checkResults(gResultsArray);
	}else{
		MPI_Isend(&resultsArray[0],nSamples,MPI_DOUBLE,0, 1,MPI_COMM_WORLD,&req2);
	}
	MPI_Wait(&req2, &status2);

	double totalTime = std::chrono::duration<double>(end-start).count();
	printf("%f s \n", totalTime);

*/
	
	MPI_Finalize();
	return 0;
}
