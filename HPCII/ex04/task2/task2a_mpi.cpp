#include <stdio.h>
#include <chrono>
#include "sampler/sampler.hpp"
#include <mpi.h>
#include <cstdlib>
#include <cstring>

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
	nSamples = NSAMPLES/size;	//
	nParameters = NPARAMETERS; //fixed 
	double* sampleArray =(double*) calloc (nSamples*nParameters, sizeof(double));

	if(rank==0){
		//printf("%d \n", size);
		//printf("%d \n", nSamples);
		double* gSampleArray = initializeSampler(NSAMPLES, nParameters);
		//for(int i=0;i<NSAMPLES;i++)	printf("%f \t",gSampleArray[i] );
		std::memcpy(&sampleArray[0], &gSampleArray[0], nSamples*nParameters*sizeof(double));
		for (int i = 1; i < size; i++){
			MPI_Send(&gSampleArray[i*nSamples*nParameters],nSamples*nParameters,MPI_DOUBLE,i, 1,MPI_COMM_WORLD);
		 }
	}else{
		MPI_Status status;
		MPI_Recv(&sampleArray[0],nSamples*nParameters, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status );
	}
	//MPI_Barrier(MPI_COMM_WORLD);
	double* resultsArray = (double*) calloc (nSamples, sizeof(double));

	if(rank==0)	printf("Processing %ld Samples each with %ld Parameter(s)...\n", nSamples, nParameters);

	auto start = std::chrono::system_clock::now();
	for (size_t i = 0; i < nSamples; i++) resultsArray[i] = evaluateSample(&sampleArray[i*nParameters]);//printf("%f \n ",resultsArray[i]);}
	auto end = std::chrono::system_clock::now();
	MPI_Barrier(MPI_COMM_WORLD);

	
	if(rank==0){
		double* gResultsArray = (double*) calloc (NSAMPLES, sizeof(double));
		std::memcpy(&gResultsArray[0], &resultsArray[0], nSamples*sizeof(double));
		
		MPI_Status status[size-1];
		for (int i = 1; i < size; i++){
			MPI_Recv(&gResultsArray[i*nSamples],nSamples, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status[i-1]);	
		 }
		 //for(int i=0;i<NSAMPLES;i++)	printf("%f \t",gResultsArray[i] );
		 checkResults(gResultsArray);
	}else{
		MPI_Send(&resultsArray[0],nSamples,MPI_DOUBLE,0, 1,MPI_COMM_WORLD);
	}
	
	double totalTime = std::chrono::duration<double>(end-start).count();
	printf("%f s \n", totalTime);

	
	MPI_Finalize();

	return 0;
}
