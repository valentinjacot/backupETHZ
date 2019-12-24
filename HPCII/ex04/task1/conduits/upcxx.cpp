#include "conduits/upcxx.h"
#include "solvers/base.h"
#include <chrono>
#include <queue>

extern Korali::Solver::Base*  _solver;
extern Korali::Problem::Base* _problem;

int rankId;
int rankCount;
size_t nSamples;
size_t nParameters;
upcxx::global_ptr<double> sampleArrayPointer;
std::queue<int> actives;
bool finished = false;


Korali::Conduit::UPCXX::UPCXX(Korali::Solver::Base* solver) : Base::Base(solver) {};

void Korali::Conduit::UPCXX::processSample(size_t sampleId)
{
	upcxx::init();
	rankId    = upcxx::rank_me();
	rankCount = upcxx::rank_n();

	auto t0 = std::chrono::system_clock::now();
	if (rankId == 0){
		upcxx::progress();
		//while(sampleId < nSamples){
			while(actives.empty()){upcxx::progress();}
			if(!actives.empty()){
				int temp = actives.front(); actives.pop();
				upcxx::rpc_ff(temp, [](int sampleId){
					double fitness = _problem -> evaluateSample(sampleArrayPointer.local() + sampleId*nParameters);
	//				printf("%d  \t %d \t %f \n",rankId, sampleId, fitness);
					upcxx::rpc_ff(0,[sampleId,fitness](int idx){_solver->updateEvaluation(sampleId,fitness);actives.push(idx);}, rankId);},
				sampleId);
		}
		upcxx::progress();
	}
	auto t1 = std::chrono::system_clock::now();

	if (rankId == 0)
	{
		double evalTime = std::chrono::duration<double>(t1-t0).count();
//		printf("Total Running Time: %.3fs\n", evalTime);
	}
}

void Korali::Conduit::UPCXX::run()
{
	upcxx::init();
	rankId    = upcxx::rank_me();
	rankCount = upcxx::rank_n();
	nSamples = _solver->_sampleCount;
	nParameters = _solver->N;

	// Creating sample array in global shared memory
	if(rankId == 0)for (int i=1; i < rankCount;i++)actives.push(i);
	if (rankId == 0) printf("Processing %ld Samples (24 initially available), each with %ld Parameter(s)...\n", nSamples, nParameters);
	if (rankId == 0) sampleArrayPointer  = upcxx::new_array<double>(nSamples*nParameters);
	upcxx::broadcast(&sampleArrayPointer,  1, 0).wait();
	//upcxx::barrier();
	if(rankId==0) _solver->runSolver();

  upcxx::barrier();
  upcxx::finalize();
}


