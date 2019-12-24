#include "conduits/single.h"
#include "solvers/base.h"

extern Korali::Solver::Base*  _solver;
extern Korali::Problem::Base* _problem;

size_t _nSamples;
size_t _nParameters;
double* _sampleArrayPointer;

Korali::Conduit::Single::Single(Korali::Solver::Base* solver) : Base::Base(solver) {};

void Korali::Conduit::Single::run()
{
	_nSamples = _solver->_sampleCount;
	_nParameters = _solver->N;
	_sampleArrayPointer = (double*) calloc (_nSamples*_nParameters, sizeof(double));
	 _solver->runSolver();
}

void Korali::Conduit::Single::processSample(size_t sampleId)
{
	double fitness = _problem->evaluateSample(&_sampleArrayPointer[_nParameters*sampleId]);
	_solver->updateEvaluation(sampleId, fitness);
}
