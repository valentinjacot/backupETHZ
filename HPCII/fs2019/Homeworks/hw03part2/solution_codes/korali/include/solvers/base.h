#ifndef _BASESOLVER_H_
#define _BASESOLVER_H_

#include <stdlib.h>
#include "problems/base.h"

namespace Korali::Conduit {
    class Base;
}

namespace Korali::Solver
{

class Base {
  public:

	// Korali Runtime Variables
  size_t _maxGens;                  // Max number of Conduit Generations
  size_t _sampleCount;
	size_t  N; // Parameter Count
  bool _verbose;

  Korali::Conduit::Base* _conduit;
	Korali::Problem::Base* _problem;

	Base(Korali::Problem::Base* problem);
	void setPopulationSize(int size) { _sampleCount = size; }
	void setVerbose(bool verbose) { _verbose = verbose; }
	void setMaxGenerations(int maxGens) { _maxGens = maxGens; }

	void run();
	virtual void runSolver() = 0;
	virtual void processSample(size_t sampleId, double fitness) = 0;
};

} // namespace Korali

#endif // _BASESOLVER_H_
