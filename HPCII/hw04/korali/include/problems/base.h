#ifndef _KORALI_BASEPROBLEM_H_
#define _KORALI_BASEPROBLEM_H_

#include "parameters/base.h"
#include <vector>
#include "stdlib.h"

namespace Korali::Conduit {
		class Base;
}

namespace Korali::Problem
{

class Base
{
  public:

	Base(size_t seed = 0);

	void addParameter(Korali::Parameter::Base* p);
  virtual double evaluateSample(double* sample) = 0;

  size_t _parameterCount;
  size_t _referenceDataSize;
	size_t _seed;

	Korali::Conduit::Base* _conduit;
  std::vector<Korali::Parameter::Base*> _parameters;

  bool isSampleOutsideBounds(double* sample);
  double getPriorsLogProbabilityDensity(double *x);
  double getPriorsProbabilityDensity(double *x);
  void initializeParameters();
};

} // namespace Korali


#endif // _KORALI_BASEPROBLEM_H_
