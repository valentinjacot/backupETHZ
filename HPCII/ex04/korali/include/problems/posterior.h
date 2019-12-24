#ifndef _KORALI_POSTERIOR_H
#define _KORALI_POSTERIOR_H

#include "problems/likelihood.h"

namespace Korali::Problem
{

class Posterior : public Korali::Problem::Likelihood
{
  public:

	Posterior(void (*modelFunction) (double*, double*), size_t seed = -1);
	double evaluateSample(double* sample);
};

} // namespace Korali


#endif // _KORALI_POSTERIOR_H
