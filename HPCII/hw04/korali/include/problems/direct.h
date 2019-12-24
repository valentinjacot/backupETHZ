#ifndef _KORALI_DIRECT_H_
#define _KORALI_DIRECT_H_

#include "problems/base.h"

namespace Korali::Problem
{

class Direct : public Korali::Problem::Base
{
  public:
	double (*_modelFunction) (double*);
	Direct(double (*modelFunction) (double*), size_t seed = -1);
	double evaluateSample(double* sample);
};

} // namespace Korali


#endif // _KORALI_DIRECT_H_
