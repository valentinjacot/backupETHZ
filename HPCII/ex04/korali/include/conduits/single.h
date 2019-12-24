#ifndef _KORALI_SINGLE_H_
#define _KORALI_SINGLE_H_

#include "conduits/base.h"

extern double* _sampleArrayPointer;
namespace Korali::Conduit
{

class Single : public Base {
  public:

	Single(Korali::Solver::Base* solver);
  void run();
	void processSample(size_t sampleId);
	double* getSampleArrayPointer()  { return _sampleArrayPointer;  }
	void checkProgress() { };
};

} // namespace Korali

#endif // _KORALI_SINGLE_H_
