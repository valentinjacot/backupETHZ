#ifndef _KORALI_BASECONDUIT_H_
#define _KORALI_BASECONDUIT_H_

#include <stdlib.h>
#include "problems/base.h"

namespace Korali::Solver {
    class Base;
}

namespace Korali::Conduit
{

class Base {
  public:

  Base(Korali::Solver::Base* solver);
  virtual void run() = 0;
  virtual void processSample(size_t sampleId) = 0;
  virtual void checkProgress() = 0;
  virtual double* getSampleArrayPointer() = 0;
};

class Conduit;

} // namespace Korali

#endif // _KORALI_BASECONDUIT_H_
