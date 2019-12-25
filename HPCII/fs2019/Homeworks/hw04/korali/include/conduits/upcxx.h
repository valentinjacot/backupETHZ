#ifndef _KORALI_UPCXX_H_
#define _KORALI_UPCXX_H_

#include "conduits/base.h"
#include <upcxx/upcxx.hpp>

extern upcxx::global_ptr<double> sampleArrayPointer;

namespace Korali::Conduit
{

class UPCXX : public Base
{
  public:
  UPCXX(Korali::Solver::Base* solver);
  void run();
  void processSample(size_t sampleId);
  double* getSampleArrayPointer()  { return sampleArrayPointer.local();  }
  void checkProgress() { upcxx::progress(); }
};

} // namespace Korali

#endif // _KORALI_UPCXX_H_
