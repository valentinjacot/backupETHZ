#ifndef _KORALI_EXPONENTIAL_H_
#define _KORALI_EXPONENTIAL_H_

#include "parameters/base.h"

namespace Korali::Parameter
{

class Exponential : public Korali::Parameter::Base
{
 private:
  double _mean;

 public:
  Exponential(double mean);
  Exponential(std::string name, double mean);
  double getDensity(double x);
  double getDensityLog(double x);
  double getRandomNumber();
};

} // namespace Korali

#endif // _KORALI_EXPONENTIAL_H_
