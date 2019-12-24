#ifndef _KORALI_GAMMA_H_
#define _KORALI_GAMMA_H_

#include "parameters/base.h"

namespace Korali::Parameter
{

class Gamma : public Korali::Parameter::Base
{
 private:
  double _shape;
  double _rate;

 public:
  Gamma(double shape, double rate);
  Gamma(std::string name, double shape, double rate);
  double getDensity(double x);
  double getDensityLog(double x);
  double getRandomNumber();
};

} // namespace Korali

#endif // _KORALI_GAMMA_H_
