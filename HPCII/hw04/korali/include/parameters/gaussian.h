#ifndef _KORALI_GAUSSIAN_H_
#define _KORALI_GAUSSIAN_H_

#include "parameters/base.h"

namespace Korali::Parameter
{

class Gaussian : public Korali::Parameter::Base
{
 private:
  double _mean;
  double _sigma;

 public:
  Gaussian(double mean, double sigma);
  Gaussian(std::string name, double mean, double sigma);
  double getDensity(double x);
  double getDensityLog(double x);
  double getRandomNumber();
  static double logLikelihood(double sigma, int nData, double* x, double* u);
};

} // namespace Korali

#endif // _KORALI_GAUSSIAN_H_
