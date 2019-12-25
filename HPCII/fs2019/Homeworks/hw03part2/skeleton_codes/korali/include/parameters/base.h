#ifndef _KORALI_BASEPARAMETER_H_
#define _KORALI_BASEPARAMETER_H_

#include <string>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>

namespace Korali::Parameter
{

class Base
{
 public:
 Base(std::string name);
 gsl_rng* _range;

 std::string _name;
 double _lowerBound;
 double _upperBound;

 void setBounds(double lowerBound, double upperBound) { _lowerBound = lowerBound;	_upperBound = upperBound; }
 void setName(std::string name)    { _name = name; }
 void initializeDistribution(int seed);
 void checkBounds();

 virtual double getDensity(double x) = 0;
 virtual double getDensityLog(double x) = 0;
 virtual double getRandomNumber() = 0;
};

} // namespace Korali

#endif // _KORALI_BASEPARAMETER_H_
