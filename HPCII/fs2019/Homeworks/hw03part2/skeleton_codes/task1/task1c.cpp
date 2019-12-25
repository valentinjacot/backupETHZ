#include "model/grass.hpp"
#include "korali.h"

#define MM 80.0
#define PH 6.0
double maximize_grassHeight(double* x)
{
	return getGrassHeight(x[0], x[1], PH, MM);
}

int main(int argc, char* argv[])
{

  // We want to maximize the objective function directly.
  auto problem = Korali::Problem::Direct(maximize_grassHeight);

  // We only want to find the global minimum within the 0:5 2D space
  // For this, we use two uniform variables, named x1 and x2.
  Korali::Parameter::Uniform x1("x1", 0.0, +5.0);
  Korali::Parameter::Uniform x2("x2", 0.0, +5.0);
  problem.addParameter(&x1);
  problem.addParameter(&x2);

  // Use TMCMC to find the maximum 
  auto sampler = Korali::Solver::TMCMC(&problem);
	
	
  sampler.setPopulationSize(10000);
  
  sampler.setCovarianceScaling(0.2);


  // Run CMAES and report the result
  sampler.run();
	
	
	return 0;
}
