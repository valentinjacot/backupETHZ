#include "model/grass.hpp"
#include "korali.h"

#define MM 80.0
#define PH 6.0
// Objective function
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

  // Use CMAES to find the maximum
  auto maximizer = Korali::Solver::CMAES(&problem);

  // CMAES-specific configuration.
  // StopMinDeltaX defines how close we want the result to be
  // from the actual global minimum. The smaller this value is, 
  // the more generations it may take to find it.
  maximizer.setStopMinDeltaX(1e-11);

  // Population size defines how many samples per generations we want to run
  // For CMAES, a small number of samples (64-256) will do the trick.
  maximizer.setPopulationSize(128);

  // Run CMAES and report the result
  maximizer.run();
	
	
//	return 0;scp jacotdev@euler.ethz.ch:tmcmc.txt .

}
