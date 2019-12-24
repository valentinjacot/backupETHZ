#include "model/grass.hpp"
#include "korali.h"

#define MM 80.0
#define PH 6.0

double grassSolver(double* pars)
{
	return getGrassHeight(pars[0], pars[1], PH, MM);
}

int main(int argc, char* argv[])
{
  auto problem = Korali::Problem::Direct(grassSolver);

  Korali::Parameter::Uniform psx("PosX",      0.0, 5.0);
  Korali::Parameter::Uniform psy("PosY",      0.0, 5.0);

  problem.addParameter(&psx);
  problem.addParameter(&psy);

  auto solver = Korali::Solver::CMAES(&problem);

	solver.setStopMinDeltaX(1e-4);
	solver.setPopulationSize(32);
	solver.setMaxGenerations(1000);
	solver.run();

	return 0;
}
