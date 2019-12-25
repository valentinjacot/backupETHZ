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

  Korali::Parameter::Uniform psx("PosX",  0.0, 5.0);
  Korali::Parameter::Uniform psy("PosY",  0.0, 5.0);

  problem.addParameter(&psx);
  problem.addParameter(&psy);

  auto solver = Korali::Solver::TMCMC(&problem);

	solver.setPopulationSize(10000);
//	solver.setCovarianceScaling(0.08);
//	solver.setToleranceCOV(0.6);
	solver.run();

	return 0;
}
