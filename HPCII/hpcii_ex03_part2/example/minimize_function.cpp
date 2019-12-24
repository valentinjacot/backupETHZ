#include "korali.h"

// Computational Model
double ackley(const double a, const double b, const double c, const double x1, const double x2)
{
  double s1 = x1*x1 + x2*x2;
  double s2 = cos(c*x1) + cos(c*x2);

  return -a*exp(-b*sqrt(s1/2)) - exp(s2/2) + a + exp(1.);
}

// Objective function
double maximize_ackley(double* x)
{
	// For this example, the a, b, and c parameters that define the shape of Ackley are well-known
	double a = 20.0;
	double b = 0.2;
	double c = 2.0*M_PI;

	// The goal of this code is to find the minimum for Ackley, therefore
	// we define our objective function as the negative of the model -> -ackley(x)
	return -ackley(a, b, c, x[0], x[1]);
}

int main(int argc, char* argv[])
{
  // We want to maximize the objective function directly.
  auto problem = Korali::Problem::Direct(maximize_ackley);

  // We only want to find the global minimum within the -32:+32 2D space
  // For this, we use two uniform variables, named x1 and x2.
  Korali::Parameter::Uniform x1("x1", -32.0, +32.0);
  Korali::Parameter::Uniform x2("x2", -32.0, +32.0);
  problem.addParameter(&x1);
  problem.addParameter(&x2);

  // Use CMAES to find the maximum of -ackley(x)
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

  return 0;
}
