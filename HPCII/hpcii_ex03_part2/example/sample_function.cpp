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

  // Use TMCMC to sample -ackley(x)
  auto sampler = Korali::Solver::TMCMC(&problem);

  // TMCMC-specific configuration.
  // Number of samples to represent the distribution.
  // The more samples, the more precise the representation will be
  // but may take more time to run per generation, and more generations
  // to find a perfectly annealing representation.
  sampler.setPopulationSize(10000);

  // Defines the 'sensitivity' of re-sampling. That is, how much the new
  // samples within a chain will scale during evaluation. A higher value
  // is better to explore a larger space, while a lower value will be
  // more precise for small magnitude parameters.
  sampler.setCovarianceScaling(0.2);

	// Run TMCMC to produce tmcmc.txt.
  // Use plotmatrix_hist to see the result of the sampling.
  sampler.run();

  return 0;
}
