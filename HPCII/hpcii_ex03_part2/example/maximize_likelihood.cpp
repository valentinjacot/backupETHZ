#include "korali.h"

// Computational Model
double ackley(const double a, const double b, const double c, const double x1, const double x2)
{
  double s1 = x1*x1 + x2*x2;
  double s2 = cos(c*x1) + cos(c*x2);

  return -a*exp(-b*sqrt(s1/2)) - exp(s2/2) + a + exp(1.);
}

// We store the value of the provided points in data.in as global variables, for convenience.
size_t  nPoints;
double* x1Pos;
double* x2Pos;
double* fxValues;
void likelihood_ackley(double* x, double* fx)
{
	// The goal of this code is to find the values of a, b, c that best reflect
	// the provided data points, so these are our parameters.
	double a = x[0];
	double b = x[1];
	double c = x[2];

	// Filling fx for Korali to calculate likelihood
	// No need to use the negative because this is not about maximizing or minimizing
	// the model, but the likelihood. Korali will take care of that.
	for (int i = 0; i < nPoints; i++)
	 fx[i] = ackley(a, b, c, x1Pos[i], x2Pos[i]);
}

int main(int argc, char* argv[])
{
	// Loading reference data: position and values of the ackley function
	// at several points, given a, b, c parameters that we don't currently know

	FILE* dataFile = fopen("data.in", "r");

	fscanf(dataFile, "%lu", &nPoints);
	x1Pos     = (double*) calloc (sizeof(double), nPoints);
	x2Pos     = (double*) calloc (sizeof(double), nPoints);
	fxValues  = (double*) calloc (sizeof(double), nPoints);

	for (int i = 0; i < nPoints; i++)
	{
		fscanf(dataFile, "%le ", &x1Pos[i]);
		fscanf(dataFile, "%le ", &x2Pos[i]);
		fscanf(dataFile, "%le ", &fxValues[i]);
	}

  // We want to maximize the likelihood of the parameters given the data.
  auto problem = Korali::Problem::Likelihood(likelihood_ackley);

  // We only want to find the values of a,b,c. We do not have any prior knowledge
  // except from their lower/upper bounds, so we will use a uniform distribution.
  Korali::Parameter::Uniform a("a", 0.0, +50.0);
  Korali::Parameter::Uniform b("b", 0.0, +1.0);
  Korali::Parameter::Uniform c("c", +0.0*M_PI, +2.0*M_PI);
  problem.addParameter(&a);
  problem.addParameter(&b);
  problem.addParameter(&c);

  // Very important: dont forget to give the reference data to Korali!
  problem.setReferenceData(nPoints, fxValues);

  // Use CMAES to find the maximum of -ackley(x)
  auto maximizer = Korali::Solver::CMAES(&problem);

  // CMAES-specific configuration.
  // StopMinDeltaX defines how close we want the result to be
  // from the actual global minimum. The smaller this value is, 
  // the more generations it may take to find it.
  maximizer.setStopMinDeltaX(1e-11);

  // Population size defines how many samples per generations we want to run
  // For CMAES, a small number of samples (64-256) will do the trick.
  maximizer.setPopulationSize(64);

  // For this problem, we may need to run more generations
  maximizer.setMaxGenerations(1000);

  // Run CMAES and report the result
  maximizer.run();

  return 0;
}
