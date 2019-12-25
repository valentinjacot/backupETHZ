#include "model/grass.hpp"
#include "korali.h"

// Grass Height at different spots, as measured by Herr Kueheli.
size_t  nSpots;
double* xPos;
double* yPos;
double* heights;

void grassSolver(double* pars, double* output)
{
	for (int i = 0; i < nSpots; i++) output[i] = getGrassHeight(xPos[i], yPos[i], pars[0], pars[1]);
}

int main(int argc, char* argv[])
{
	// Loading grass height data

	FILE* dataFile = fopen("grass.in", "r");

	fscanf(dataFile, "%lu", &nSpots);
	xPos     = (double*) calloc (sizeof(double), nSpots);
	yPos     = (double*) calloc (sizeof(double), nSpots);
	heights  = (double*) calloc (sizeof(double), nSpots);

	for (int i = 0; i < nSpots; i++)
	{
		fscanf(dataFile, "%le ", &xPos[i]);
		fscanf(dataFile, "%le ", &yPos[i]);
		fscanf(dataFile, "%le ", &heights[i]);
	}

  auto problem = Korali::Problem::Likelihood(grassSolver);

  Korali::Parameter::Uniform  ph("pH",   4.0, 9.0);
  Korali::Parameter::Gaussian mm("mm",  90.0, 20.0);

  ph.setBounds(4.0, 9.0);
  mm.setBounds(20.0, 200.0);

  problem.addParameter(&ph);
  problem.addParameter(&mm);

  problem.setReferenceData(nSpots, heights);

  auto solver = Korali::Solver::TMCMC(&problem);

	solver.setPopulationSize(10000);
	solver.setMaxGenerations(1000);
	solver.run();

	return 0;
}
