#include "model/grass.hpp"
#include "korali.h"

// Grass Height at different spots, as measured by Herr Kueheli.
size_t  nSpots;
double* xPos;
double* yPos;
double* heights;
void maximize_grassHeight(double* x, double* fx)
{
	double MM = x[0];
	double PH = x[1];
	for (int i = 0; i < nSpots; i++)
	 fx[i] = getGrassHeight(xPos[i], yPos[i], PH, MM);
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
	auto problem = Korali::Problem::Posterior(maximize_grassHeight);
	
	Korali::Parameter::Gaussian MM("MM", 90.0, 20.0);
	MM.setBounds(0.0, 200.0);
	
	Korali::Parameter::Uniform PH("PH", 4.0, 9.0);
	
	problem.addParameter(&MM);
	problem.addParameter(&PH);
	
	problem.setReferenceData(nSpots, heights);

	auto maximizer = Korali::Solver::CMAES(&problem);
	maximizer.setStopMinDeltaX(1e-11);

	maximizer.setPopulationSize(64);

	maximizer.setMaxGenerations(1000);

	maximizer.run();

	return 0;

}
