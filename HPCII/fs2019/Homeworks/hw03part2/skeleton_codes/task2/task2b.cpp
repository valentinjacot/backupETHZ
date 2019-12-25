#include "model/heat2d.hpp"
#include "korali.h"

int main(int argc, char* argv[])
{
 // Loading temperature measurement data

 FILE* dataFile = fopen("data.in", "r");
 fscanf(dataFile, "%lu", &p.nPoints);

 p.xPos    = (double*) calloc (sizeof(double), p.nPoints);
 p.yPos    = (double*) calloc (sizeof(double), p.nPoints);
 p.refTemp = (double*) calloc (sizeof(double), p.nPoints);

 for (int i = 0; i < p.nPoints; i++)
 {
  fscanf(dataFile, "%le ", &p.xPos[i]);
  fscanf(dataFile, "%le ", &p.yPos[i]);
  fscanf(dataFile, "%le ", &p.refTemp[i]);
 }

 // How many candles will we simulate?
 // p.nCandles = 1; // 1-Candle Model - Requires 2 parameters (PosX, PosY)
 // p.nCandles = 2; // 2-Candle Model - Requires 4 parameters (PosX, PosY)x2
 //  p.nCandles = 3; // 3-Candle Model - Requires 6 parameters (PosX, PosY)x3

 // Start configuring the Problem and the Korali Engine
	p.nCandles = 3;
	auto problem = Korali::Problem::Posterior(heat2DSolver);
	//auto problem = Korali::Problem::Likelihood(heat2DSolver);
	//auto problem = Korali::Problem::Direct(heat2DSolver);


	// We only want to find the values of a,b,c. We do not have any prior knowledge
	// except from their lower/upper bounds, so we will use a uniform distribution.
	Korali::Parameter::Gaussian x1("x1", 0.25, 0.05);
	Korali::Parameter::Gaussian y1("y1", 0.25, 0.05);
	problem.addParameter(&x1);
	problem.addParameter(&y1);
	Korali::Parameter::Gaussian x2("x2", 0.75, 0.05);
	Korali::Parameter::Gaussian y2("y2", 0.75, 0.05);
	problem.addParameter(&x2);
	problem.addParameter(&y2);	
	Korali::Parameter::Gaussian x3("x3", 0.75, 0.05);
	Korali::Parameter::Gaussian y3("y3", 0.75, 0.05);
	problem.addParameter(&x3);
	problem.addParameter(&y3);
	//parameter.setBounds(0,1);
	problem.setReferenceData(p.nPoints, p.refTemp);
	auto sampler = Korali::Solver::CMAES(&problem);
	sampler.setPopulationSize(128);
//	sampler.setCovarianceScaling(0.2);
	sampler.setStopMinDeltaX(1e-8);

	sampler.run();
return 0;
}
