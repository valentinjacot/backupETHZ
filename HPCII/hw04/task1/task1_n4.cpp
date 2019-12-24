#include "model/heat2d.hpp"
#include "korali.h"

int main(int argc, char* argv[])
{

	// Loading temperature measurement data
	FILE* dataFile = fopen("data_n4.in", "r");
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

	auto problem = Korali::Problem::Posterior(heat2DSolver);
	// 4-Candle Model x4 parameters/candle 
	p.nCandles = 4;
        
	/*
	TODO : fill in all parameter information required 
	*/
	Korali::Parameter::Uniform x1("x1", 0.0, +0.5);
	Korali::Parameter::Uniform y1("y1", 0.0, +0.5);
	Korali::Parameter::Uniform w1("w1", 0.04, 0.06);
	Korali::Parameter::Uniform i1("i1", 0.4, 0.6);
	problem.addParameter(&x1);
	problem.addParameter(&y1);
	problem.addParameter(&i1);
	problem.addParameter(&w1);
	Korali::Parameter::Uniform x2("x2", 0.0, +0.5);
	Korali::Parameter::Uniform y2("y2", 0.0, +0.5);
	Korali::Parameter::Uniform w2("w2", 0.04, 0.06);
	Korali::Parameter::Uniform i2("i2", 0.4, 0.6);
	problem.addParameter(&x2);
	problem.addParameter(&y2);
	problem.addParameter(&i2);
	problem.addParameter(&w2);
	Korali::Parameter::Uniform x3("x3", 0.5, +1.0);
	Korali::Parameter::Uniform y3("y3", 0.5, +1.0);
	Korali::Parameter::Uniform w3("w3", 0.04, 0.06);
	Korali::Parameter::Uniform i3("i3", 0.4, 0.6);
	problem.addParameter(&x3);
	problem.addParameter(&y3);
	problem.addParameter(&i3);
	problem.addParameter(&w3);
	Korali::Parameter::Uniform x4("x4", 0.5, +1.0);
	Korali::Parameter::Uniform y4("y4", 0.5, +1.0);
	Korali::Parameter::Uniform w4("w4", 0.04, 0.06);
	Korali::Parameter::Uniform i4("i4", 0.4, 0.6);
	problem.addParameter(&x4);
	problem.addParameter(&y4);
	problem.addParameter(&i4);
	problem.addParameter(&w4);
	
	problem.setReferenceData(p.nPoints, p.refTemp);

  	auto solver = Korali::Solver::CMAES(&problem);

    int Ng = 2000; // max generations for CMAES
    solver.setStopMinDeltaX(1e-6);
	solver.setPopulationSize(8); // ~4+3*log(N)
	solver.setMu(4);// =: PopSize/2
	solver.setMaxGenerations(Ng);
	solver.run();
/*
	auto sampler = Korali::Solver::TMCMC(&problem);
	sampler.setPopulationSize(10000);
	sampler.setCovarianceScaling(0.2);
	sampler.run();
*/

	return 0;
}
