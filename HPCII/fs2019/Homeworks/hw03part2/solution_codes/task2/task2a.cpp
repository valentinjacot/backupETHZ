#include "model/heat2d.hpp"
#include "korali.h"

int main(int argc, char* argv[])
{
	// Loading temperature measurement data
    
    bool tmcmc = true;  // for model selection
    bool cmaes = false;   // for optimization (finding x,y)


	if(argc<3){
		printf("\nusage: %s Np Nsamples\n", argv[0]);
		printf("        Np = 1,2 or 3 \n");
		printf("        Nsamples > 0  \n");
		exit(1);
	}


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


	auto problem = Korali::Problem::Posterior(heat2DSolver);


	Korali::Parameter::Uniform *psx1, *psx2, *psx3;
	Korali::Parameter::Uniform *psy1, *psy2, *psy3;


	// How many candles will we simulate?
	int choice = atoi(argv[1]);

	switch( choice ){

		// 1 candle
		case 1:
						p.nCandles = 1; // 1-Candle Model - Requires 2 parameters (PosX, PosY)

						psx1 = new Korali::Parameter::Uniform("PosX1",  0.0,  1.0);
						psy1 = new Korali::Parameter::Uniform("PosY1",  0.0,  1.0);

						problem.addParameter( psx1 );
						problem.addParameter( psy1 );

						break;

		// 2 candles
		case 2:
						p.nCandles = 2; // 2-Candle Model - Requires 4 parameters (PosX, PosY)x2

						psx1 = new Korali::Parameter::Uniform("PosX1",  0.0,  0.5);
						psy1 = new Korali::Parameter::Uniform("PosY1",  0.0,  1.0);

						problem.addParameter( psx1 );
						problem.addParameter( psy1 );

						psx2 = new Korali::Parameter::Uniform("PosX2",  0.5,  1.0);
						psy2 = new Korali::Parameter::Uniform("PosY2",  0.0,  1.0);

						problem.addParameter( psx2 );
						problem.addParameter( psy2 );

						break;


	// 3 candles
		case 3:
						p.nCandles = 3; // 3-Candle Model - Requires 6 parameters (PosX, PosY)x3

						psx1 = new Korali::Parameter::Uniform("PosX1",  0.0,  0.5);
						psy1 = new Korali::Parameter::Uniform("PosY1",  0.0,  1.0);

						problem.addParameter( psx1 );
						problem.addParameter( psy1 );

						psx2 = new Korali::Parameter::Uniform("PosX2",  0.5,  1.0);
						psy2 = new Korali::Parameter::Uniform("PosY2",  0.0,  1.0);

						problem.addParameter( psx2 );
						problem.addParameter( psy2 );

						psx3 = new Korali::Parameter::Uniform("PosX3",  0.5,  1.0);
						psy3 = new Korali::Parameter::Uniform("PosY3",  0.0,  1.0);

						problem.addParameter( psx3 );
						problem.addParameter( psy3 );

						break;

		default:
						printf("\nusage: %s {1,2,3} \n\n", argv[0]);
						exit(1);

  }

  problem.setReferenceData( p.nPoints, p.refTemp );

  auto solver = Korali::Solver::TMCMC(&problem);

  int nsamples = atoi(argv[2]);
  printf("\nsample size: %d\n\n", nsamples);
  solver.setPopulationSize(nsamples);
  solver.setMaxGenerations(100);
  if (tmcmc) solver.run();
 

  int lambda = 4+floor(3*log(2*choice)); // theoretical optimal sample size (see Hansen[2016] 'The CMA Evolution Strategy: A Tutorial')
  int mu     = lambda/2;                 // dito

  printf("\nrunning cmaes with lambda %d and mu %d\n\n", lambda, mu);
  auto optimizer = Korali::Solver::CMAES(&problem);
  optimizer.setPopulationSize(lambda);
  optimizer.setMu(mu);

  if (cmaes) optimizer.run();

  return 0;
}