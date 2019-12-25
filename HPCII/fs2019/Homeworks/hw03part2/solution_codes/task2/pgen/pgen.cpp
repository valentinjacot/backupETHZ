/**********************************************************************/
// A now optimized Multigrid Solver for the Heat Equation             //
// Course Material for HPCSE-II, Spring 2019, ETH Zurich              //
// Authors: Sergio Martin, Georgios Arampatzis                        //
// License: Use if you like, but give us credit.                      //
/**********************************************************************/

#include <stdio.h>
#include <math.h>
#include <limits>
#include "pgen.hpp"
#include "string.h"

int main(int argc, char* argv[])
{
  double tolerance = 1e-8; // L2 Difference Tolerance before reaching convergence.
  size_t N0 = 10; // 2^N0 + 1 elements per side

	// Multigrid parameters -- Find the best configuration!
	int gridCount       = 9;     // Number of Multigrid levels to use
	int downRelaxations = 5; // Number of Relaxations before restriction
	int upRelaxations   = 3;   // Number of Relaxations after prolongation

	gridLevel* g = generateInitialConditions(N0, gridCount);

	int iteration=0;
	while (g[0].L2NormDiff > tolerance)  // Multigrid solver start
	{
		applyGaussSeidel(g, 0, downRelaxations); // Relaxing the finest grid first
		calculateResidual(g, 0); // Calculating Initial Residual

		for (int grid = 1; grid < gridCount; grid++) // Going down the V-Cycle
		{
			applyRestriction(g, grid); // Restricting the residual to the coarser grid's solution vector (f)
			applyGaussSeidel(g, grid, downRelaxations); // Smoothing coarser level
			calculateResidual(g, grid); // Calculating Coarse Grid Residual
		}

		for (int grid = gridCount-1; grid > 0; grid--) // Going up the V-Cycle
		{
			applyProlongation(g, grid); // Prolonging solution for coarser level up to finer level
			applyGaussSeidel(g, grid, upRelaxations); // Smoothing finer level
		}

		calculateL2Norm(g, 0); // Calculating Residual L2 Norm
	}  // Multigrid solver end


  std::time_t now_c = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now() - std::chrono::nanoseconds(0));
	unsigned int seed  = std::chrono::nanoseconds(now_c).count();
  gsl_rng_env_setup();
  auto range = gsl_rng_alloc (gsl_rng_default);
  gsl_rng_set(range, seed);

  size_t nPoints =  50;
  double sigma = 1;
  char* outputName = "../data.in";
  FILE *fp = fopen(outputName, "w");
  fprintf(fp, "%ld\n", nPoints);
	// Saving the value of temperatures at specified points
	double h = 1.0/(g[0].N-1);
	for(int i = 0; i < nPoints; i++)
	{
		double xPos = gsl_ran_flat(range, 0.0, 1.0);
		double yPos = gsl_ran_flat(range, 0.0, 1.0);
		int k = ceil(xPos/h);
		int l = ceil(yPos/h);
    double res   = g[0].U[k][l] + gsl_ran_gaussian(range,sigma);
		double value = GSL_MAX_DBL( res , 0. ) ;
	  fprintf(fp, "%3.20f %3.20f %3.20f\n", xPos, yPos, value);
	}

  freeGrids(g, gridCount);
}

void applyGaussSeidel(gridLevel* g, int l, int relaxations)
{
 double h2 = g[l].h*g[l].h;
 for (int r = 0; r < relaxations; r++)
 {
   for (int j = 1; j < g[l].N - 1; j++) // Gauss-Seidel Iteration -- Credit: Claudio Cannizzaro
   #pragma ivdep
   #pragma vector aligned
    for (int i = 1; i < g[l].N - 1; i++)
     g[l].U[i][j] = ((g[l].U[i - 1][j] + g[l].U[i + 1][j] + g[l].U[i][j - 1] + g[l].U[i][j + 1]) +  g[l].f[i][j] * h2 ) * 0.25;
 }
}

void calculateResidual(gridLevel* g, int l)
{
	double h2 = 1.0 / pow(g[l].h,2);

 #pragma vector aligned
 for (int i = 1; i < g[l].N-1; i++)
	#pragma ivdep
	for (int j = 1; j < g[l].N-1; j++)
	g[l].Res[i][j] = g[l].f[i][j] + (g[l].U[i-1][j] + g[l].U[i+1][j] - 4*g[l].U[i][j] + g[l].U[i][j-1] + g[l].U[i][j+1]) * h2;
}

void calculateL2Norm(gridLevel* g, int l)
{
  double tmp = 0.0;

	for (int i = 0; i < g[l].N; i++)
	 #pragma ivdep
	 #pragma vector aligned
	 for (int j = 0; j < g[l].N; j++)
		 g[l].Res[i][j] = g[l].Res[i][j]*g[l].Res[i][j];

	for (int i = 0; i < g[l].N; i++)
	 #pragma ivdep
	 #pragma vector aligned
	 for (int j = 0; j < g[l].N; j++)
		 tmp += g[l].Res[i][j];

	g[l].L2Norm = sqrt(tmp);
	g[l].L2NormDiff = fabs(g[l].L2NormPrev - g[l].L2Norm);
	g[l].L2NormPrev = g[l].L2Norm;
}

void applyRestriction(gridLevel* g, int l)
{
	for (int i = 1; i < g[l].N-1; i++)
   #pragma ivdep
   #pragma vector aligned
	 for (int j = 1; j < g[l].N-1; j++)
				 g[l].f[i][j] = ( 1.0*( g[l-1].Res[2*i-1][2*j-1] + g[l-1].Res[2*i-1][2*j+1] + g[l-1].Res[2*i+1][2*j-1]   + g[l-1].Res[2*i+1][2*j+1] )   +
													2.0*( g[l-1].Res[2*i-1][2*j]   + g[l-1].Res[2*i][2*j-1]   + g[l-1].Res[2*i+1][2*j]     + g[l-1].Res[2*i][2*j+1] ) +
													4.0*( g[l-1].Res[2*i][2*j] ) ) * 0.0625;

	for (int i = 0; i < g[l].N; i++)
   #pragma ivdep
   #pragma vector aligned
	 for (int j = 0; j < g[l].N; j++) // Resetting U vector for the coarser level before smoothing -- Find out if this is really necessary.
		g[l].U[i][j] = 0;
}

void applyProlongation(gridLevel* g, int l)
{
	for (int i = 1; i < g[l].N-1; i++)
   #pragma ivdep
	 for (int j = 1; j < g[l].N-1; j++)
			g[l-1].U[2*i][2*j] += g[l].U[i][j];

  #pragma vector aligned
	for (int i = 1; i < g[l].N; i++)
   #pragma ivdep
	 for (int j = 1; j < g[l].N-1; j++)
			g[l-1].U[2*i-1][2*j] += ( g[l].U[i-1][j] + g[l].U[i][j] ) *0.5;

  #pragma vector aligned
	for (int i = 1; i < g[l].N-1; i++)
   #pragma ivdep
	 for (int j = 1; j < g[l].N; j++)
			g[l-1].U[2*i][2*j-1] += ( g[l].U[i][j-1] + g[l].U[i][j] ) *0.5;

  #pragma vector aligned
	for (int i = 1; i < g[l].N; i++)
   #pragma ivdep
	 for (int j = 1; j < g[l].N; j++)
			g[l-1].U[2*i-1][2*j-1] += ( g[l].U[i-1][j-1] + g[l].U[i-1][j] + g[l].U[i][j-1] + g[l].U[i][j] ) *0.25;
}

gridLevel* generateInitialConditions(size_t N0, int gridCount)
{
	// Problem Parameters
	gridLevel* g = (gridLevel*) _mm_malloc(sizeof(gridLevel) * gridCount, 16);
	for (int i = 0; i < gridCount; i++)
	{
		g[i].N = pow(2, N0-i) + 1;
		g[i].h = 1.0/(g[i].N-1);

		g[i].U   = (double**) _mm_malloc(sizeof(double*) * g[i].N, 16); for (int j = 0; j < g[i].N ; j++)	g[i].U[j]   = (double*) _mm_malloc(sizeof(double) * g[i].N, 16);
		g[i].Res = (double**) _mm_malloc(sizeof(double*) * g[i].N, 16); for (int j = 0; j < g[i].N ; j++)	g[i].Res[j] = (double*) _mm_malloc(sizeof(double) * g[i].N, 16);
		g[i].f   = (double**) _mm_malloc(sizeof(double*) * g[i].N, 16); for (int j = 0; j < g[i].N ; j++)	g[i].f[j]   = (double*) _mm_malloc(sizeof(double) * g[i].N, 16);

		g[i].L2Norm = 0.0;
		g[i].L2NormPrev = std::numeric_limits<double>::max();
		g[i].L2NormDiff = std::numeric_limits<double>::max();
	}

	// Initial Guess
	for (int i = 0; i < g[0].N; i++) for (int j = 0; j < g[0].N; j++) g[0].U[i][j] = 1.0;

	// Boundary Conditions
	for (int i = 0; i < g[0].N; i++) g[0].U[0][i]        = 0.0;
	for (int i = 0; i < g[0].N; i++) g[0].U[g[0].N-1][i] = 0.0;
	for (int i = 0; i < g[0].N; i++) g[0].U[i][0]        = 0.0;
	for (int i = 0; i < g[0].N; i++) g[0].U[i][g[0].N-1] = 0.0;

	// F
	for (int i = 0; i < g[0].N; i++)
	for (int j = 0; j < g[0].N; j++)
	{
		double c1, c2, c3, c4;
		double h = 1.0/(g[0].N-1);
		double x = i*h;
		double y = j*h;

		g[0].f[i][j] = 0.0;

		// Heat Source: Candle 1
		c1 = 0.35; c1 *= 100000;// intensity
		c2 = 0.05;  c2 *= 0.01;// Width
		c3 = 0.2; // x0
		c4 = 0.2; // y0
		g[0].f[i][j] += c1*exp(-(pow(c4 - y, 2) + pow(c3 - x, 2)) / c2);

		// Heat Source: Candle 2
		c1 = 0.35; c1 *= 100000;// intensity
		c2 = 0.05;  c2 *= 0.01;// Width
		c3 = 0.7; // x0
		c4 = 0.7; // y0
		g[0].f[i][j] += c1*exp(-(pow(c4 - y, 2) + pow(c3 - x, 2)) / c2);

		// Heat Source: Candle 3
		c1 = 0.35; c1 *= 100000;// intensity
		c2 = 0.05;  c2 *= 0.01;// Width
		c3 = 0.8; // x0
		c4 = 0.8; // y0
		g[0].f[i][j] += c1*exp(-(pow(c4 - y, 2) + pow(c3 - x, 2)) / c2);
	}

	return g;
}

void freeGrids(gridLevel* g, int gridCount)
{
	for (int i = 0; i < gridCount; i++)
	{
		for (int j = 0; j < g[i].N ; j++) _mm_free(g[i].U[j]);
		for (int j = 0; j < g[i].N ; j++) _mm_free(g[i].f[j]);
		for (int j = 0; j < g[i].N ; j++) _mm_free(g[i].Res[j]);
		_mm_free(g[i].U);
		_mm_free(g[i].f);
		_mm_free(g[i].Res);
	}
	_mm_free(g);
}
