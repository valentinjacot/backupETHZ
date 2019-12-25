#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits>
#include <chrono>
#include "auxiliar.hpp"

int main(int argc, char** argv)
{
	// User-defined Parameters
	Heat2DSetup s;

	for (int i = 0; i < argc; i++)
	{
			if(!strcmp(argv[i], "-p"))  s.problemNumber = atoi(argv[++i]);
			if(!strcmp(argv[i], "-s") || !strcmp(argv[i], "--save"))   { s.saveOutput = true; }
			if(!strcmp(argv[i], "-v") || !strcmp(argv[i], "--verify")) { s.checkSolution = true;}
			if(!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))   { s.printHelp(); exit(0); }
	}

  s.loadProblem();

  auto start = std::chrono::system_clock::now();
	heat2DSolver(s);
  auto end = std::chrono::system_clock::now();
	s.totalTime = std::chrono::duration<double>(end-start).count();

	double* timePerGrid = (double*) calloc (sizeof(double), s.gridCount);
	double* timePerOp = (double*) calloc (sizeof(double), 5);
	double totalSmoothingTime = 0.0;
	double totalResidualTime = 0.0;
	double totalRestrictionTime = 0.0;
	double totalProlongTime = 0.0;
	double totalL2NormTime = 0.0;

	for (int i = 0; i < s.gridCount; i++) timePerGrid[i] = s.smoothingTime[i] + s.residualTime[i] + s.restrictionTime[i] + s.prolongTime[i] + s.L2NormTime[i];
	for (int i = 0; i < s.gridCount; i++) totalSmoothingTime += s.smoothingTime[i];
	for (int i = 0; i < s.gridCount; i++) totalResidualTime += s.residualTime[i];
	for (int i = 0; i < s.gridCount; i++) totalRestrictionTime += s.restrictionTime[i];
	for (int i = 0; i < s.gridCount; i++) totalProlongTime += s.prolongTime[i];
	for (int i = 0; i < s.gridCount; i++) totalL2NormTime += s.L2NormTime[i];

	double totalMeasured = totalSmoothingTime + totalResidualTime + totalRestrictionTime + totalProlongTime + totalL2NormTime;

	printf("   Time (s)    "); for (int i = 0; i < s.gridCount; i++) printf("Grid%d   ", i);                    printf("   Total  \n");
	printf("-------------|-"); for (int i = 0; i < s.gridCount; i++) printf("--------"); printf("|---------\n");
	printf("Smoothing    | "); for (int i = 0; i < s.gridCount; i++) printf("%2.3f   ", s.smoothingTime[i]);    printf("|  %2.3f  \n", totalSmoothingTime);
	printf("Residual     | "); for (int i = 0; i < s.gridCount; i++) printf("%2.3f   ", s.residualTime[i]);     printf("|  %2.3f  \n", totalResidualTime);
	printf("Restriction  | "); for (int i = 0; i < s.gridCount; i++) printf("%2.3f   ", s.restrictionTime[i]);  printf("|  %2.3f  \n", totalRestrictionTime);
	printf("Prolongation | "); for (int i = 0; i < s.gridCount; i++) printf("%2.3f   ", s.prolongTime[i]);      printf("|  %2.3f  \n", totalProlongTime);
	printf("L2Norm       | "); for (int i = 0; i < s.gridCount; i++) printf("%2.3f   ", s.L2NormTime[i]);       printf("|  %2.3f  \n", totalL2NormTime);
	printf("-------------|-"); for (int i = 0; i < s.gridCount; i++) printf("--------"); printf("|---------\n");
	printf("Total        | "); for (int i = 0; i < s.gridCount; i++) printf("%2.3f   ", timePerGrid[i]); printf("|  %2.3f  \n", totalMeasured);
	printf("-------------|-"); for (int i = 0; i < s.gridCount; i++) printf("--------"); printf("|---------\n");
	printf("\n");
	printf("Fine grid elements: %lu x %lu (n = %lu)\n", s.N, s.N, s.N0);
	printf("V-Cycle Iterations: %d\n", s.iteration);
	printf("Final L2 Residual : %e\n", s.L2Norm);
	printf("Convergence Rate  : %e\n", s.L2NormDiff);
	printf("Running Time      : %.3fs\n", s.totalTime);

  if (s.saveOutput) s.outputSolution();
  if (s.checkSolution) s.verifySolution();

	return 0;
}

Heat2DSetup::Heat2DSetup()
{
	iteration = 0;
	problemNumber = 1;

  saveOutput = false; // Write output to file flag
  checkSolution = false; // Verify solution

  L2Norm = 0.0;
  L2NormPrev = std::numeric_limits<double>::max();
  L2NormDiff = std::numeric_limits<double>::max();
}

void Heat2DSetup::setGridCount(int count)
{
  gridCount = count;

  smoothingTime   = (double*) calloc (sizeof(double), gridCount);
  residualTime    = (double*) calloc (sizeof(double), gridCount);
  restrictionTime = (double*) calloc (sizeof(double), gridCount);
  prolongTime     = (double*) calloc (sizeof(double), gridCount);
  L2NormTime      = (double*) calloc (sizeof(double), gridCount);

  for (int i = 0; i < gridCount; i++) smoothingTime[i] = 0.0;
  for (int i = 0; i < gridCount; i++) residualTime[i] = 0.0;
  for (int i = 0; i < gridCount; i++) restrictionTime[i] = 0.0;
  for (int i = 0; i < gridCount; i++) prolongTime[i] = 0.0;
  for (int i = 0; i < gridCount; i++) L2NormTime[i] = 0.0;
}

void Heat2DSetup::applyJacobi_(GridLevel* g, int l, int relaxations)
{
	auto t0 = std::chrono::system_clock::now();
	applyJacobi(g, l, relaxations);
	auto t1 = std::chrono::system_clock::now();
  smoothingTime[l] += std::chrono::duration<double>(t1-t0).count();
}

void Heat2DSetup::applyProlongation_(GridLevel* g, int l)
{
  auto t0 = std::chrono::system_clock::now();
  applyProlongation(g, l);
	auto t1 = std::chrono::system_clock::now();
	prolongTime[l-1] += std::chrono::duration<double>(t1-t0).count();
}

void Heat2DSetup::applyRestriction_(GridLevel* g, int l)
{
  auto t0 = std::chrono::system_clock::now();
  applyRestriction(g, l);
	auto t1 = std::chrono::system_clock::now();
	restrictionTime[l-1] += std::chrono::duration<double>(t1-t0).count();
}

void Heat2DSetup::calculateResidual_(GridLevel* g, int l)
{
  auto t0 = std::chrono::system_clock::now();
  calculateResidual(g, l);
	auto t1 = std::chrono::system_clock::now();
	residualTime[l] += std::chrono::duration<double>(t1-t0).count();
}

void Heat2DSetup::calculateL2Norm_(GridLevel* g, int l)
{
  auto t0 = std::chrono::system_clock::now();
	L2Norm = calculateL2Norm(g, l);
	auto t1 = std::chrono::system_clock::now();
	L2NormTime[l] += std::chrono::duration<double>(t1-t0).count();

	L2NormDiff = abs(L2NormPrev - L2Norm);
  L2NormPrev = L2Norm;
	iteration++;
}


void Heat2DSetup::printHelp()
{
    printf("/**********************************************************************/\n");
    printf("// A still unoptimized Multigrid Solver for the Heat Equation         //\n");
    printf("// Course Material for HPCSE-II, Spring 2019, ETH Zurich              //\n");
    printf("// Authors: Sergio Martin, Georgios Arampatzis                        //\n");
    printf("// License: Use if you like, but give us credit.                      //\n");
    printf("/**********************************************************************/\n");
    printf("\n");
    printf("Usage:\n");
    printf("  -p N - Loads a problem from problemN.in\n");
    printf("  -v or --verify - verifies solution with that of problemN.sol\n");
    printf("  -s or --save - Saves computed solution problemN.cpt\n");
    printf("  -h  -  Prints this help info.\n");
}

void Heat2DSetup::printGrid(double** g, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%.4f  ", g[i][j]);
        }
        printf("\n");
    }
}

void Heat2DSetup::loadProblem()
{
    // Read solution from file
    FILE *fid;
    char pfile[50];
    sprintf(pfile, "auxiliar/problem%d.in", problemNumber);

    printf("Running problem from file %s... \n", pfile);

    fid = fopen(pfile, "r");

    // N0
    fscanf(fid, "%lu\n", &N0);

    // N
    N = pow(2, N0) + 1;

    // Tolerance
    fscanf(fid, "%le\n", &tolerance);

    // U
    U = (double*) calloc (sizeof(double), N*N);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            fscanf(fid, "%le ", &U[i*N + j]);
        }
        fscanf(fid, "\n");
    }

    // f
    f = (double*) calloc (sizeof(double), N*N);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            fscanf(fid, "%le ", &f[i*N + j]);
        }
        fscanf(fid, "\n");
    }

    fclose(fid);
}


void Heat2DSetup::outputSolution()
{
		FILE *fid;
		char pfile[50];
		sprintf(pfile, "problem%d.own", problemNumber);

    printf("Writing computed solution to file %s...\n", pfile);

    fid = fopen(pfile, "w");

    // U
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            fprintf(fid, "%22.15e ", U[i*N + j]);
        }
        fprintf(fid, "\n");
    }

    fclose(fid);
}


void Heat2DSetup::verifySolution()
{
  // Read solution from file
  FILE *fid;
  char pfile[50];
  sprintf(pfile, "auxiliar/problem%d.sol", problemNumber);

  printf("Loading actual solution from file %s... \n", pfile);

  fid = fopen(pfile, "r");

  // Solution
  sol = (double*) calloc (sizeof(double), N*N);
  for (int i = 0; i < N; i++)
  {
      for (int j = 0; j < N; j++)
      {
          fscanf(fid, "%le ", &sol[i*N + j]);
      }
      fscanf(fid, "\n");
  }
  fclose(fid);

	double L2_err, Linf_err=0;
	for (int i = 0; i < N; i++)
	{
			for (int j = 0; j < N; j++)
			{
					Linf_err = fmax(Linf_err, abs(U[i*N+j] - sol[i*N+j]));
					L2_err  += pow(U[i*N+j] - sol[i*N+j], 2);
			}
	}
	L2_err = pow(L2_err, 0.5);

	double err_tol = 1e-3;
	if (L2_err > err_tol) { printf("Verification Failed!\n"); printf(" L2_err   = %e > %e\n", L2_err,err_tol); exit(-1); }
	if (Linf_err > err_tol) { printf("Verification Failed!\n"); printf(" Linf_err = %e > %e\n", Linf_err,err_tol); exit(-1); }
  printf("Verification Passed.\n");

}
