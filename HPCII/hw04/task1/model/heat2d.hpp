#ifndef _HEAT2D_H_
#define _HEAT2D_H_

#include <stdlib.h>

typedef struct pointsInfoStruct {
	  size_t nCandles;
    size_t nPoints;
    double* xPos;
    double* yPos;
    double* inten;
    double* width;
    double* refTemp;
} pointsInfo;

typedef struct gridLevelStruct {
	size_t N; // Number of points per dimension in the grid level
	double h; // DeltaX = DeltaY, the distance between points in the discretized [0,1]x[0,1] domain
	double** f; // Right hand side (external heat sources)
	double** U; // Main grid
	double** Res; // Residual Grid
	double L2Norm; // L2 Norm of the residual
  double L2NormPrev; // Previous L2 Norm
  double L2NormDiff; // L2Norm Difference compared to previous step
} gridLevel;

// Main solver
void heat2DSolver(double* pars, double* output);

// Helper Functions
gridLevel* generateInitialConditions(size_t N0, int gridCount, double* pars);
void freeGrids(gridLevel* g, int gridCount);

// Solver functions
void applyGaussSeidel(gridLevel* g, int l, int relaxations);
void calculateResidual(gridLevel* g, int l);
void applyRestriction(gridLevel* g, int l);
void applyProlongation(gridLevel* g, int l);
void calculateL2Norm(gridLevel* g, int l);

extern pointsInfo p;

#endif // _HEAT2D_H_
