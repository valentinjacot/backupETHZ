#ifndef _HEAT2D_H_
#define _HEAT2D_H_

#include <stdlib.h>
#include <vector>

typedef struct pointsInfoStruct {
  std::vector<double> xPos;
  std::vector<double> yPos;
  std::vector<double> inten;
  std::vector<double> width;
  std::vector<double> refTemp;
  size_t nCandles;
} pointsInfo;

typedef struct gridLevelStruct {
 size_t N; // Number of points per dimension in the grid level
 double h; // DeltaX = DeltaY, the distance between points in the discretized [0,1]x[0,1] domain
 double** f; // Right hand side (external heat sources)
 double* U; // Main grid
 double* Un; // Previous Jacobi grid
 double** Res; // Residual Grid
 double L2Norm; // L2 Norm of the residual
 double L2NormPrev; // Previous L2 Norm
 double L2NormDiff; // L2Norm Difference compared to previous step
} gridLevel;

// Helper Functions
gridLevel* generateInitialConditions(size_t N0, size_t gridCount);
void freeGrids(gridLevel* g, size_t gridCount);

// Solver functions
void mpi_applyJacobi(gridLevel* g, size_t l, size_t relaxations, int xdim, int ydim, int xcoord, int ycoord,int right,int left,int up,int down,MPI_Comm gridComm);
void applyJacobi(gridLevel* g, size_t l, size_t relaxations);
void mpi_calculateResidual(gridLevel* g, size_t l, int xdim, int ydim, int xcoord, int ycoord);
void calculateResidual(gridLevel* g, size_t l);
void applyRestriction(gridLevel* g, size_t l);
void applyProlongation(gridLevel* g, size_t l);
void mpi_calculateL2Norm(gridLevel* g, size_t l, int xdim, int ydim, int xcoord, int ycoord, double& part);
void calculateL2Norm(gridLevel* g, size_t l);
void printTimings(size_t gridCount);

double* smoothingTime;
double* residualTime;
double* restrictionTime;
double* prolongTime;
double* L2NormTime;
double totalTime;


#endif // _HEAT2D_H_