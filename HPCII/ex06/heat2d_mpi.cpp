/**********************************************************************/
// A now optimized Multigrid Solver for the Heat Equation             //
// Course Material for HPCSE-II, Spring 2019, ETH Zurich              //
// Authors: Sergio Martin, Georgios Arampatzis                        //
// License: Use if you like, but give us credit.                      //
/**********************************************************************/

#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <limits>
#include "heat2d_mpi.hpp"
#include "string.h"
#include <chrono>

pointsInfo __p;

int main(int argc, char* argv[])
{
 
 MPI_Init(&argc,&argv);
 int rank, rankCount;
 MPI_Comm_rank(MPI_COMM_WORLD,&rank);
 MPI_Comm_size(MPI_COMM_WORLD,&rankCount);

 double tolerance = 1e-0; // L2 Difference Tolerance before reaching convergence.
 size_t N0 = 10; //10 // 2^N0 + 1 elements per side

 // Multigrid parameters -- Find the best configuration!
 size_t gridCount       = 1;     // Number of Multigrid levels to use
 size_t downRelaxations = 3; // Number of Relaxations before restriction
 //size_t upRelaxations   = 3;   // Number of Relaxations after prolongation

 gridLevel* g = generateInitialConditions(N0, gridCount);


 int nums[2] = {0,0};
 int periodic[2] = {false, false};
 MPI_Dims_create(rankCount, 2, nums); // split the nodes automatically

 MPI_Comm cart_comm; // now everyone creates a a cartesian topology
 MPI_Cart_create(MPI_COMM_WORLD, 2, nums, periodic, true, &cart_comm);

 int coords[2];
 MPI_Cart_coords(cart_comm, rank, 2, coords);

 int left, right, bottom, top;
 MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
 MPI_Cart_shift(cart_comm, 1, 1, &bottom, &top);
 
 printf("Grid: (%d, %d)\n", nums[0], nums[1]);
 printf("Coord: (%d, %d)\n", coords[0], coords[1]);
 printf("(%d) Left: %d, Right: %d\n", rank, left, right);
 printf("(%d) Top: %d, Bottom: %d\n", rank, top, bottom);

 auto startTime = std::chrono::system_clock::now();
 while (g[0].L2NormDiff > tolerance)  // Multigrid solver start
 {

  mpi_applyJacobi(g, 0, downRelaxations, nums[0], nums[1], coords[0], coords[1], left, right, top, bottom, cart_comm);
  mpi_calculateResidual(g, 0, nums[0], nums[1], coords[0], coords[1]);


  /* TODO
  for (size_t grid = 1; grid < gridCount; grid++) // Going down the V-Cycle
  {
   applyRestriction(g, grid); // Restricting the residual to the coarser grid's solution vector (f)
   applyJacobi(g, grid, downRelaxations); // Smoothing coarser level
   calculateResidual(g, grid); // Calculating Coarse Grid Residual
  }

  for (size_t grid = gridCount-1; grid > 0; grid--) // Going up the V-Cycle
  {
   applyProlongation(g, grid); // Prolonging solution for coarser level up to finer level
   applyJacobi(g, grid, upRelaxations); // Smoothing finer level
  }
  */
  
  
  double partres = 0.0;
  double totL2   = 0.0;
  mpi_calculateL2Norm(g, 0, nums[0], nums[1], coords[0], coords[1], partres);

  MPI_Reduce(&partres, &totL2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&totL2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  g[0].L2Norm     = sqrt(totL2);
  g[0].L2NormDiff = fabs(g[0].L2NormPrev - g[0].L2Norm);
  g[0].L2NormPrev = g[0].L2Norm;

 }  // Multigrid solver end

 auto endTime = std::chrono::system_clock::now();
 totalTime = std::chrono::duration<double>(endTime-startTime).count();
 
 if (rank == 0) printTimings(gridCount);
 if (rank == 0) printf("L2Norm: %.4f\n", g[0].L2Norm);
 
 freeGrids(g, gridCount);
 
 
 MPI_Comm_free(&cart_comm);
 MPI_Finalize();
 return 0;
}

void mpi_applyJacobi(gridLevel* g, size_t l, size_t relaxations, int xdim, int ydim, int xcoord, int ycoord, int left, int right, int top, int bottom, MPI_Comm comm)
{

 size_t N = g[l].N;
 auto t0 = std::chrono::system_clock::now();

 double h1 = 0.25;
 double h2 = g[l].h*g[l].h;

 size_t iFrom = ceil(( ( (double) N-2) / ydim ) * ycoord ) + 1;
 size_t iTo   = ceil(( ( (double) N-2) / ydim ) * (ycoord+1)) + 1;
 size_t jFrom = ceil(( ( (double) N-2) / xdim ) * xcoord ) + 1;
 size_t jTo   = ceil(( ( (double) N-2) / xdim ) * (xcoord+1))  + 1;

 // types
 MPI_Datatype faceXType, faceYType;
 MPI_Type_vector(N/ydim, 1,  N, MPI_DOUBLE, &faceXType); //count, blocklength, stride
 MPI_Type_vector(1, N/xdim, N/xdim, MPI_DOUBLE, &faceYType); // contiguous
 MPI_Type_commit(&faceXType);
 MPI_Type_commit(&faceYType);

 // communicate boundaries
 size_t xstart, xstartRecv, ystart, ystartRecv;  

 for (size_t r = 0; r < relaxations; r++)
 {
  
  double* tmp = g[l].Un; g[l].Un = g[l].U; g[l].U = tmp;
  for (size_t i = iFrom; i < iTo; i++) for (size_t j = jFrom; j < jTo; j++) g[l].U[i * N + j] = (g[l].Un[ (i-1) * N + j] + g[l].Un[ (i+1) * N + j] + g[l].Un[i * N + j-1] + g[l].Un[i * N + j+1] + g[l].f[i][j]*h2)*h1;

  MPI_Status status;
  
  // send bottom from upper to lower (contiguous)
  xstart     = jFrom;
  xstartRecv = jFrom;
  ystart     = iFrom; 
  ystartRecv = iTo;

  if (bottom >= 0) MPI_Send( &g[l].U[ ystart * N  + xstart ], 1,  faceYType, bottom, 123, comm );
  if (top    >= 0) MPI_Recv( &g[l].U[ ystartRecv * N  + xstartRecv ], 1,  faceYType, top, 123, comm, &status );
  

  // send ceiling from lower to upper (contiguous)
  xstart     = jFrom;
  xstartRecv = jFrom;
  ystart     = iTo-1; 
  ystartRecv = iFrom-1;

  if (top >= 0)    MPI_Send( &g[l].U[ ystart * N  + xstart ], 1,  faceYType, top, 123, comm );
  if (bottom >= 0) MPI_Recv( &g[l].U[ ystartRecv * N  + xstartRecv ], 1,  faceYType, bottom, 123, comm, &status );
  

  // send from right to left
  xstart     = jFrom;
  xstartRecv = jTo;
  ystart     = iFrom; 
  ystartRecv = iFrom;

  if (left >= 0)  MPI_Send( &g[l].U[ ystart * N  + xstart ], 1,  faceXType, left, 123, comm );
  if (right >= 0) MPI_Recv( &g[l].U[ ystartRecv * N  + xstartRecv ], 1,  faceXType, right, 123, comm, &status );
 
 
  // send from left to right
  xstart     = jTo-1; 
  xstartRecv = jFrom-1;
  ystart     = iFrom; 
  ystartRecv = iFrom;

  if (right >= 0) MPI_Send( &g[l].U[ ystart * N  + xstart ], 1,  faceXType, right, 123, comm );
  if (left >= 0) MPI_Recv( &g[l].U[ ystartRecv * N  + xstartRecv ], 1,  faceXType, left, 123, comm, &status );
 
 }

 MPI_Type_free(&faceXType);
 MPI_Type_free(&faceYType);
 
 auto t1 = std::chrono::system_clock::now();
 smoothingTime[l] += std::chrono::duration<double>(t1-t0).count();
}


void applyJacobi(gridLevel* g, size_t l, size_t relaxations)
{
 auto t0 = std::chrono::system_clock::now();

 double h1 = 0.25;
 double h2 = g[l].h*g[l].h;
 for (size_t r = 0; r < relaxations; r++)
 {
  double* tmp = g[l].Un; g[l].Un = g[l].U; g[l].U = tmp;
  for (size_t i = 1; i < g[l].N-1; i++)
   for (size_t j = 1; j < g[l].N-1; j++) // Perform a Jacobi Iteration
    g[l].U[i * g[l].N + j] = (g[l].Un[ (i-1) * g[l].N + j] + g[l].Un[ (i+1) * g[l].N + j] + g[l].Un[i * g[l].N + j-1] + g[l].Un[i * g[l].N + j+1] + g[l].f[i][j]*h2)*h1;
 }

 auto t1 = std::chrono::system_clock::now();
 smoothingTime[l] += std::chrono::duration<double>(t1-t0).count();
}



void mpi_calculateResidual(gridLevel* g, size_t l, int xdim, int ydim, int xcoord, int ycoord)
{
 auto t0 = std::chrono::system_clock::now();

 double h2 = 1.0 / pow(g[l].h,2); 
 
 size_t iFrom = ceil(( ( (double) g[l].N-2) / ydim ) * ycoord ) + 1;
 size_t iTo   = ceil(( ( (double) g[l].N-2) / ydim ) * (ycoord+1)) + 1;
 size_t jFrom = ceil(( ( (double) g[l].N-2) / xdim ) * xcoord ) + 1;
 size_t jTo   = ceil(( ( (double) g[l].N-2) / xdim ) * (xcoord+1))  + 1;
 
 for (size_t i = iFrom; i < iTo; i++)
  for (size_t j = jFrom; j < jTo; j++)
     g[l].Res[i][j] = g[l].f[i][j] + (g[l].U[ (i-1) * g[l].N + j] + g[l].U[ (i+1) * g[l].N + j] - 4*g[l].U[ i * g[l].N + j] + g[l].U[ i * g[l].N + (j-1)] + g[l].U[ i * g[l].N + j+1]) * h2;

 auto t1 = std::chrono::system_clock::now();
 residualTime[l] += std::chrono::duration<double>(t1-t0).count();
}


void calculateResidual(gridLevel* g, size_t l)
{
 auto t0 = std::chrono::system_clock::now();

 double h2 = 1.0 / pow(g[l].h,2);

 for (size_t i = 1; i < g[l].N-1; i++)
 for (size_t j = 1; j < g[l].N-1; j++)
   g[l].Res[i][j] = g[l].f[i][j] + (g[l].U[ (i-1) * g[l].N + j] + g[l].U[ (i+1) * g[l].N + j] - 4*g[l].U[ i * g[l].N + j] + g[l].U[ i * g[l].N + (j-1)] + g[l].U[ i * g[l].N + j+1]) * h2;

 auto t1 = std::chrono::system_clock::now();
 residualTime[l] += std::chrono::duration<double>(t1-t0).count();
}


void mpi_calculateL2Norm(gridLevel* g, size_t l, int xdim, int ydim, int xcoord, int ycoord, double& partres)
{
 auto t0 = std::chrono::system_clock::now();

 partres = 0.0;
 
 size_t iFrom = ceil(( ( (double) g[l].N-2) / ydim ) * ycoord ) + 1;
 size_t iTo   = ceil(( ( (double) g[l].N-2) / ydim ) * (ycoord+1)) + 1;
 size_t jFrom = ceil(( ( (double) g[l].N-2) / xdim ) * xcoord ) + 1;
 size_t jTo   = ceil(( ( (double) g[l].N-2) / xdim ) * (xcoord+1))  + 1;

 for (size_t i = iFrom; i < iTo; i++)
  for (size_t j = jFrom; j < jTo; j++)
   g[l].Res[i][j] = g[l].Res[i][j]*g[l].Res[i][j];

 for (size_t i = iFrom; i < iTo; i++)
  for (size_t j = jFrom; j < jTo; j++)
   partres += g[l].Res[i][j];

 auto t1 = std::chrono::system_clock::now();
 L2NormTime[l] += std::chrono::duration<double>(t1-t0).count();
}


void calculateL2Norm(gridLevel* g, size_t l)
{
 auto t0 = std::chrono::system_clock::now();

 double tmp = 0.0;

 for (size_t i = 0; i < g[l].N; i++)
  for (size_t j = 0; j < g[l].N; j++)
   g[l].Res[i][j] = g[l].Res[i][j]*g[l].Res[i][j];

 for (size_t i = 0; i < g[l].N; i++)
  for (size_t j = 0; j < g[l].N; j++)
   tmp += g[l].Res[i][j];

 g[l].L2Norm = sqrt(tmp);
 g[l].L2NormDiff = fabs(g[l].L2NormPrev - g[l].L2Norm);
 g[l].L2NormPrev = g[l].L2Norm;
 //printf("L2Norm: %.4f\n",  g[0].L2Norm);

 auto t1 = std::chrono::system_clock::now();
 L2NormTime[l] += std::chrono::duration<double>(t1-t0).count();
}


/* TODO
void applyRestriction(gridLevel* g, size_t l)
{
  auto t0 = std::chrono::system_clock::now();

	for (size_t i = 1; i < g[l].N-1; i++)
  for (size_t j = 1; j < g[l].N-1; j++)
     g[l].f[i][j] = ( 1.0*( g[l-1].Res[2*i-1][2*j-1] + g[l-1].Res[2*i-1][2*j+1] + g[l-1].Res[2*i+1][2*j-1]   + g[l-1].Res[2*i+1][2*j+1] )   +
             2.0*( g[l-1].Res[2*i-1][2*j]   + g[l-1].Res[2*i][2*j-1]   + g[l-1].Res[2*i+1][2*j]     + g[l-1].Res[2*i][2*j+1] ) +
             4.0*( g[l-1].Res[2*i][2*j] ) ) * 0.0625;

 for (size_t i = 0; i < g[l].N; i++)
  for (size_t j = 0; j < g[l].N; j++) // Resetting U vector for the coarser level before smoothing -- Find out if this is really necessary.
  g[l].U[i][j] = 0;

 auto t1 = std::chrono::system_clock::now();
 restrictionTime[l] += std::chrono::duration<double>(t1-t0).count();
}
*/

/*
void applyProlongation(gridLevel* g, size_t l)
{
 auto t0 = std::chrono::system_clock::now();

 for (size_t i = 1; i < g[l].N-1; i++)
  for (size_t j = 1; j < g[l].N-1; j++)
   g[l-1].U[2*i][2*j] += g[l].U[i][j];

 for (size_t i = 1; i < g[l].N; i++)
  for (size_t j = 1; j < g[l].N-1; j++)
   g[l-1].U[2*i-1][2*j] += ( g[l].U[i-1][j] + g[l].U[i][j] ) *0.5;

 for (size_t i = 1; i < g[l].N-1; i++)
  for (size_t j = 1; j < g[l].N; j++)
   g[l-1].U[2*i][2*j-1] += ( g[l].U[i][j-1] + g[l].U[i][j] ) *0.5;

 for (size_t i = 1; i < g[l].N; i++)
  for (size_t j = 1; j < g[l].N; j++)
   g[l-1].U[2*i-1][2*j-1] += ( g[l].U[i-1][j-1] + g[l].U[i-1][j] + g[l].U[i][j-1] + g[l].U[i][j] ) *0.25;

 auto t1 = std::chrono::system_clock::now();
 prolongTime[l] += std::chrono::duration<double>(t1-t0).count();
}*/


gridLevel* generateInitialConditions(size_t N0, size_t gridCount)
{
 // Default values:
 __p.nCandles = 4;
 std::vector<double> pars;
 pars.push_back(0.228162);
 pars.push_back(0.226769);
 pars.push_back(0.437278);
 pars.push_back(0.0492324);
 pars.push_back(0.65915);
 pars.push_back(0.499616);
 pars.push_back(0.59006);
 pars.push_back(0.0566329);
 pars.push_back(0.0186672);
 pars.push_back(0.894063);
 pars.push_back(0.424229);
 pars.push_back(0.047725);
 pars.push_back(0.256743);
 pars.push_back(0.754483);
 pars.push_back(0.490461);
 pars.push_back(0.0485152);

 // Allocating Timers
 smoothingTime = (double*) calloc (gridCount, sizeof(double));
 residualTime = (double*) calloc (gridCount, sizeof(double));
 restrictionTime = (double*) calloc (gridCount, sizeof(double));
 prolongTime = (double*) calloc (gridCount, sizeof(double));
 L2NormTime = (double*) calloc (gridCount, sizeof(double));

 // Allocating Grids
 gridLevel* g = (gridLevel*) malloc(sizeof(gridLevel) * gridCount);
 for (size_t i = 0; i < gridCount; i++)
 {
  g[i].N = pow(2, N0-i) + 1;
  g[i].h = 1.0/(g[i].N-1);

  g[i].U   = (double*) calloc(sizeof(double*), g[i].N * g[i].N ); 
  g[i].Un  = (double*) calloc(sizeof(double*), g[i].N * g[i].N ); 
  g[i].Res = (double**) malloc(sizeof(double*) * g[i].N); for (size_t j = 0; j < g[i].N ; j++) g[i].Res[j] = (double*) malloc(sizeof(double) * g[i].N);
  g[i].f   = (double**) malloc(sizeof(double*) * g[i].N); for (size_t j = 0; j < g[i].N ; j++) g[i].f[j]   = (double*) malloc(sizeof(double) * g[i].N);

  g[i].L2Norm = 0.0;
  g[i].L2NormPrev = std::numeric_limits<double>::max();
  g[i].L2NormDiff = std::numeric_limits<double>::max();
 }

 // Initial Guess
 for (size_t i = 0; i < g[0].N * g[0].N; i++) g[0].U[i] = 1.0;

 // Boundary Conditions
 for (size_t i = 0; i < g[0].N; i++) 
    for (size_t j = 0; j < g[0].N; j++)
    {
        if (i == 0) g[0].U[ i * g[0].N + j] = 0;
        if (j == 0) g[0].U[ i * g[0].N + j] = 0;
        if (i == g[0].N-1) g[0].U[ i * g[0].N + j] = 0;
        if (j == g[0].N-1) g[0].U[ i * g[0].N + j] = 0;
    }
 
 // F
 for (size_t i = 0; i < g[0].N; i++)
 for (size_t j = 0; j < g[0].N; j++)
 {
  double h = 1.0/(g[0].N-1);
  double x = i*h;
  double y = j*h;

  g[0].f[i][j] = 0.0;

  for (size_t c = 0; c < __p.nCandles; c++)
  {
   double c3 = pars[c*4  + 0]; // x0
   double c4 = pars[c*4  + 1]; // y0
   double c1 = pars[c*4  + 2]; c1 *= 100000;// intensity
   double c2 = pars[c*4  + 3]; c2 *= 0.01;// Width
   g[0].f[i][j] += c1*exp(-(pow(c4 - y, 2) + pow(c3 - x, 2)) / c2);
  }
 }

 return g;
}

void freeGrids(gridLevel* g, size_t gridCount)
{
 for (size_t i = 0; i < gridCount; i++)
 {
  for (size_t j = 0; j < g[i].N ; j++) free(g[i].f[j]);
  for (size_t j = 0; j < g[i].N ; j++) free(g[i].Res[j]);
  free(g[i].U);
  free(g[i].Un);
  free(g[i].f);
  free(g[i].Res);
 }
 free(g);
}

void printTimings(size_t gridCount)
{
	double* timePerGrid = (double*) calloc (sizeof(double), gridCount);
	double totalSmoothingTime = 0.0;
	double totalResidualTime = 0.0;
	double totalRestrictionTime = 0.0;
	double totalProlongTime = 0.0;
	double totalL2NormTime = 0.0;

	for (size_t i = 0; i < gridCount; i++) timePerGrid[i] = smoothingTime[i] + residualTime[i] + restrictionTime[i] + prolongTime[i] + L2NormTime[i];
	for (size_t i = 0; i < gridCount; i++) totalSmoothingTime += smoothingTime[i];
	for (size_t i = 0; i < gridCount; i++) totalResidualTime += residualTime[i];
	for (size_t i = 0; i < gridCount; i++) totalRestrictionTime += restrictionTime[i];
	for (size_t i = 0; i < gridCount; i++) totalProlongTime += prolongTime[i];
	for (size_t i = 0; i < gridCount; i++) totalL2NormTime += L2NormTime[i];

	double totalMeasured = totalSmoothingTime + totalResidualTime + totalRestrictionTime + totalProlongTime + totalL2NormTime;

	printf("   Time (s)    "); for (size_t i = 0; i < gridCount; i++) printf("Grid%lu   ", i);                    printf("   Total  \n");
	printf("-------------|-"); for (size_t i = 0; i < gridCount; i++) printf("--------"); printf("|---------\n");
	printf("Smoothing    | "); for (size_t i = 0; i < gridCount; i++) printf("%2.3f   ", smoothingTime[i]);    printf("|  %2.3f  \n", totalSmoothingTime);
	printf("Residual     | "); for (size_t i = 0; i < gridCount; i++) printf("%2.3f   ", residualTime[i]);     printf("|  %2.3f  \n", totalResidualTime);
	printf("Restriction  | "); for (size_t i = 0; i < gridCount; i++) printf("%2.3f   ", restrictionTime[i]);  printf("|  %2.3f  \n", totalRestrictionTime);
	printf("Prolongation | "); for (size_t i = 0; i < gridCount; i++) printf("%2.3f   ", prolongTime[i]);      printf("|  %2.3f  \n", totalProlongTime);
	printf("L2Norm       | "); for (size_t i = 0; i < gridCount; i++) printf("%2.3f   ", L2NormTime[i]);       printf("|  %2.3f  \n", totalL2NormTime);
	printf("-------------|-"); for (size_t i = 0; i < gridCount; i++) printf("--------"); printf("|---------\n");
	printf("Total        | "); for (size_t i = 0; i < gridCount; i++) printf("%2.3f   ", timePerGrid[i]); printf("|  %2.3f  \n", totalMeasured);
	printf("-------------|-"); for (size_t i = 0; i < gridCount; i++) printf("--------"); printf("|---------\n");
	printf("\n");
	printf("Running Time      : %.3fs\n", totalTime);
}

