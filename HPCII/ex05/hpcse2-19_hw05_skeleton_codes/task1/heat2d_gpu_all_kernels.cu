/**********************************************************************/
// A now optimized Multigrid Solver for the Heat Equation             //
// Course Material for HPCSE-II, Spring 2019, ETH Zurich              //
// Authors: Sergio Martin, Georgios Arampatzis                        //
// License: Use if you like, but give us credit.                      //
/**********************************************************************/

#include <stdio.h>
#include <math.h>
#include <limits>
#include "heat2d_gpu.hpp"
#include "string.h"
#include <chrono>

pointsInfo __p;

void checkCUDAError(const char *msg)
{
 cudaError_t err = cudaGetLastError();
 if( cudaSuccess !=err )
 {
   fprintf(stderr," CUDA Error: %s: %s. \n",msg, cudaGetErrorString(err));
	exit(-1);	
 }
}
//__global__ void dgemm(double *A, double *B, double *C, size_t N)

int main(int argc, char* argv[])
{
 double tolerance = 1e-0; // L2 Difference Tolerance before reaching convergence.
 size_t N0 = 10; // 2^N0 + 1 elements per side

 // Multigrid parameters -- Find the best configuration!
 size_t gridCount       = N0-1;     // Number of Multigrid levels to use
 size_t downRelaxations = 5; // Number of Relaxations before restriction
 size_t upRelaxations   = 0;   // Number of Relaxations after prolongation
 
  
 gridLevel* g = generateInitialConditions(N0, gridCount);
	
  gridLevel* d_g;
  cudaMalloc(&d_g, sizeof(gridLevel) * gridCount);
  cudaMemcpy(d_g,&g,sizeof(gridLevel) * gridCount, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  
  
 auto startTime = std::chrono::system_clock::now();
 while (g[0].L2NormDiff > tolerance)  // Multigrid solver start
 {
  
  
  applyJacobi(d_g, 0, downRelaxations); // Relaxing the finest grid first
  calculateResidual(g, 0); // Calculating Initial Residual

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

  calculateL2Norm(g, 0); // Calculating Residual L2 Norm
 }  // Multigrid solver end

  cudaDeviceSynchronize();
  cudaMemcpy(g,&d_g,sizeof(gridLevel) * gridCount, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
//  __syncthreads(); //  cudaDeviceSynchronize();


 auto endTime = std::chrono::system_clock::now();
 totalTime = std::chrono::duration<double>(endTime-startTime).count();
 printTimings(gridCount);
 printf("L2Norm: %.4f\n",  g[0].L2Norm);
 freeGrids(g, gridCount);
 return 0;
}

//kernel function. Works only on the assigned thread Eex9Koh4pha-
__global__ void applyJacobiKernel(gridLevel* g, size_t l, double h1, double h2)
{
 size_t myRow = blockIdx.y*blockDim.y + threadIdx.y;
 size_t myCol = blockIdx.x*blockDim.x + threadIdx.x;
 if(myRow >= g[l].N-1 || myCol >= g[l].N-1)return; 
 if(myRow < 1 || myCol < 1 )return;
  //for (size_t i = 1; i < g[l].N-1; i++)
   //for (size_t j = 1; j < g[l].N-1; j++) // Perform a Jacobi Iteration
 g[l].U[myRow][myCol] = (g[l].Un[myRow-1][myCol] + g[l].Un[myRow+1][myCol] + g[l].Un[myRow][myCol-1] + g[l].Un[myRow][myCol+1] + g[l].f[myRow][myCol]*h2)*h1;
__syncthreads();
}

void applyJacobi(gridLevel* d_g, size_t l, size_t relaxations)
{
 int N = d_g[l].N;

 dim3 threadsPerBlock(32,32);
 dim3 blocksPerGrid(N/32,N/32);//depends on the size of the problem. int or dim3
 auto t0 = std::chrono::system_clock::now();
 double h1 = 0.25;
 double h2 = d_g[l].h*d_g[l].h;
 for (size_t r = 0; r < relaxations; r++)
 {
  double** tmp = d_g[l].Un; d_g[l].Un = d_g[l].U; d_g[l].U = tmp;
  applyJacobiKernel<<< blocksPerGrid,threadsPerBlock >>>(d_g, l, h1, h2);
 }
auto t1 = std::chrono::system_clock::now();
 smoothingTime[l] += std::chrono::duration<double>(t1-t0).count();
}

__global__ void calculateResidualKernel(gridLevel* g, size_t l, double h2)
{
 size_t myRow = blockIdx.y*blockDim.y + threadIdx.y;
 size_t myCol = blockIdx.x*blockDim.x + threadIdx.x;
 if(myRow >= g[l].N-1 || myCol >= g[l].N-1)return; 
 if(myRow < 1 || myCol < 1 )return;
 g[l].Res[myRow][myCol] = g[l].f[myRow][myCol] + (g[l].U[myRow-1][myCol] + g[l].U[myRow+1][myCol] - 4*g[l].U[myRow][myCol] + g[l].U[myRow][myCol-1] + g[l].U[myRow][myCol+1]) * h2;
}

void calculateResidual(gridLevel* g, size_t l)
{
 int N = g[l].N;
 dim3 threadsPerBlock(32,32);
 dim3 blocksPerGrid(N/32,N/32);//depends on the size of the problem. int or dim3
 auto t0 = std::chrono::system_clock::now();
 double h2 = 1.0 / pow(g[l].h,2);
 calculateResidualKernel<<< blocksPerGrid,threadsPerBlock >>>(g, l, h2);
 auto t1 = std::chrono::system_clock::now();
 residualTime[l] += std::chrono::duration<double>(t1-t0).count();
}

__global__ void calculateL2NormKernel(gridLevel* g, size_t l)
{
 double tmp = 0.0;
 size_t i = blockIdx.y*blockDim.y + threadIdx.y;
 size_t j = blockIdx.x*blockDim.x + threadIdx.x;
 if(i >= g[l].N-1 || j >= g[l].N-1)return; 
 if(i < 1 || j < 1 )return;
   g[l].Res[i][j] = g[l].Res[i][j]*g[l].Res[i][j];
   __syncthreads();
   tmp += g[l].Res[i][j];
   __syncthreads();
}

void calculateL2Norm(gridLevel* g, size_t l)
{
 int N = d_g[l].N;
 dim3 threadsPerBlock(32,32);
 dim3 blocksPerGrid(N/32,N/32);//depends on the size of the problem. int or dim3
 double tmp = 0.0;
 auto t0 = std::chrono::system_clock::now();

 calculateL2NormKernel<<< blocksPerGrid,threadsPerBlock >>>(d_g, l);

 auto t1 = std::chrono::system_clock::now();
 g[l].L2Norm = sqrt(tmp);
 g[l].L2NormDiff = fabs(g[l].L2NormPrev - g[l].L2Norm);
 g[l].L2NormPrev = g[l].L2Norm;
 
 L2NormTime[l] += std::chrono::duration<double>(t1-t0).count();
}

__global__ void applyRestrictionKernel(gridLevel* g, size_t l)
{
 double tmp = 0.0;
 size_t i = blockIdx.y*blockDim.y + threadIdx.y;
 size_t j = blockIdx.x*blockDim.x + threadIdx.x;
 if(i >= g[l].N-1 || j >= g[l].N-1)return; 
 if(i < 1 || j < 1 )return;
 g[l].f[i][j] = ( 1.0*( g[l-1].Res[2*i-1][2*j-1] + g[l-1].Res[2*i-1][2*j+1] + g[l-1].Res[2*i+1][2*j-1]   + g[l-1].Res[2*i+1][2*j+1] )   +
             2.0*( g[l-1].Res[2*i-1][2*j]   + g[l-1].Res[2*i][2*j-1]   + g[l-1].Res[2*i+1][2*j]     + g[l-1].Res[2*i][2*j+1] ) +
             4.0*( g[l-1].Res[2*i][2*j] ) ) * 0.0625;
 __syncthreads();
  g[l].U[i][j] = 0;// Resetting U vector for the coarser level before smoothing -- Find out if this is really necessary.
}

void applyRestriction(gridLevel* g, size_t l)
{ 
 int N = g[l].N;
 dim3 threadsPerBlock(32,32);
 dim3 blocksPerGrid(N/32,N/32);//depends on the size of the problem. int or dim3
 auto t0 = std::chrono::system_clock::now();

 applyRestrictionKernel<<< blocksPerGrid,threadsPerBlock >>>(d_g, l);

 auto t1 = std::chrono::system_clock::now();
 restrictionTime[l] += std::chrono::duration<double>(t1-t0).count();
}

__global__ void applyProlongationKernel(gridLevel* g, size_t l)
{
 double tmp = 0.0;
 size_t i = blockIdx.y*blockDim.y + threadIdx.y;
 size_t j = blockIdx.x*blockDim.x + threadIdx.x;
 if(i < 1 || j < 1 );
 if(myRow < g[l].N-1 || myCol >= g[l].N-1)
}

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
}

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

//	std::vector<double>* d_pars;
//	cudaMalloc(&d_pars, pars.size()*sizeof(double));
//	cudaMemcpy(d_pars,&pars,pars.size()*sizeof(double), cudaMemcpyDeviceToHost);

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

  g[i].U   = (double**) malloc(sizeof(double*) * g[i].N); for (size_t j = 0; j < g[i].N ; j++) g[i].U[j]   = (double*) malloc(sizeof(double) * g[i].N);
  g[i].Un  = (double**) malloc(sizeof(double*) * g[i].N); for (size_t j = 0; j < g[i].N ; j++) g[i].Un[j]  = (double*) malloc(sizeof(double) * g[i].N);
  g[i].Res = (double**) malloc(sizeof(double*) * g[i].N); for (size_t j = 0; j < g[i].N ; j++) g[i].Res[j] = (double*) malloc(sizeof(double) * g[i].N);
  g[i].f   = (double**) malloc(sizeof(double*) * g[i].N); for (size_t j = 0; j < g[i].N ; j++) g[i].f[j]   = (double*) malloc(sizeof(double) * g[i].N);

  g[i].L2Norm = 0.0;
  g[i].L2NormPrev = std::numeric_limits<double>::max();
  g[i].L2NormDiff = std::numeric_limits<double>::max();
 }

 // Initial Guess
 for (size_t i = 0; i < g[0].N; i++) for (size_t j = 0; j < g[0].N; j++) g[0].U[i][j] = 1.0;

 // Boundary Conditions
 for (size_t i = 0; i < g[0].N; i++) g[0].U[0][i]        = 0.0;
 for (size_t i = 0; i < g[0].N; i++) g[0].U[g[0].N-1][i] = 0.0;
 for (size_t i = 0; i < g[0].N; i++) g[0].U[i][0]        = 0.0;
 for (size_t i = 0; i < g[0].N; i++) g[0].U[i][g[0].N-1] = 0.0;

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
  for (size_t j = 0; j < g[i].N ; j++) free(g[i].U[j]);
  for (size_t j = 0; j < g[i].N ; j++) free(g[i].f[j]);
  for (size_t j = 0; j < g[i].N ; j++) free(g[i].Res[j]);
  free(g[i].U);
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

