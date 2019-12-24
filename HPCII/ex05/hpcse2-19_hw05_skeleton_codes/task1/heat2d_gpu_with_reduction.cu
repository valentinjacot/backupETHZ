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

void checkCUDAError(const char *msg)
{
 cudaError_t err = cudaGetLastError();
 if( cudaSuccess !=err )
 {
   fprintf(stderr," CUDA Error: %s: %s. \n",msg, cudaGetErrorString(err));
	exit(-1);	
 }
}

void custommemcpyToHost(gridLevel* g, size_t l){
 for (size_t i = 0; i < g[l].N; i++){
 	cudaMemcpy(g[l].f[i],	&g[l].df[i*g[l].N], 	g[l].N*sizeof(double),cudaMemcpyDeviceToHost);	checkCUDAError("cpy f To Host");
	cudaMemcpy(g[l].U[i],	&g[l].dU[i*g[l].N], 	g[l].N*sizeof(double),cudaMemcpyDeviceToHost);	checkCUDAError("cpy U To Host");
 	cudaMemcpy(g[l].Un[i], 	&g[l].dUn[i*g[l].N], 	g[l].N*sizeof(double),cudaMemcpyDeviceToHost);	checkCUDAError("cpy Un To Host");
 	cudaMemcpy(g[l].Res[i], &g[l].dRes[i*g[l].N], 	g[l].N*sizeof(double),cudaMemcpyDeviceToHost);	checkCUDAError("cpy Res To Host");
cudaDeviceSynchronize();
 }
}

void custommemcpyToDevice(gridLevel* g, size_t l){
 for (size_t i = 0; i < g[l].N; i++){
 	cudaMemcpy(&g[l].df[i*g[l].N],	g[l].f[i],  	g[l].N*sizeof(double),cudaMemcpyHostToDevice);	checkCUDAError("cpy f To Device");
	cudaMemcpy(&g[l].dU[i*g[l].N],	g[l].U[i], 	g[l].N*sizeof(double),cudaMemcpyHostToDevice);	checkCUDAError("cpy U To Device");
 	cudaMemcpy(&g[l].dUn[i*g[l].N],	g[l].Un[i],	g[l].N*sizeof(double),cudaMemcpyHostToDevice);	checkCUDAError("cpy Un To Device");
 	cudaMemcpy(&g[l].dRes[i*g[l].N],g[l].Res[i],	g[l].N*sizeof(double),cudaMemcpyHostToDevice);	checkCUDAError("cpy Res To Device");
cudaDeviceSynchronize();
 }
}

pointsInfo __p;

int main(int argc, char* argv[])
{
 double tolerance = 1e-0; // L2 Difference Tolerance before reaching convergence.
 size_t N0 = 10; // 2^N0 + 1 elements per side

 // Multigrid parameters -- Find the best configuration!
 size_t gridCount       = N0-1;     // Number of Multigrid levels to use
 size_t downRelaxations = 5; // Number of Relaxations before restriction
 size_t upRelaxations   = 0;   // Number of Relaxations after prolongation)
//I coudln't find a much better configuration, as the times are so small already

 gridLevel* g = generateInitialConditions(N0, gridCount); 
  
// gridLevel* d_g;
// cudaMalloc(&d_g, sizeof(gridLevel) * gridCount);checkCUDAError("Alloc error");
// cudaMemcpy(d_g,&g,sizeof(gridLevel) * gridCount, cudaMemcpyHostToDevice);checkCUDAError("Memcpy error");
// cudaDeviceSynchronize();  
//for (size_t grid = 1; grid < gridCount; grid++) custommemcpyToDevice(g,grid);
 auto startTime = std::chrono::system_clock::now();
 while (g[0].L2NormDiff > tolerance)  // Multigrid solver start
 {
  applyJacobi(g, 0, downRelaxations); // Relaxing the finest grid first
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
//custommemcpyToHost(g,grid);
 printf("L2Norm: %.4f\n",  g[0].L2Norm);
  calculateL2Norm(g, 0); // Calculating Residual L2 Norm
 }  // Multigrid solver end

 cudaDeviceSynchronize();
 auto endTime = std::chrono::system_clock::now();
 totalTime = std::chrono::duration<double>(endTime-startTime).count();
 printTimings(gridCount);
 printf("L2Norm: %.4f\n",  g[0].L2Norm);
 freeGrids(g, gridCount);
 return 0;
}

//kernel function. Works only on the assigned thread Eex9Koh4pha-
__global__ void applyJacobiKernel( size_t N, double h1, double h2, double* Un, double* f, double* U)
{
 size_t myRow = blockIdx.y*blockDim.y + threadIdx.y;
 size_t myCol = blockIdx.x*blockDim.x + threadIdx.x;
 if(myRow >= N-1 || myCol >= N-1)return; 
 if(myRow < 1 || myCol < 1 )return; 
 U[myRow*N + myCol] = (Un[(myRow-1)*N + myCol]+Un[(myRow+1)*N + myCol]+Un[myRow*N + myCol-1]+Un[myRow*N + myCol+1]+f[myRow*N+myCol]*h2)*h1;
 //g[l].U[myRow][myCol] = (g[l].Un[myRow-1][myCol] + g[l].Un[myRow+1][myCol] + g[l].Un[myRow][myCol-1] + g[l].Un[myRow][myCol+1] + g[l].f[myRow][myCol]*h2)*h1;
//__syncthreads();
}

void applyJacobi(gridLevel* g, size_t l, size_t relaxations)
{
//custommemcpyToDevice(g,l);
int N = g[l].N;
dim3 threadsPerBlock=dim3(32,32);
dim3 blocksPerGrid;
if(N<32) {blocksPerGrid=dim3(ceil(N),ceil(N));}
else {blocksPerGrid=dim3(ceil(N/32),ceil(N/32));}//depends on the size of the problem. int or dim3
 auto t0 = std::chrono::system_clock::now();
 double h1 = 0.25;
 double h2 = g[l].h*g[l].h;
 for (size_t r = 0; r < relaxations; r++)
 {
//*
  double* tmp = g[l].dUn; g[l].dUn = g[l].dU; g[l].dU = tmp;
  applyJacobiKernel<<< blocksPerGrid,threadsPerBlock >>>(N, h1, h2, g[l].dUn, g[l].df, g[l].dU);checkCUDAError("apply Jacobi error");
  //cudaDeviceSynchronize();
//*/
/*
  for (size_t i = 1; i < g[l].N-1; i++)
   for (size_t j = 1; j < g[l].N-1; j++) // Perform a Jacobi Iteration
    g[l].U[i][j] = (g[l].Un[i-1][j] + g[l].Un[i+1][j] + g[l].Un[i][j-1] + g[l].Un[i][j+1] + g[l].f[i][j]*h2)*h1;
//*/
}
 cudaDeviceSynchronize();
 auto t1 = std::chrono::system_clock::now();
 smoothingTime[l] += std::chrono::duration<double>(t1-t0).count();
//custommemcpyToHost(g, l);
}

//kernel fct. Eex9Koh4pha-
__global__ void calculateResidualKernel( size_t N, double h2, double* Res, double* f, double* U)
{
 size_t myRow = blockIdx.y*blockDim.y + threadIdx.y;
 size_t myCol = blockIdx.x*blockDim.x + threadIdx.x;
 if(myRow >= N-1 || myCol >= N-1)return; 
 if(myRow < 1 || myCol < 1 )return;
 Res[myRow*N + myCol] = f[myRow*N + myCol] + (U[(myRow-1)*N+myCol] + U[(myRow+1)*N + myCol] - 4*U[myRow*N +myCol] + U[myRow*N +myCol-1] + U[myRow*N +myCol+1]) * h2;
}

void calculateResidual(gridLevel* g, size_t l)
{
//custommemcpyToDevice(g,l);
 int N = g[l].N;
dim3 threadsPerBlock=dim3(32,32);
dim3 blocksPerGrid;
if(N<32) {blocksPerGrid=dim3(ceil(N),ceil(N));}
else {blocksPerGrid=dim3(ceil(N/32),ceil(N/32));}

 auto t0 = std::chrono::system_clock::now();
 double h2 = 1.0 / pow(g[l].h,2);
/*
 for (size_t i = 1; i < g[l].N-1; i++)
 for (size_t j = 1; j < g[l].N-1; j++)
 g[l].Res[i][j] = g[l].f[i][j] + (g[l].U[i-1][j] + g[l].U[i+1][j] - 4*g[l].U[i][j] + g[l].U[i][j-1] + g[l].U[i][j+1]) * h2;
*/
//*
 calculateResidualKernel<<< blocksPerGrid,threadsPerBlock >>>(N, h2, g[l].dRes, g[l].df, g[l].dU);checkCUDAError("Calculate Residual Kernel error");
//  cudaDeviceSynchronize();
//*/
 cudaDeviceSynchronize();
 auto t1 = std::chrono::system_clock::now();
 residualTime[l] += std::chrono::duration<double>(t1-t0).count();
//custommemcpyToHost(g, l);
}

//kernel fct. Eex9Koh4pha-
__global__ void calculateSquareKernel(double *Res, size_t N)
{
size_t i = blockIdx.y*blockDim.y + threadIdx.y;
size_t j = blockIdx.x*blockDim.x + threadIdx.x;
if(i >= N || j >= N)return;
Res[i*N+j] = Res[i*N+j]*Res[i*N+j];
}


#define BLOCKSIZE 1024
//from the class example 3

__global__ void reduce(double* dVec, double* dAux, size_t N,size_t N2)
{
 __shared__ unsigned int sdata[BLOCKSIZE];

 size_t tid = threadIdx.x;
 size_t i = blockIdx.x*blockDim.x + threadIdx.x;
if(i> N2) return;
 sdata[tid] = dVec[i];
 __syncthreads();

 for (unsigned int s=blockDim.x/2; s>0; s>>=1)
 {
  if (tid < s)  sdata[tid] += sdata[tid + s];

  __syncthreads();
 }

 if (tid == 0) dAux[blockIdx.x] = sdata[0];
}

//Eex9Koh4pha-
void calculateL2Norm(gridLevel* g, size_t l)
{
 //double tmp = 0.0;
 auto t0 = std::chrono::system_clock::now();
//*custommemcpyToDevice(g,l);
//*
int N = g[l].N;
dim3 threadsPerBlock=dim3(32,32);
dim3 blocksPerGrid;
if(N<32) {blocksPerGrid=dim3(ceil(N),ceil(N));}
else {blocksPerGrid=dim3(ceil(N/32),ceil(N/32));}
calculateSquareKernel<<< blocksPerGrid,threadsPerBlock >>>(g[l].dRes, N);checkCUDAError("Calculate square Kernel error");
cudaDeviceSynchronize();
double *dtemp; 
cudaMalloc(&dtemp, sizeof(double)*N*N); checkCUDAError("Error allocating dtemp");
cudaMemset(dtemp, 0.0, sizeof(double)*N*N);checkCUDAError("Error memset dtemp");
double *dRescpy; //sizeof(double)*BLOCKSIZE*BLOCKSIZE*4
cudaMalloc(&dRescpy, sizeof(double)*N*N); checkCUDAError("Error allocating dtemp");
//cudaMemset(dRescpy, 0.0, sizeof(double)*N*N);checkCUDAError("Error memset dtemp");
cudaDeviceSynchronize();
cudaMemcpy(dRescpy, g[l].dRes, sizeof(double)*N*N,cudaMemcpyDeviceToDevice);	checkCUDAError("cpy Res To Host in L2");
cudaDeviceSynchronize();
for (size_t n =BLOCKSIZE*BLOCKSIZE*4; n > 1; n = n / BLOCKSIZE)
  {
    	size_t bSize = BLOCKSIZE;           if (bSize > n) bSize = n;
  	size_t gSize = ceil((double)n / (double)BLOCKSIZE); if (bSize > n) gSize = 1;
  	printf("bSize: %lu - gSize: %lu\n", bSize, gSize);
     reduce<<<gSize, bSize,BLOCKSIZE*sizeof(unsigned int)>>>(dRescpy, dtemp, n, N*N);
    double *tmp = dRescpy; dRescpy = dtemp; dtemp = tmp;
  }
cudaDeviceSynchronize();
/*
 for (size_t i = 0; i < g[l].N; i++)
  for (size_t j = 0; j < g[l].N; j++)
   g[l].Res[i][j] = g[l].Res[i][j]*g[l].Res[i][j];
//*/
/*
 for (size_t i = 0; i < g[l].N; i++)
  for (size_t j = 0; j < g[l].N; j++)
   tmp += temp[i*N+j];
//*/

//printf("%d \n", tmp);
 double result = 0.0; 
 cudaMemcpy(&result, dRescpy, sizeof(double), cudaMemcpyDeviceToHost);	checkCUDAError("error copying result");
 //cudaMemcpy(&result2, dRescpy, sizeof(double), cudaMemcpyDeviceToHost);	checkCUDAError("error copying result2");
 cudaDeviceSynchronize();
 g[l].L2Norm = sqrt(result);
 g[l].L2NormDiff = fabs(g[l].L2NormPrev - g[l].L2Norm);
 g[l].L2NormPrev = g[l].L2Norm;

 cudaDeviceSynchronize();
 auto t1 = std::chrono::system_clock::now();
 L2NormTime[l] += std::chrono::duration<double>(t1-t0).count();
cudaFree(dtemp);
cudaFree(dRescpy);
//custommemcpyToDevice(g,l);
}
//Eex9Koh4pha-

//kernel fct. Eex9Koh4pha-
__global__ void applyRestrictionKernel( size_t N, size_t N2, double* Res, double* f)
{
 size_t i = blockIdx.y*blockDim.y + threadIdx.y;
 size_t j = blockIdx.x*blockDim.x + threadIdx.x;
double c =0.0;
 if(i >= N-1 || j >= N-1)return; 
 if(i < 1 || j < 1 )return;
   c = ( 1.0*( Res[(2*i-1)*N2 +(2*j-1)] + Res[(2*i-1)*N2 +(2*j+1)] + Res[(2*i+1)*N2 +(2*j-1)]   + Res[(2*i+1)*N2 +(2*j+1)] )   +
             2.0*( Res[(2*i-1)*N2 +2*j]   + Res[(2*i)*N2 +(2*j-1)]   + Res[(2*i+1)*N2 +2*j]     + Res[(2*i)*N2 +(2*j+1)] ) +
             4.0*( Res[(2*i)*N2 +(2*j)] ) ) * 0.0625;
f[i*N +j]=c;
}

void applyRestriction(gridLevel* g, size_t l)
{ 
//custommemcpyToDevice(g, l);custommemcpyToDevice(g, l-1);
int N = g[l].N;
int N2 = g[l-1].N;//
double *temp; 
cudaMalloc(&temp, sizeof(double) * N2 * N2);	checkCUDAError("Alloc error temp");
cudaMemcpy(temp,g[l-1].dRes, N2*N2*sizeof(double),cudaMemcpyDeviceToDevice);	checkCUDAError("cpy temp");
dim3 threadsPerBlock=dim3(32,32);
dim3 blocksPerGrid;
if(N<32) {blocksPerGrid=dim3(ceil(N),ceil(N));}
else {blocksPerGrid=dim3(ceil(N/32),ceil(N/32));}
  auto t0 = std::chrono::system_clock::now();
/*
 for (size_t i = 1; i < g[l].N-1; i++)
  for (size_t j = 1; j < g[l].N-1; j++)
     g[l].f[i][j] = ( 1.0*( g[l-1].Res[2*i-1][2*j-1] + g[l-1].Res[2*i-1][2*j+1] + g[l-1].Res[2*i+1][2*j-1]   + g[l-1].Res[2*i+1][2*j+1] )   +
             2.0*( g[l-1].Res[2*i-1][2*j]   + g[l-1].Res[2*i][2*j-1]   + g[l-1].Res[2*i+1][2*j]     + g[l-1].Res[2*i][2*j+1] ) +
             4.0*( g[l-1].Res[2*i][2*j] ) ) * 0.0625;

 for (size_t i = 0; i < g[l].N; i++)
  for (size_t j = 0; j < g[l].N; j++) // Resetting U vector for the coarser level before smoothing -- Find out if this is really necessary.
  	g[l].U[i][j] = 0;
//*/
//*
 applyRestrictionKernel<<< blocksPerGrid,threadsPerBlock >>>(N, N2, temp, g[l].df);checkCUDAError("Apply Restriction Kernel error");
//*/
 cudaDeviceSynchronize();
cudaMemset(g[l].dU, 0.0, sizeof(double)* N*N);// Resetting U vector for the coarser level before smoothing
 auto t1 = std::chrono::system_clock::now();
 restrictionTime[l] += std::chrono::duration<double>(t1-t0).count();
//custommemcpyToHost(g, l);
}
//kernel fct. Eex9Koh4pha-
__global__ void applyProlongationKernel1( size_t N, size_t N2, double* U, double* Um)
{
 size_t i = blockIdx.y*blockDim.y + threadIdx.y;
 size_t j = blockIdx.x*blockDim.x + threadIdx.x;
 if( i > 0 && j > 0 && i < N-1 && j < N-1) 	Um[2*i * N2 + 2*j] += U[i* N + j];
}
__global__ void applyProlongationKernel2( size_t N, size_t N2, double* U, double* Um)
{
 size_t i = blockIdx.y*blockDim.y + threadIdx.y;
 size_t j = blockIdx.x*blockDim.x + threadIdx.x;
 if( i > 0 && j > 0 && i < N && j < N-1)		Um[(2*i-1)* N2 + 2*j] += ( U[(i-1)* N + j] + U[i* N + j] ) *0.5;
}

__global__ void applyProlongationKernel3( size_t N, size_t N2, double* U, double* Um)
{
 size_t i = blockIdx.y*blockDim.y + threadIdx.y;
 size_t j = blockIdx.x*blockDim.x + threadIdx.x;
 if( i > 0 && j > 0 && i < N-1 && j < N)		Um[2*i* N2 + (2*j-1)] += ( U[i* N + (j-1)] + U[i* N + j] ) *0.5;
}
__global__ void applyProlongationKernel4( size_t N, size_t N2, double* U, double* Um)
{
 size_t i = blockIdx.y*blockDim.y + threadIdx.y;
 size_t j = blockIdx.x*blockDim.x + threadIdx.x;
 if( i > 0 && j > 0 && i < N && j < N)			Um[(2*i-1)* N2 + (2*j-1)] += ( U[(i-1)* N + (j-1)] + U[(i-1)* N + j] + U[i* N + (j-1)] + U[i* N + j] ) *0.25;
}

void applyProlongation(gridLevel* g, size_t l)
{
//custommemcpyToDevice(g, l);
//custommemcpyToDevice(g, l-1);
int N = g[l].N;
int N2 = g[l-1].N;
double *temp; 
cudaMalloc(&temp, sizeof(double) * g[l-1].N * g[l-1].N);checkCUDAError("Alloc error temp");
cudaMemcpy(temp,	g[l-1].dU, g[l-1].N*g[l-1].N*sizeof(double),cudaMemcpyDeviceToDevice);	checkCUDAError("cpy temp dres");
dim3 threadsPerBlock=dim3(32,32);
dim3 blocksPerGrid;
if(N<32) {blocksPerGrid=dim3(ceil(N),ceil(N));}
else {blocksPerGrid=dim3(ceil(N/32),ceil(N/32));}
//2 pointers to U from 2 different levels of g
//Kernel size is definded by g[l] not g [l-1] 
 auto t0 = std::chrono::system_clock::now();
/*
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
//*/
 applyProlongationKernel1<<< blocksPerGrid,threadsPerBlock >>>( N, N2, g[l].dU, temp);
 applyProlongationKernel2<<< blocksPerGrid,threadsPerBlock >>>( N, N2, g[l].dU, temp);
 applyProlongationKernel3<<< blocksPerGrid,threadsPerBlock >>>( N, N2, g[l].dU, temp);
 applyProlongationKernel4<<< blocksPerGrid,threadsPerBlock >>>( N, N2, g[l].dU, temp);
 cudaDeviceSynchronize();
 cudaMemcpy(g[l-1].dU, temp, g[l-1].N*g[l-1].N*sizeof(double),cudaMemcpyDeviceToDevice);	checkCUDAError("cpy temp");
 cudaDeviceSynchronize();
 auto t1 = std::chrono::system_clock::now();
 prolongTime[l] += std::chrono::duration<double>(t1-t0).count();
//custommemcpyToHost(g, l-1);
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

  cudaMalloc(&g[i].dU, sizeof(double) * g[i].N * g[i].N);checkCUDAError("Alloc error d_U");
  cudaMalloc(&g[i].dRes, sizeof(double) * g[i].N * g[i].N);checkCUDAError("Alloc error d_Res");
  cudaMalloc(&g[i].dUn, sizeof(double) * g[i].N * g[i].N);checkCUDAError("Alloc error d_Un");
  cudaMalloc(&g[i].df, sizeof(double) * g[i].N * g[i].N);checkCUDAError("Alloc error d_f");
  cudaDeviceSynchronize();
	
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
 for (size_t i = 0; i < g[0].N; i++){
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
 cudaMemcpy(&g[0].df[i*g[0].N], g[0].f[i],g[0].N*sizeof(double),cudaMemcpyHostToDevice);
 cudaMemcpy(&g[0].dU[i*g[0].N], g[0].U[i],g[0].N*sizeof(double),cudaMemcpyHostToDevice);
 }

 cudaMemset(g[0].dUn, 0, g[0].N*g[0].N*sizeof(double));
 cudaMemset(g[0].dRes, 0, g[0].N*g[0].N*sizeof(double));

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
  free(g[i].Un);
  free(g[i].f);
  free(g[i].Res);
  cudaFree(g[i].dU);  
  cudaFree(g[i].dUn);
  cudaFree(g[i].df);
  cudaFree(g[i].dRes);
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

