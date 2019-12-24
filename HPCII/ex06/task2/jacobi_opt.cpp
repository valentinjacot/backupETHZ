/**********************************************************************/
// A generic stencil-based 3D Jacobi solver                           //
// Course Material for HPCSE-II, Spring 2019, ETH Zurich              //
// Authors: Sergio Martin                                             //
// License: Use if you like, but give us credit.                      //
/**********************************************************************/

#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

typedef struct NeighborStruct {
 int rankId;
 double* recvBuffer;
 double* sendBuffer;
} Neighbor;

int main(int argc, char* argv[])
{
 MPI_Init(&argc, &argv);

 int myRank, rankCount;
 MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
 MPI_Comm_size(MPI_COMM_WORLD, &rankCount);
 bool isMainRank = myRank == 0;

 int N = 512;
 int nIters = 50;
 int nx, ny, nz;
 Neighbor X1, X0, Y0, Y1, Z0, Z1;

 int dims[3] = {0,0,0};
 MPI_Dims_create(rankCount, 3, dims);
 int px = dims[0];
 int py = dims[1];
 int pz = dims[2];

 nx = N / px; // Internal grid size (without halo) on X Dimension
 ny = N / py; // Internal grid size (without halo) on Y Dimension
 nz = N / pz; // Internal grid size (without halo) on Z Dimension

 int fx = nx + 2; // Grid size (with halo) on X Dimension
 int fy = ny + 2; // Grid size (with halo) on Y Dimension
 int fz = nz + 2; // Grid size (with halo) on Z Dimension

 double *U, *Un;
 U  = (double *)calloc(sizeof(double), fx*fy*fz);
 Un = (double *)calloc(sizeof(double), fx*fy*fz);

 int periodic[3] = {false, false, false};
 MPI_Comm gridComm; // now everyone creates a a cartesian topology
 MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periodic, true, &gridComm);

 MPI_Cart_shift(gridComm, 0, 1, &X0.rankId, &X1.rankId);
 MPI_Cart_shift(gridComm, 1, 1, &Y0.rankId, &Y1.rankId);
 MPI_Cart_shift(gridComm, 2, 1, &Z0.rankId, &Z1.rankId);

 for (int k = 0; k < fz; k++)
 for (int j = 0; j < fy; j++)
 for (int i = 0; i < fx; i++)
   U[k*fy*fx + j*fx + i] = 1;

 if (X0.rankId == MPI_PROC_NULL) for (int i = 0; i < fy; i++) for (int j = 0; j < fz; j++) U[j*fx*fy + i*fx  ] = 0;
 if (Y0.rankId == MPI_PROC_NULL) for (int i = 0; i < fx; i++) for (int j = 0; j < fz; j++) U[j*fx*fy + fx + i] = 0;
 if (Z0.rankId == MPI_PROC_NULL) for (int i = 0; i < fx; i++) for (int j = 0; j < fy; j++) U[fx*fy + j*fx + i] = 0;

 if (X1.rankId == MPI_PROC_NULL) for (int i = 0; i < fy; i++) for (int j = 0; j < fz; j++) U[j*fx*fy + i*fx + (nx+1)] = 0;
 if (Y1.rankId == MPI_PROC_NULL) for (int i = 0; i < fx; i++) for (int j = 0; j < fz; j++) U[j*fx*fy + (ny+1)*fx + i] = 0;
 if (Z1.rankId == MPI_PROC_NULL) for (int i = 0; i < fx; i++) for (int j = 0; j < fy; j++) U[(nz+1)*fx*fy + j*fx + i] = 0;

 MPI_Datatype faceXType, faceYType, faceZType;

 MPI_Type_vector(fz*fy, 1,  fx,    MPI_DOUBLE, &faceXType); MPI_Type_commit(&faceXType);
 MPI_Type_vector(fz,    fx, fx*fy, MPI_DOUBLE, &faceYType); MPI_Type_commit(&faceYType);
 MPI_Type_vector(fy,    fx, fx,    MPI_DOUBLE, &faceZType); MPI_Type_commit(&faceZType);

 MPI_Request request[12];

 MPI_Barrier(MPI_COMM_WORLD);
 double computeTime = 0;
 double packTime = 0;
 double unpackTime = 0;
 double sendTime = 0;
 double recvTime = 0;
 double waitTime = 0;
 double execTime = -MPI_Wtime();
 double t_init = MPI_Wtime();
 double t_end = MPI_Wtime();

 for (int iter=0; iter<nIters; iter++)
 {
#pragma omp parallel for
  for (int k = 1; k <= nz; k++)
  for (int j = 1; j <= ny; j++)
  for (int i = 1; i <= nx; i++)
  {
   double sum = 0.0;
   sum += U[fx*fy*k + fx*j  + i + 0]; // Central
   sum += U[fx*fy*k + fx*j  + i - 1]; // Y0
   sum += U[fx*fy*k + fx*j  + i + 1]; // Y1
   sum += U[fx*fy*k + fx*(j-1)  + i]; // X0
   sum += U[fx*fy*k + fx*(j+1)  + i]; // X1
   sum += U[fx*fy*(k-1) + fx*j  + i]; // Z0
   sum += U[fx*fy*(k+1) + fx*j  + i]; // Z1
   Un[fx*fy*k + fx*j + i] = sum/7.0;
  }
  double *temp = U; U = Un; Un = temp;

  int request_count = 0;

  t_end = MPI_Wtime();  computeTime += double(t_end-t_init); t_init = MPI_Wtime();

  MPI_Irecv(&U[0            ], 1, faceXType, X0.rankId, 1, gridComm, &request[request_count++]);
  MPI_Irecv(&U[(nx+1)       ], 1, faceXType, X1.rankId, 1, gridComm, &request[request_count++]);
  MPI_Irecv(&U[0            ], 1, faceYType, Y0.rankId, 1, gridComm, &request[request_count++]);
  MPI_Irecv(&U[fx*(ny+1)    ], 1, faceYType, Y1.rankId, 1, gridComm, &request[request_count++]);
  MPI_Irecv(&U[0            ], 1, faceZType, Z0.rankId, 1, gridComm, &request[request_count++]);
  MPI_Irecv(&U[fx*fy*(nz+1) ], 1, faceZType, Z1.rankId, 1, gridComm, &request[request_count++]);

  t_end = MPI_Wtime(); recvTime += double(t_end-t_init); t_init = MPI_Wtime();

  MPI_Isend(&U[1            ], 1, faceXType, X0.rankId, 1, gridComm, &request[request_count++]);
  MPI_Isend(&U[nx           ], 1, faceXType, X1.rankId, 1, gridComm, &request[request_count++]);
  MPI_Isend(&U[fx           ], 1, faceYType, Y0.rankId, 1, gridComm, &request[request_count++]);
  MPI_Isend(&U[fx*ny        ], 1, faceYType, Y1.rankId, 1, gridComm, &request[request_count++]);
  MPI_Isend(&U[fx*fy        ], 1, faceZType, Z0.rankId, 1, gridComm, &request[request_count++]);
  MPI_Isend(&U[fx*fy*nz     ], 1, faceZType, Z1.rankId, 1, gridComm, &request[request_count++]);

  t_end = MPI_Wtime(); sendTime += double(t_end-t_init);  t_init = MPI_Wtime();

  MPI_Waitall(request_count, request, MPI_STATUS_IGNORE);

  t_end = MPI_Wtime(); waitTime += double(t_end-t_init); t_init = MPI_Wtime();
 }

 MPI_Barrier(MPI_COMM_WORLD);
 execTime += MPI_Wtime();

 double res = 0;
 double err = 0;
 for (int k = 1; k <= nz; k++)
 for (int j = 1; j <= ny; j++)
 for (int i = 1; i <= nx; i++)
 { double r = U[k*fy*fx + j*fx + i]; err += r * r; }
 MPI_Reduce (&err, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
 res = sqrt(res/((double)(N-1)*(double)(N-1)*(double)(N-1)));

 double meanComputeTime = 0, sumComputeTime = 0;
 double meanPackTime    = 0, sumPackTime    = 0;
 double meanUnpackTime  = 0, sumUnpackTime  = 0;
 double meanSendTime    = 0, sumSendTime    = 0;
 double meanRecvTime    = 0, sumRecvTime    = 0;
 double meanWaitTime    = 0, sumWaitTime    = 0;

 MPI_Reduce(&computeTime, &sumComputeTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
 MPI_Reduce(&packTime,    &sumPackTime,    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
 MPI_Reduce(&unpackTime,  &sumUnpackTime,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
 MPI_Reduce(&sendTime,    &sumSendTime,    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
 MPI_Reduce(&recvTime,    &sumRecvTime,    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
 MPI_Reduce(&waitTime,    &sumWaitTime,    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

 meanComputeTime = sumComputeTime / rankCount;
 meanPackTime    = sumPackTime    / rankCount;
 meanUnpackTime  = sumUnpackTime  / rankCount;
 meanSendTime    = sumSendTime    / rankCount;
 meanRecvTime    = sumRecvTime    / rankCount;
 meanWaitTime    = sumWaitTime    / rankCount;

 double totalTime = meanComputeTime + meanRecvTime + meanPackTime + meanSendTime + meanWaitTime + meanUnpackTime;

 totalTime = execTime;

 if(isMainRank) {
   printf("Execution Times:\n");
   printf("  Compute:     %.4fs\n", meanComputeTime);
   printf("  MPI_Irecv:   %.4fs\n", meanRecvTime);
   printf("  MPI_Isend:   %.4fs\n", meanSendTime);
   printf("  Packing:     %.4fs\n", meanPackTime);
   printf("  Unpacking:   %.4fs\n", meanUnpackTime);
   printf("  MPI_Waitall: %.4fs\n", meanWaitTime);
   printf("---------------------\n");
   printf("Total Time:    %.4fs\n", totalTime);
   printf("L2 Norm:       %.10f\n", res);
 }

 MPI_Finalize(); return 0;
}
