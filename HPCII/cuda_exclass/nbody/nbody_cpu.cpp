/**********************************************************************/
// An unoptimized Naive N-Body solver for Gravity Simulations         //
// G is assumed to be 1.0                                             //
// Course Material for HPCSE-II, Spring 2019, ETH Zurich              //
// Authors: Sergio Martin                                             //
// License: Use if you like, but give us credit.                      //
/**********************************************************************/

#include <stdio.h>
#include <math.h>
#include <chrono>

int main(int argc, char* argv[])
{
 size_t N0 = 32;
 size_t N  = N0*N0*N0;

 // Initializing N-Body Problem

 double* xPos   = (double*) calloc (N, sizeof(double));
 double* yPos   = (double*) calloc (N, sizeof(double));
 double* zPos   = (double*) calloc (N, sizeof(double));
 double* xFor   = (double*) calloc (N, sizeof(double));
 double* yFor   = (double*) calloc (N, sizeof(double));
 double* zFor   = (double*) calloc (N, sizeof(double));
 double* mass   = (double*) calloc (N, sizeof(double));

 size_t current = 0;
 for (size_t i = 0; i < N0; i++)
 for (size_t j = 0; j < N0; j++)
 for (size_t k = 0; k < N0; k++)
 {
  xPos[current] = i;
  yPos[current] = j;
  zPos[current] = k;
  mass[current] = 1.0;
  xFor[current] = 0.0;
  yFor[current] = 0.0;
  zFor[current] = 0.0;
  current++;
 }

 // Running Force-calculation kernel
 auto startTime = std::chrono::system_clock::now();

 for (size_t i = 0; i < N; i++)
 for (size_t j = 0; j < N; j++) if (j != i)
 {
  double xDist = xPos[i] - xPos[j];
  double yDist = yPos[i] - yPos[j];
  double zDist = zPos[i] - zPos[j];
  double r     = sqrt(xDist*xDist + yDist*yDist + zDist*zDist);
  xFor[i] += xDist*mass[i]*mass[j] / (r*r*r);
  yFor[i] += yDist*mass[i]*mass[j] / (r*r*r);
  zFor[i] += zDist*mass[i]*mass[j] / (r*r*r);
 }

 auto endTime = std::chrono::system_clock::now();

 double forceChecksum = 0.0;
 for (size_t i = 0; i < N; i++) forceChecksum += xFor[i] + yFor[i] + zFor[i];
 if (fabs(forceChecksum) > 0.00001) { printf("Verification Failed: Forces are not conserved! Sum: %.10f\n", forceChecksum); exit(-1); }
 printf("Verification Passed! Time: %.8fs\n", std::chrono::duration<double>(endTime-startTime).count());
 return 0;
}
