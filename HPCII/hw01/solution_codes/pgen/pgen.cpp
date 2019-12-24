#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char* argv[])
{
  FILE *fid;
  fid = fopen("problem4.in", "w");

  size_t N0 = 9; // 2^N0 + 1 elements per side
  size_t N = pow(2, N0) + 1;
  double tol = 1e-6;

  // N0
  fprintf(fid, "%lu\n", N0);

  // Tolerance
  fprintf(fid, "%22.15e\n", tol);

  double** U   = (double**) calloc (sizeof(double*), N);
  double** f   = (double**) calloc (sizeof(double*), N);
  for (int i = 0; i < N ; i++) U[i]   = (double*) calloc (sizeof(double), N);
  for (int i = 0; i < N ; i++) f[i]   = (double*) calloc (sizeof(double), N);

	// Initial Guess
//	for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) U[i][j] = 1.0;

	// Initial Guess
	for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) U[i][j] = 2.0;

	// Boundary Conditions
	for (int i = 0; i < N; i++) U[0][i]     = 0.0;
	for (int i = 0; i < N; i++) U[N-1][i]   = 0.0;
	for (int i = 0; i < N; i++) U[i][0]     = 0.0;
	for (int i = 0; i < N; i++) U[i][N-1]   = 0.0;

  // U
  for (int i = 0; i < N; i++)
  {
      for (int j = 0; j < N; j++)
      {
          fprintf(fid, "%22.15e", U[i][j]);
      }
      fprintf(fid, "\n");
  }

// // Initial Guess
//	for (int i = 0; i < N; i++) for (int j = 0; j < N; j++)
//	{
//			double c1, c2, c3, c4;
//			double h = 1.0/(N-1);
//			double x = i*h;
//			double y = j*h;
//
//			f[i][j] = 0.0;
//
//			// Heat Source: Candle 1
//			c1 = 10; // intensity
//			c2 = 0.05; // variance
//			c3 = 0.3; // x0
//			c4 = 0.3; // y0
//			f[i][j] += -(4*c1*exp( -(pow(c3 - y, 2) + pow(c4 - x, 2)) / c2 ) * (pow(c3,2) - 2*c3*y + pow(c4,2) - 2*c4*x + pow(y,2) + pow(x,2) - c2))/pow(c2,2);
//
//			// Heat Source: Candle 2
//			c1 = 20; // intensity
//			c2 = 0.05; // variance
//			c3 = 0.7; // x0
//			c4 = 0.3; // y0
//			f[i][j] += -(4*c1*exp( -(pow(c3 - y, 2) + pow(c4 - x, 2)) / c2 ) * (pow(c3,2) - 2*c3*y + pow(c4,2) - 2*c4*x + pow(y,2) + pow(x,2) - c2))/pow(c2,2);
//
//			// Heat Source: Candle 3
//			c1 = 30; // intensity
//			c2 = 0.05; // variance
//			c3 = 0.5; // x0
//			c4 = 0.7; // y0
//			f[i][j] += -(4*c1*exp( -(pow(c3 - y, 2) + pow(c4 - x, 2)) / c2 ) * (pow(c3,2) - 2*c3*y + pow(c4,2) - 2*c4*x + pow(y,2) + pow(x,2) - c2))/pow(c2,2);
//	}



	for (int i = 0; i < N; i++) for (int j = 0; j < N; j++)
	{
			double c1, c2, c3, c4;
			double h = 1.0/(N-1);
			double x = i*h;
			double y = j*h;

			f[i][j] = 0.0;

			// Heat Source: Candle 1
			c1 = 30; // intensity
			c2 = 0.02; // variance
			c3 = 0.1; // x0
			c4 = 0.8; // y0
			f[i][j] += -(4*c1*exp( -(pow(c3 - y, 2) + pow(c4 - x, 2)) / c2 ) * (pow(c3,2) - 2*c3*y + pow(c4,2) - 2*c4*x + pow(y,2) + pow(x,2) - c2))/pow(c2,2);

			// Heat Source: Candle 2
			c1 = 60; // intensity
			c2 = 0.06; // variance
			c3 = 0.7; // x0
			c4 = 0.3; // y0
			f[i][j] += -(4*c1*exp( -(pow(c3 - y, 2) + pow(c4 - x, 2)) / c2 ) * (pow(c3,2) - 2*c3*y + pow(c4,2) - 2*c4*x + pow(y,2) + pow(x,2) - c2))/pow(c2,2);

			// Heat Source: Candle 3
			c1 = 10; // intensity
			c2 = 0.08; // variance
			c3 = 0.5; // x0
			c4 = 0.5; // y0
			f[i][j] += -(4*c1*exp( -(pow(c3 - y, 2) + pow(c4 - x, 2)) / c2 ) * (pow(c3,2) - 2*c3*y + pow(c4,2) - 2*c4*x + pow(y,2) + pow(x,2) - c2))/pow(c2,2);
	}

  // F
  for (int i = 0; i < N; i++)
  {
      for (int j = 0; j < N; j++)
      {
          fprintf(fid, "%22.15e ", f[i][j]);
      }
      fprintf(fid, "\n");
  }

	fclose(fid);
}
