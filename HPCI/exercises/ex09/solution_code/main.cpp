#include <cassert>
#include <mpi.h>

#include "def.h"
#include "run.h"
#include "init.h"

#ifndef CASE
#define CASE 0
#endif


int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  MPI_Comm comm = MPI_COMM_WORLD;

  int r, nr;
  MPI_Comm_rank(comm, &r);
  MPI_Comm_size(comm, &nr);

  assert(nr == NBX * NBY);

  int c = CASE;

  if (c == 0) { // diffusion
    Vect u = InitEllipse(0.5, 0.5, 0.2, 0.2, r);
    RunDiffusion(u, r, comm);
  } else if (c == 1) { // life
    Vect u0 = InitEllipseLife(0.25, 0.5, 0.2, 0.2, r, false);
    Vect u1 = InitEllipseLife(0.75, 0.5, 0.2, 0.2, r, true);
    Vect u = Add(u0, u1);
    RunLife(u, r, comm);
  } else if (c == 2) { // wave
    Vect u0 = InitWave(0.25, 0.5, 0.15, 0.15, r);
    Vect u1 = InitWave(0.6, 0.5, 0.25, 0.25, r);
    Vect u = Add(u0, u1);
    RunWave(u, r, comm);
  } else {
    assert(false);
  }

  MPI_Finalize();
}
