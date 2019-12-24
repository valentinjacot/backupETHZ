#include <cassert>
#include <map>
#include <mpi.h>
#include <iostream>

#include "def.h"
#include "index.h"
#include "op.h"
#include "io.h"

// Adds vectors.
Vect Add(const Vect& a, const Vect& b) {
  auto r = a;
  for (Size i = 0; i < a.v.size(); ++i) {
    r.v[i] += b.v[i];
  }
  return r;
}

// Multiplies vector and scalar.
Vect Mul(const Vect& a, Real k) {
  auto r = a;
  for (auto& e : r.v) {
    e *= k;
  }
  return r;
}

// Multiplies matrix and vector.
Vect Mul(const Matr& a, const Vect& u, MPI_Comm comm) {
  // TODO 2a

  return u;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  MPI_Comm comm = MPI_COMM_WORLD;

  int r, nr;
  MPI_Comm_rank(comm, &r);
  MPI_Comm_size(comm, &nr);

  // Laplacian
  Matr a = GetLapl(r);

  // Initial vector
  Vect u;
  for (Size i = 0; i < L; ++i) {
    Size gi = LocToGlb(i, r);
    auto xy = GlbToCoord(gi);
    Real x = Real(xy[0]) / NX;
    Real y = Real(xy[1]) / NY;
    Real dx = x - 0.5;
    Real dy = y - 0.5;
    Real r = 0.2;
    u.v.push_back(dx*dx + dy*dy < r*r ? 1. : 0.);
  }

  Write(u, comm, "u0");

  const Size nt = 10; // number of time steps
  for (Size t = 0; t < nt; ++t) {
    Vect du = Mul(a, u, comm);

    Real k = 0.25; // scaling, k <= 0.25 required for stability.
    du = Mul(du, k);
    u = Add(u, du);
  }

  Write(u, comm, "u1");

  MPI_Finalize();
}
