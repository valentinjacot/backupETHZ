#pragma once

#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>

#include "def.h"
#include "index.h"
#include "op.h"
#include "io.h"
#include "mul.h"

#ifndef NTFACTOR
#define NTFACTOR 1 // factor for number of time steps, applies to RunDiffusion
#endif

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

std::string GetName(Size i) {
  std::stringstream n;
  n << "u_" << std::setfill('0') << std::setw(3) << i << ".dat";
  return n.str();
}

void RunDiffusion(Vect& u, Size r, MPI_Comm comm) {
  Matr a = GetLapl(r); // Laplacian

  Dump(u, comm, GetName(0));

  const Size nt = 10 * NTFACTOR; // number of time steps
  for (Size t = 0; t < nt; ++t) {
    Vect du = Mul(a, u, comm);

    Real k = 0.25; // scaling, k <= 0.25 required for stability.
    du = Mul(du, k);
    u = Add(u, du);
  }

  Dump(u, comm, GetName(1));
}


void RunLife(Vect& u, Size r, MPI_Comm comm) {
  Matr a = GetNeighbSum(r); // sum over 3x3 neighbours

  Dump(u, comm, GetName(0));

  const Size nt = 2000; // number of time steps
  const Size nd = 200; // number of dumps
  Size fr = 0;
  for (Size t = 0; t < nt; ++t) {
    Vect s = Mul(a, u, comm);

    for (Size i = 0; i < u.v.size(); ++i) {
      int q(s.v[i] + 0.5 - u.v[i]); // round
      u.v[i] = (u.v[i] == 1. ? (q == 2 || q == 3) : (q == 3));
    }
    if ((t + 1) % (std::max<Size>(1, nt / nd)) == 0) {
      auto fn = GetName(++fr);
      if (r == 0) {
        std::cout << fn << std::endl;
      }
      Dump(u, comm, fn);
    }
  }
}

void RunWave(Vect& u, Size r, MPI_Comm comm) {
  Matr a = GetLapl(r); // Laplacian

  Dump(u, comm, GetName(0));

  Vect um = u;
  const Size nt = 1000; // number of time steps
  const Size nd = 100; // number of dumps
  Size fr = 0;
  for (Size t = 0; t < nt; ++t) {
    Vect du = Mul(a, u, comm);

    Real k = 0.25; // scaling
    std::swap(u, um);
    for (Size i = 0; i < u.v.size(); ++i) {
      u.v[i] = 2. * um.v[i] - u.v[i] + k * du.v[i];
    }
    if (t % (std::max<Size>(1, nt / nd)) == 0) {
      Dump(u, comm, GetName(++fr));
    }
  }
}
