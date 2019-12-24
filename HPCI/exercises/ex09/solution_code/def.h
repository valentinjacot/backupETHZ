#pragma once

#include <cstddef>
#include <vector>

using Real = double;
using Size = size_t;

const MPI_Datatype MR = MPI_DOUBLE;
const MPI_Datatype MS = MPI_UINT64_T;

// grid size
#ifndef N
#define N 64
#endif
#define NX N
#define NY N

// number of blocks
#ifndef NB
#define NB 2
#endif
#define NBX NB
#define NBY NB

// size of block
const Size BX = NX / NBX;
const Size BY = NY / NBY;
// number of points per rank
const Size L = BX * BY;

static_assert(NX % NBX == 0 && NY % NBY == 0, 
    "Grid size not divisible by number of blocks");

using VR = std::vector<Real>;
using VS = std::vector<Size>;

// Distributed matrix.
// i: local row, gj: global column, k: index in a
struct Matr {
  VR a;         // data, size nnz, a[k], k=[0..nnz)
  VS ki = {0};  // index in a, size n+1, k[i], i=[0..n]
  VS gjk;       // global column, size nnz, j[k], k=[0..nnz)
  Size n = 0;   // number of rows
};

struct Vect {
  VR v;
};

