#pragma once

#include <cstddef>
#include <vector>

using Real = double;
using Size = size_t;

const MPI_Datatype MS = MPI_UNSIGNED_LONG;
const MPI_Datatype MR = MPI_DOUBLE;

// grid size
const Size N = 64;
const Size NX = N;
const Size NY = N;
// number of blocks
const Size NBX = 2;
const Size NBY = 2;
// size of block
const Size BX = NX / NBX;
const Size BY = NY / NBY;
// number of points per rank
const Size L = BX * BY;

static_assert(NX % NBX == 0 && NY % NBY == 0, 
    "Grid size not divisible by number of blocks");

using VR = std::vector<Real>;
using VI = std::vector<Size>;

// Distributed matrix.
// i: local row, gj: global column, k: index in a
struct Matr {
  VR a;         // data, size nnz, a[k], k=[0..nnz)
  VI ki = {0};  // index in a, size n+1, k[i], i=[0..n]
  VI gjk;       // global column, size nnz, j[k], k=[0..nnz)
  Size n = 0; // number of equations
};

struct Vect {
  VR v;
};

