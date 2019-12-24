#pragma once

#include <map>
#include <mpi.h>

#include "def.h"
#include "index.h"


// selection of matrix 
struct Sel {
  std::vector<Real> a;  // matrix elements
  std::vector<Size> gj; // global indices of columns
  std::vector<Size> i;  // local indices of rows
};

// Traverse matrix, multiply local elements, collect remote.
// r: local rank
// Returns:
// ss[re]: selection of elements to get from remote rank re
std::map<int, Sel> Mul_Traverse(
    const Matr& a, const Vect& u, int r, Vect& b) {
  std::map<int, Sel> ss; // result

  // loop over rows
  for (Size i = 0; i < a.n; ++i) {
    // loop over matrix elements
    for (Size k = a.ki[i]; k < a.ki[i + 1]; ++k) {
      int re = GlbToRank(a.gjk[k]);  // rank
      if (re == r) {  // local
        b.v[i] += a.a[k] * u.v[GlbToLoc(a.gjk[k])];
      } else {  // remote
        auto& s = ss[re];
        s.a.push_back(a.a[k]);
        s.gj.push_back(a.gjk[k]);
        s.i.push_back(i);
      }
    }
  }

  return ss;
}

// Returns the number of messages to receive.
// ss: output of Mul_Traverse
int Mul_GetNumMsg(const std::map<int, Sel>& ss, MPI_Comm comm) {
  int r, nr;
  MPI_Comm_rank(comm, &r);
  MPI_Comm_size(comm, &nr);

  std::vector<int> m(nr, 0); // number of messages to send

  // loop over ranks
  for (auto& p : ss) {
    int re = p.first; // remote rank
    ++m[re];
  }

  // sum over all ranks
  MPI_Allreduce(MPI_IN_PLACE, m.data(), m.size(), MPI_INT, MPI_SUM, comm);

  return m[r];
}

// Send and receive indices.
// nm: number of messages to receive
// Returns:
// dd[re]: global indices requested by rank re
std::map<int, std::vector<Size>> Mul_XchgIdx(
    const std::map<int, Sel>& ss, int nm, MPI_Comm comm) {
  std::map<int, std::vector<Size>> dd; // result

  int tag = 0;

  std::vector<MPI_Request> ee(ss.size());

  // send indices
  int k = 0;
  for (auto& p : ss) {
    int re = p.first;   // remote rank
    auto& s = p.second; // selection of rows
    MPI_Isend(s.gj.data(), s.gj.size(), MS, re, tag, comm, &ee[k]);
    ++k;
  }

  // receive indices
  for (int i = 0; i < nm; ++i) {
    MPI_Status st;
    MPI_Probe(MPI_ANY_SOURCE, tag, comm, &st);
    int re = st.MPI_SOURCE; // remote rank
    int c; // count
    MPI_Get_count(&st, MS, &c);
    auto& d = dd[re]; // create buffer
    d.resize(c);
    MPI_Recv(d.data(), d.size(), MS, re, tag, comm, MPI_STATUS_IGNORE);
  }

  MPI_Waitall(ee.size(), ee.data(), MPI_STATUSES_IGNORE);

  return dd;
}

// Send and receive values, append remote to product 
// ss: output of Mul_Traverse
// dd: output of Mul_XchgIdx
// u: vector to multiply
// b: product
void Mul_XchgVal(const std::map<int, Sel>& ss,
    const std::map<int, std::vector<Size>>& dd, 
    const Vect& u, Vect& b, MPI_Comm comm) {
  std::vector<MPI_Request> ee(dd.size());
  std::vector<std::vector<Real>> vv(dd.size()); // buffer for values to send

  int tag = 1;

  // send values
  Size k = 0;
  for (auto& p : dd) {
    int re = p.first; // remote rank
    auto& d = p.second; // indices
    auto& v = vv[k]; // create buffer
    // loop over indices
    for (auto gj : d) {
      v.push_back(u.v[GlbToLoc(gj)]);
    }
    MPI_Isend(v.data(), v.size(), MR, re, tag, comm, &ee[k]);
    ++k;
  }

  // receive values
  for (auto& p : ss) {
    int re = p.first; // remote rank
    auto& s = p.second; // selection 
    Size n = s.gj.size();
    std::vector<Real> v(n);
    MPI_Recv(v.data(), v.size(), MS, re, tag, comm, MPI_STATUS_IGNORE);
    // append to result
    for (Size k = 0; k < v.size(); ++k) {
      b.v[s.i[k]] += s.a[k] * v[k];
    }
  }

  MPI_Waitall(ee.size(), ee.data(), MPI_STATUSES_IGNORE);
}

// Multiplies matrix and vector.
Vect Mul(const Matr& a, const Vect& u, MPI_Comm comm) {
  Vect b; // result
  b.v.resize(a.n, 0);

  int r; // rank
  MPI_Comm_rank(comm, &r);

  // traverse matrix, multiply local, collect remote
  std::map<int, Sel> ss = Mul_Traverse(a, u, r, b);

  // number of messages to receive from remote
  int nm = Mul_GetNumMsg(ss, comm);

  // exchange indices
  std::map<int, std::vector<Size>> dd = Mul_XchgIdx(ss, nm, comm);

  MPI_Barrier(comm);

  // exchange values, append to result
  Mul_XchgVal(ss, dd, u, b, comm);

  return b;
}
