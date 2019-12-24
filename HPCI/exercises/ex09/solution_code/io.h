#pragma once

#include <mpi.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>

#ifndef OUTFMT
#define OUTFMT 0
#endif


// Prints std::vector as space-separated list.
template <class T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& v) {
  std::string p = "";
  for (auto a : v) {
    o << p << a;
    p = " ";
  }
  return o;
}

// Prints matrix to stream.
void Print0(const Matr& a, std::ostream& o) {
  for (Size i = 0; i < L; ++i) {
    o << i << ":";
    for (Size k = a.ki[i]; k < a.ki[i + 1]; ++k) {
      o << " " << a.a[k] << "[" << a.gjk[k] << "]";
    }
    o << "\n";
  }
  o.flush();
}

// Prints matrix to stream, with synchronization.
void Print(const Matr& a, std::ostream& o, MPI_Comm comm) {
  int r, nr;
  MPI_Comm_rank(comm, &r);
  MPI_Comm_size(comm, &nr);

  for (int i = 0; i < nr; ++i) {
    MPI_Barrier(comm);
    if (i == r) {
      o << "rank=" << r << std::endl;
      Print0(a, o);
    }
  }
}

// Prints vector to stream.
void Print0(const Vect& u, int r, std::ostream& o) {
  for (Size i = 0; i < L; ++i) {
    auto gi = LocToGlb(i, r);
    auto xy = GlbToCoord(gi);
    o << Real(xy[0] + 0.5) / NX << " " << Real(xy[1] + 0.5) / NX 
        << " " << u.v[i] << "\n";
  }
  o.flush();
}

// Prints vector to stream, with synchronization.
void Print(const Vect& u, MPI_Comm comm, std::ostream& o) {
  int r, nr;
  MPI_Comm_rank(comm, &r);
  MPI_Comm_size(comm, &nr);

  for (int i = 0; i < nr; ++i) {
    MPI_Barrier(comm);
    if (i == r) {
      Print0(u, r, o);
    }
  }
}

// Writes vector to file.
// fn: filename
void Write(const Vect& u, MPI_Comm comm, std::string fn) {
  int r, nr;
  MPI_Comm_rank(comm, &r);
  MPI_Comm_size(comm, &nr);

  if (r == 0) {
    std::ofstream f(fn, std::ofstream::trunc);
  }
  for (int i = 0; i < nr; ++i) {
    MPI_Barrier(comm);
    if (i == r) {
      std::ofstream f(fn, std::ofstream::app);
      Print0(u, r, f);
    }
  }
}

// Writes vector to file gathering data on root.
// fn: filename
void WriteMpiGather(const Vect& u, MPI_Comm comm, std::string fn) {
  int cr, cs;
  MPI_Comm_rank(comm, &cr);
  MPI_Comm_size(comm, &cs);

  if (cr == 0) {
    std::vector<int> rc(cs, 0); // recvcounts
    std::vector<int> ds(cs + 1, 0); // displs

    // gather recvcounts
    int s = u.v.size();
    MPI_Gather(&s, 1, MPI_INT, rc.data(), 1, MPI_INT, 0, comm);

    // calc displs
    for (int r = 0; r < cs; ++r) {
      ds[r + 1] = ds[r] + rc[r];
    }

    std::vector<Real> g(NX * NY);

    MPI_Gatherv(u.v.data(), u.v.size(), MR, 
        g.data(), rc.data(), ds.data(), MR, 0, comm);

    std::ofstream o(fn);
    for (int r = 0; r < cs; ++r) {
      for (int i = 0; i < rc[r]; ++i) {
        Size gi = LocToGlb(i, r);
        auto xy= GlbToCoord(gi);
        o << Real(xy[0] + 0.5) / NX << " " << Real(xy[1] + 0.5) / NX 
            << " " << g[ds[r] + i] << "\n";
      }
    }
  } else {
    int s = u.v.size();
    MPI_Gather(&s, 1, MPI_INT, nullptr, 0, MPI_INT, 0, comm);
    MPI_Gatherv(u.v.data(), u.v.size(), MR, 
        nullptr, nullptr, nullptr, MR, 0, comm);
  }
}

// Writes vector to file using MPI IO.
// fn: filename
void WriteMpiIo(const Vect& u, MPI_Comm comm, std::string fn) {
  int cr;
  MPI_Comm_rank(comm, &cr);

  // generate string
  std::stringstream st;
  Print0(u, cr, st);
  std::string s = st.str();

  MPI_File f;

  MPI_Offset ls = s.size();  // length on current rank
  MPI_Offset o = 0;  // offset
  MPI_Exscan(&ls, &o, 1, MPI_OFFSET, MPI_SUM, comm);

  MPI_File_open(comm, fn.c_str(), 
      MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f);

  MPI_File_write_at_all(f, o, s.data(), s.size(), 
      MPI_CHAR, MPI_STATUS_IGNORE);

  MPI_File_close(&f);
}

// Writes vector to file.
// fn: filename
void WriteBin(const Vect& u, MPI_Comm comm, std::string fn) {
  int cr, cs;
  MPI_Comm_rank(comm, &cr);
  MPI_Comm_size(comm, &cs);

  if (cr == 0) {
    std::vector<int> rc(cs, 0); // recvcounts
    std::vector<int> ds(cs + 1, 0); // displs

    // gather recvcounts
    int s = u.v.size();
    MPI_Gather(&s, 1, MPI_INT, rc.data(), 1, MPI_INT, 0, comm);

    // calc displs
    for (int r = 0; r < cs; ++r) {
      ds[r + 1] = ds[r] + rc[r];
    }

    std::vector<Real> g(NX * NY); // concatenation
    std::vector<Real> gl(NX * NY); // data with global index

    MPI_Gatherv(u.v.data(), u.v.size(), MR, 
        g.data(), rc.data(), ds.data(), MR, 0, comm);

    // reorder with global index
    for (int r = 0; r < cs; ++r) {
      for (int i = 0; i < rc[r]; ++i) {
        Size gi = LocToGlb(i, r);
        gl[gi] = g[ds[r] + i];
      }
    }

    // open file in binary mode
    std::ofstream o(fn, std::ios::binary);
    uint32_t nx = NX;
    uint32_t ny = NY;
    o.write((char*)&nx, sizeof(nx));
    o.write((char*)&ny, sizeof(ny));
    o.write((char*)gl.data(), sizeof(gl[0]) * gl.size());
  } else {
    int s = u.v.size();
    MPI_Gather(&s, 1, MPI_INT, nullptr, 0, MPI_INT, 0, comm);
    MPI_Gatherv(u.v.data(), u.v.size(), MR, 
        nullptr, nullptr, nullptr, MR, 0, comm);
  }
}

// Plots vector with ASCII output.
// fn: filename
void Plot(const Vect& u, MPI_Comm comm, std::string fn) {
  Write(u, comm, fn);

  int cr;
  MPI_Comm_rank(comm, &cr);
  if (cr == 0) {
    std::system(("./plot " + fn + " ; rm " + fn).c_str());
  }
}

// Plots vector with binary output.
// fn: filename
void PlotBin(const Vect& u, MPI_Comm comm, std::string fn) {
  WriteBin(u, comm, fn);

  int cr;
  MPI_Comm_rank(comm, &cr);
  if (cr == 0) {
    std::system(("./plotbin " + fn + " ; rm " + fn).c_str());
  }
}


void Dump0(const Vect& u, MPI_Comm comm, std::string fn, int fmt) {
  switch (fmt) {
    case 0: break;
    case 1: Write(u, comm, fn); break;
    case 2: WriteMpiGather(u, comm, fn); break;
    case 3: WriteMpiIo(u, comm, fn); break;
    case 4: WriteBin(u, comm, fn); break;
    case 5: Plot(u, comm, fn); break;
    case 6: PlotBin(u, comm, fn); break;
    default: assert(false);
  }
}

void Dump(const Vect& u, MPI_Comm comm, std::string fn) {
  Dump0(u, comm, fn, OUTFMT);
}

