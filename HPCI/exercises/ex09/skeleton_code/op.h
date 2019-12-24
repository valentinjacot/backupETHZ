#pragma once

#include "index.h"

// Returns identity matrix.
// r: rank
Matr GetEye(Size r) {
  Matr a;
  // loop over local
  for (Size i = 0; i < L; ++i) {
    a.a.push_back(1.);
    a.ki.push_back(a.a.size());
    a.gjk.push_back(LocToGlb(i, r));
    ++a.n;
  }
  return a;
}

// Returns shift operator: u(x+dx,y+dy).
// dx,dy: shift
// r: rank
Matr GetShift(Size dx, Size dy, Size r) {
  Matr a;
  // loop over local
  for (Size i = 0; i < L; ++i) {
    a.a.push_back(1.);
    a.ki.push_back(a.a.size());
    auto gi = LocToGlb(i, r);
    auto xy = GlbToCoord(gi);
    xy[0] = (xy[0] + dx + NX) % NX;
    xy[1] = (xy[1] + dy + NY) % NY;
    a.gjk.push_back(CoordToGlb(xy[0], xy[1]));
    ++a.n;
  }
  return a;
}

// Returns laplacian.
// r: rank
Matr GetLapl(Size r) {
  Matr a;
  // loop over local
  for (Size i = 0; i < L; ++i) {
    // xy: m:minus, p:plus
  
    Size nx = NX;
    Size ny = NY;
    auto gi = LocToGlb(i, r);
    auto xy = GlbToCoord(gi);

    assert(CoordToGlb(xy[0], xy[1]) == gi);
    Size x = xy[0];
    Size y = xy[1];

    Size xm = (x + nx - 1) % nx;
    Size xp = (x + 1) % nx;
    Size ym = (y + ny - 1) % ny;
    Size yp = (y + 1) % ny;

    Size gxm = CoordToGlb(xm, y);
    Size gxp = CoordToGlb(xp, y);
    Size gym = CoordToGlb(x, ym);
    Size gyp = CoordToGlb(x, yp);

    a.a.push_back(-4.);
    a.gjk.push_back(gi);

    a.a.push_back(1.);
    a.gjk.push_back(gxm);

    a.a.push_back(1.);
    a.gjk.push_back(gxp);

    a.a.push_back(1.);
    a.gjk.push_back(gym);

    a.a.push_back(1.);
    a.gjk.push_back(gyp);

    a.ki.push_back(a.a.size());
    ++a.n;
  }

  return a;
}


