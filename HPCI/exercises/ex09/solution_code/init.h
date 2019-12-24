#pragma once

#include <vector>
#include <cmath>

#include "def.h"
#include "index.h"

Real sqr(Real a) {
  return a * a;
}

// Returns ellipse.
// cx,cy: center
// rx,ry: semi-axes
// r: rank
Vect InitEllipse(Real cx, Real cy, Real rx, Real ry, Size r) {
  Vect u;
  for (Size i = 0; i < L; ++i) {
    Size gi = LocToGlb(i, r);
    auto xy = GlbToCoord(gi);
    Real x = Real(xy[0] + 0.5) / NX;
    Real y = Real(xy[1] + 0.5) / NY;
    Real dx = (x - cx) / rx;
    Real dy = (y - cy) / ry;
    u.v.push_back(dx * dx + dy * dy < 1. ? 1. : 0.);
  }
  return u;
}

// Returns pattern value at origin (0,0).
// x,y: coordinates
// ph: phase (0 or 1)
Real GetPattern(int x, int y, int ph) {
  // pattern size
  int sx = 5;
  int sy = 4;
  // pattern mask: lightweight spaceship LWSS, 2 phases
  std::vector<Real> p0 = {0,0,1,1,0,1,1,0,1,1,1,1,1,1,0,0,1,1,0,0};
  std::vector<Real> p1 = {0,1,1,1,1,1,0,0,0,1,0,0,0,0,1,1,0,0,1,0};
  if (x >= 0 && x < sx && y >= 0 && y < sy) {
    auto& p = (ph == 0 ? p0 : p1);
    return p[sx * y + x];
  }
  return 0.;
}

// Returns ellipse tiled by pattern.
// cx,cy: center
// rx,ry: semi-axes
// r: rank
// mrx: mirror along x
Vect InitEllipseLife(Real cx, Real cy, Real rx, Real ry, Size r, bool mrx) {
  Vect u;

  // tile size
  int tx = 9;
  int ty = 7;

  // center 
  int x0 = cx * NX;
  int y0 = cy * NY;

  for (Size i = 0; i < L; ++i) {
    Size gi = LocToGlb(i, r);
    auto xy = GlbToCoord(gi);
    // relative to center
    int x = int(xy[0]) - x0;
    int y = int(xy[1]) - y0;
    if (mrx) { // mirror
      x = -x;
    }
    // base cell (left borrom corner)
    int xb = x - (x >= 0 ? x % tx : (x + 1) % tx + tx - 1);
    int yb = y - (y >= 0 ? y % ty : (y + 1) % ty + ty - 1);
    // is inside circle
    bool q = sqr(xb / (rx * NX)) + sqr(yb / (ry * NY)) < 1.;
    // phase
    int ph = (xb + NX) / tx % 2 ? 0 : 1;
    // append
    u.v.push_back(q ? GetPattern(x - xb, y - yb, ph) : 0.);
  }
  return u;
}

// Returns ellipse modulated by sine wave.
// cx,cy: center
// rx,ry: semi-axes
// r: rank
Vect InitWave(Real cx, Real cy, Real rx, Real ry, Size r) {
  Vect u;
  for (Size i = 0; i < L; ++i) {
    Size gi = LocToGlb(i, r);
    auto xy = GlbToCoord(gi);
    Real x = Real(xy[0] + 0.5) / NX;
    Real y = Real(xy[1] + 0.5) / NY;
    Real dx = (x - cx) / rx;
    Real dy = (y - cy) / ry;
    Real d = std::sqrt(sqr(dx) + sqr(dy));
    u.v.push_back(std::cos(d * 10.) * std::exp(-d*d));
  }
  return u;
}

