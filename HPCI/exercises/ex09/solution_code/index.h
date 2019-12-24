#pragma once

#include <array>

#include "def.h"


// Converts coordinates to global index.
Size CoordToGlb(Size x, Size y) {
  return NX * y + x;
}

// Converts global index to coordinates.
std::array<Size, 2> GlbToCoord(Size gi) {
  Size x = gi % NX;
  Size y = gi / NX;
  return {x, y};
}

// Converts global index to rank.
// gi: global index
Size GlbToRank(Size gi) {
  auto xy = GlbToCoord(gi);
  // block
  Size bx = xy[0] / BX;
  Size by = xy[1] / BY;
  return NBX * by + bx;
}

// Converts global index to local.
// gi: global index
Size GlbToLoc(Size gi) {
  auto xy = GlbToCoord(gi);
  // offset
  Size ox = xy[0] % BX;
  Size oy = xy[1] % BY;
  return BX * oy + ox;
}

// Converts local index to global.
// i: local index
// r: rank
Size LocToGlb(Size i, Size r) {
  // block
  Size bx = r % NBX;
  Size by = r / NBX;
  // offset 
  Size ox = i % BX;
  Size oy = i / BX;
  // coord
  Size x = bx * BX + ox;
  Size y = by * BY + oy;
  return CoordToGlb(x, y);
}
