/*
 *  Activations.h
 *
 *  Created by Guido Novati on 29.10.18.
 *  Copyright 2018 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "Utils.h"

struct Activation
{
  const int batchSize, layersSize;
  //matrix of size batchSize * layersSize with layer outputs:
  Real* const output;
  //matrix of same size containing:
  Real* const dError_dOutput;

  Activation(const int bs, const int ls) : batchSize(bs), layersSize(ls),
    output(_myalloc(bs*ls)), dError_dOutput(_myalloc(bs*ls))
  {
    clearErrors();
    clearOutput();
    assert(batchSize>0 && layersSize>0);
  }

  ~Activation() { _myfree(output); _myfree(dError_dOutput); }

  inline void clearOutput() {
    memset(output,         0, batchSize*layersSize*sizeof(Real));
  }
  inline void clearErrors() {
    memset(dError_dOutput, 0, batchSize*layersSize*sizeof(Real));
  }
};
