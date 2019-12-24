/*
 *  Params.h
 *
 *  Created by Guido Novati on 29.10.18.
 *  Copyright 2018 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "Utils.h"

struct Params
{
  const int nWeights, nBiases;
  Real* const weights; // size is nWeights
  Real* const biases;  // size is nBiases

  Params(const int _nW, const int _nB): nWeights(_nW), nBiases(_nB),
    weights(_myalloc(_nW)), biases(_myalloc(_nB))
  {
    clearBias();
    clearWeight();
  }

  ~Params() { _myfree(weights); _myfree(biases); }

  inline void clearBias() const {
    memset(biases, 0, nBiases * sizeof(Real) );
  }
  inline void clearWeight() const {
    memset(weights, 0, nWeights * sizeof(Real) );
  }

  void save(const std::string fname) const {
    FILE* wFile=fopen(("W_"+fname+".raw").c_str(),"wb");
    FILE* bFile=fopen(("b_"+fname+".raw").c_str(),"wb");
    fwrite(weights, sizeof(Real), nWeights, wFile);
    fwrite(biases,  sizeof(Real),  nBiases, bFile);
    fflush(wFile); fflush(bFile);
    fclose(wFile); fclose(bFile);
  }

  void restart(const std::string fname) {
    FILE* wFile=fopen(("W_"+fname+".raw").c_str(),"rb");
    FILE* bFile=fopen(("b_"+fname+".raw").c_str(),"rb");

    size_t wsize = fread(weights, sizeof(Real), nWeights, wFile);
    fclose(wFile);
    if((int)wsize not_eq nWeights){
      printf("Mismatch in restarted weight file %s; container:%lu read:%d. Aborting.\n", fname.c_str(), wsize, nWeights);
      abort();
    }

    size_t bsize = fread(biases, sizeof(Real),  nBiases, bFile);
    fclose(bFile);
    if((int)bsize not_eq nBiases){
      printf("Mismatch in restarted biases file %s; container:%lu read:%d. Aborting.\n", fname.c_str(), bsize, nBiases);
      abort();
    }
  }
};
