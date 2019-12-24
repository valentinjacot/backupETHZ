/*
 *  Layers.h
 *
 *  Created by Guido Novati on 29.10.18.
 *  Copyright 2018 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Layers.h"

template<int nOutputs>
struct LReLuLayer: public Layer
{
  static constexpr Real leak = 0.1;

  Params* allocate_params() const override {
    // non linear activation layers have no parameters:
    return nullptr;
  }

  LReLuLayer(const int _ID) : Layer(nOutputs, _ID) {
    printf("(%d) LReLu Layer of size Output:%d\n", ID, nOutputs);
  }

  void forward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param) const override
  {
    const int batchSize = act[ID]->batchSize;
    const Real*const inputs = act[ID-1]->output; //size is batchSize * nOutputs
    Real*const output = act[ID]->output; //size is batchSize * nOutputs

    for (int i=0; i<batchSize * size; i++) output[i] = eval(inputs[i]);
  }

  void bckward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param,
               const std::vector<Params*>& grad)  const override
  {
    const int batchSize = act[ID]->batchSize;
    const Real* const inputs = act[ID-1]->output; //size is batchSize * size
    //const Real* const output = act[ID]->output; //size is batchSize * size
    const Real* const deltas = act[ID]->dError_dOutput; // batchSize * size
    Real* const errinp = act[ID-1]->dError_dOutput; //size is batchSize * size

    for (int i=0; i<batchSize * size; i++)
      errinp[i] = deltas[i] * evalDiff(inputs[i]);
  }

  // no parameters to initialize;
  void init(std::mt19937& G, const std::vector<Params*>& P) const override {}

  static inline Real eval(const Real in) {
    return in > 0 ? in : leak * in;
  }
  static inline Real evalDiff(const Real in) {
    return in > 0 ? 1 : leak;
  }
};

template<int nOutputs>
struct SoftMaxLayer: public Layer
{
  Params* allocate_params() const override {
    // non linear activation layers have no parameters:
    return nullptr;
  }

  SoftMaxLayer(const int _ID) : Layer(nOutputs, _ID) {
    printf("(%d) SoftMax Layer of size Output:%d\n", ID, nOutputs);
  }

  void forward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param) const override
  {
    const int batchSize = act[ID]->batchSize;

    for(int i=0; i<batchSize; i++)
    {
      Real*const inputs = act[ID-1]->output + i*nOutputs; //batchSize * size
      Real*const output = act[ID]->output + i*nOutputs; //batchSize * size
      Real norm = 0;
      for(int j=0; j<nOutputs; j++) {
        inputs[j] = std::exp(inputs[j]);
        norm += inputs[j];
      }
      for(int j=0; j<nOutputs; j++) output[j] = inputs[j]/norm;
    }
  }

  void bckward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param,
               const std::vector<Params*>& grad)  const override
  {
    const int batchSize = act[ID]->batchSize;

    for(int i=0; i<batchSize; i++)
    {
      const Real* const deltas = act[ID]->dError_dOutput +i*nOutputs;
      const Real* const inputs = act[ID-1]->output +i*nOutputs;
      //const Real* const output = act[ID]->output +i*nOutputs;
      Real* const errinp = act[ID-1]->dError_dOutput +i*nOutputs;
      Real norm = 0;

      for(int j=0; j<nOutputs; j++) norm += inputs[j];
      const Real invN = 1 / norm;
      for(int j=0; j<nOutputs; j++)
        for(int k=0; k<nOutputs; k++)
          errinp[k] += deltas[j] * inputs[k] *((k==j) -inputs[j]*invN)*invN;
    }
  }

  // no parameters to initialize;
  void init(std::mt19937& G, const std::vector<Params*>&P) const override {}
};
