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
    //Each matrix has size is batchSize * size:
    const Real*const __restrict__ inputs = act[ID-1]->output;
    Real*const __restrict__ output = act[ID]->output;

    #pragma omp parallel for schedule(static)
    for (int i=0; i<batchSize * size; i++) output[i] = eval(inputs[i]);
  }

  void bckward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param,
               const std::vector<Params*>& grad)  const override
  {
    const int batchSize = act[ID]->batchSize;
    //Each matrix has size is batchSize * size:
    const Real* const __restrict__ I = act[ID-1]->output;
    const Real* const __restrict__ D = act[ID]->dError_dOutput;
    Real* const __restrict__ E = act[ID-1]->dError_dOutput;

    #pragma omp parallel for schedule(static)
    for (int i=0; i<batchSize * size; i++)  E[i] = D[i] * evalDiff(I[i]);
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

    #pragma omp parallel for schedule(static)
    for(int i=0; i<batchSize; i++)
    {
      //Both output and input have size batchSize * size
      Real*const __restrict__ I = act[ID-1]->output + i*nOutputs;
      Real*const __restrict__ O = act[ID]->output + i*nOutputs;
      Real norm = 0;
      for(int j=0; j<nOutputs; j++) {
        I[j] = std::exp(I[j]); // I won't need input anymore: over-writed
        norm += I[j];
      }
      for(int j=0; j<nOutputs; j++) O[j] = I[j]/norm;
    }
  }

  void bckward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param,
               const std::vector<Params*>& grad)  const override
  {
    const int batchSize = act[ID]->batchSize;
    memset(act[ID-1]->dError_dOutput, 0, nOutputs*batchSize*sizeof(Real) );

    #pragma omp parallel for schedule(static)
    for(int i=0; i<batchSize; i++)
    {
      const Real*const __restrict__ D = act[ID]->dError_dOutput +i*nOutputs;
      const Real*const __restrict__ I = act[ID-1]->output +i*nOutputs;
      Real* const __restrict__ E = act[ID-1]->dError_dOutput +i*nOutputs;

      Real norm = 0; // re-compute normalization
      for(int j=0; j<nOutputs; j++) norm += I[j];
      const Real invN = 1 / norm;

      for(int j=0; j<nOutputs; j++)
        for(int k=0; k<nOutputs; k++)
          E[k] += D[j] * I[k] *((k==j) -I[j]*invN)*invN;
    }
  }

  // no parameters to initialize;
  void init(std::mt19937& G, const std::vector<Params*>&P) const override {}
};


template<int nOutputs>
struct TanhLayer: public Layer
{
  Params* allocate_params() const override {
    // non linear activation layers have no parameters:
    return nullptr;
  }

  TanhLayer(const int _ID) : Layer(nOutputs, _ID) {
    printf("(%d) Tanh Layer of size Output:%d\n", ID, nOutputs);
  }

  void forward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param) const override
  {
    const int batchSize = act[ID]->batchSize;
    //array of outputs from previous layer, size is batchSize * size:
    const Real*const __restrict__ inputs = act[ID-1]->output;
    //return matrix that contains layer's output, same size
    Real*const __restrict__ output = act[ID]->output;

    #pragma omp parallel for schedule(static)
    for (int i=0; i<batchSize * size; i++) output[i] = std::tanh(inputs[i]);
  }

  void bckward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param,
               const std::vector<Params*>& grad)  const override
  {
    const int batchSize = act[ID]->batchSize;

    //const Real* const inputs = act[ID-1]->output; //size is batchSize * size
    const Real* const __restrict__ output = act[ID]->output;
    // this matrix already contains dError / dOutput for this layer:
    const Real* const __restrict__ deltas = act[ID]->dError_dOutput;
    //return matrix that contains dError / dOutput for previous layer:
    Real* const __restrict__ errinp = act[ID-1]->dError_dOutput;

    #pragma omp parallel for schedule(static)
    for (int i=0; i<batchSize * size; i++)
      errinp[i] = deltas[i] * (1 - output[i]*output[i]);
  }

  // no parameters to initialize;
  void init(std::mt19937& G, const std::vector<Params*>& P) const override {}
};
