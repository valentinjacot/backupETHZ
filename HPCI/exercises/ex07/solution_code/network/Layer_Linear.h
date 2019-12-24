/*
 *  Layers.h
 *
 *  Created by Guido Novati on 29.10.18.
 *  Copyright 2018 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Layers.h"

template<int nOutputs, int nInputs>
struct LinearLayer: public Layer
{
  Params* allocate_params() const override {
    // Allocate params: weight of size nInputs*nOutputs, bias of size nOutputs
    return new Params(nInputs*nOutputs, nOutputs);
  }

  LinearLayer(const int _ID) : Layer(nOutputs, _ID)
  {
    printf("(%d) Linear Layer of Input:%d Output:%d\n", ID, nInputs, nOutputs);
    assert(nOutputs>0 && nInputs>0);
  }

  void forward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param) const override
  {
    const int batchSize = act[ID]->batchSize;
    const Real*const inputs = act[ID-1]->output; //size is batchSize * nInputs
    const Real*const weight = param[ID]->weights; //size is nInputs * nOutputs
    Real*const __restrict__ output = act[ID]->output; //batchSize * nOutputs

    {
      const Real*const __restrict__ bias   = param[ID]->biases;
      #pragma omp parallel for schedule(static)
      for(int b=0; b<batchSize; b++)
        for(int n=0; n<nOutputs; n++) output[n + b*nOutputs] = bias[n];
    }

    // TODO : perform the forward step with gemm
    gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        batchSize, nOutputs, nInputs,
        (Real)1.0, inputs, nInputs,
                   weight, nOutputs,
        (Real)1.0, output, nOutputs);
  }


  void bckward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param,
               const std::vector<Params*>& grad)  const override
  {
    // At this point, act[ID]->dError_dOutput contins derivative of error
    // with respect to the outputs of the network.
    const Real* const __restrict__ deltas = act[ID]->dError_dOutput;
    const Real* const inputs = act[ID-1]->output;
    const Real* const weight = param[ID]->weights;
    const int batchSize = act[ID]->batchSize;

    // TODO: Implement BackProp to compute bias gradient: dError / dBias
    {
      Real* const __restrict__ grad_B = grad[ID]->biases; // size nOutputs
      std::fill(grad_B, grad_B + nOutputs, 0);
      #pragma omp parallel for schedule(static, 64/sizeof(Real))
      for(int n=0; n<nOutputs; n++)
        for(int b=0; b<batchSize; b++) grad_B[n] += deltas[n + b*nOutputs];
    }

    // TODO: Implement BackProp to compute weight gradient: dError / dWeights
    {
      Real* const grad_W = grad[ID]->weights;
      gemm(CblasRowMajor, CblasTrans, CblasNoTrans,
          nInputs, nOutputs, batchSize,
          (Real)1.0, inputs, nInputs,
                     deltas, nOutputs,
          (Real)0.0, grad_W, nOutputs);
    }

    // TODO: Implement BackProp to compute dEdO of previous layer
    {
      Real* const errinp = act[ID-1]->dError_dOutput;
      gemm(CblasRowMajor, CblasNoTrans, CblasTrans,
          batchSize, nInputs, nOutputs,
          (Real)1.0, deltas, nOutputs,
                     weight, nOutputs,
          (Real)0.0, errinp, nInputs);
    }
  }

  void init(std::mt19937& gen, const std::vector<Params*>& param) const override
  {
    assert(param[ID] not_eq nullptr);
    // get pointers to layer's weights and bias
    Real *const W = param[ID]->weights, *const B = param[ID]->biases;
    assert(param[ID]->nWeights == nInputs*size && param[ID]->nBiases == size);

    // initialize weights with Xavier initialization
    const Real scale = std::sqrt( 6.0 / (nInputs + size) );
    std::uniform_real_distribution<Real> dis(-scale, scale);
    std::generate( W, W + nInputs*nOutputs, [&]() { return dis( gen ); } );
    std::generate( B, B + nOutputs, [&]() { return dis( gen ); } );
  }
};
