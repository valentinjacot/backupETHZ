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
    const Real*const bias   = param[ID]->biases; //size is nOutputs
    Real*const output = act[ID]->output; //size is batchSize * nOutputs

    // TODO : reset layer's workspace and add the bias
    abort();

    // TODO : perform the forward step with gemm
    abort();
  }


  void bckward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param,
               const std::vector<Params*>& grad)  const override
  {
    // At this point, act[ID]->dError_dOutput contins derivative of error
    // with respect to the outputs of the network.
    const Real* const deltas = act[ID]->dError_dOutput;
    const Real* const inputs = act[ID-1]->output;
    const Real* const weight = param[ID]->weights;
    const int batchSize = act[ID]->batchSize;

    // TODO:  Implement BackProp to compute bias gradient:
    {
      // This array will contain dError / dBias, has size nOutputs
      Real* const grad_B = grad[ID]->biases;
      abort();
    }

    // TODO: Implement BackProp to compute weight gradient
    {
      // This array will contain dError / dBias, has size nInputs * nOutputs
      Real* const grad_W = grad[ID]->weights;
      abort();
    }

    // TODO: Implement BackProp to compute dEdO of previous layer
    {
      Real* const errinp = act[ID-1]->dError_dOutput;
      abort();
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
