/*
 *  Layers.h
 *
 *  Created by Guido Novati on 29.10.18.
 *  Copyright 2018 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Activations.h"
#include "Params.h"

#ifndef __STDC_VERSION__ //it should never be defined with g++
#define __STDC_VERSION__ 0
#endif
#include "cblas.h"


struct Layer
{
  const int size, ID;

  Layer(const int _size, const int _ID) : size(_size), ID(_ID) {}
  virtual ~Layer() {}

  virtual void forward(const std::vector<Activation*>& act,
                       const std::vector<Params*>& param) const=0;

  virtual void bckward(const std::vector<Activation*>& act,
                       const std::vector<Params*>& param,
                       const std::vector<Params*>& grad) const=0;

  virtual void init(std::mt19937& gen, const std::vector<Params*>& param) const=0;

  virtual void save(const std::vector<Params*>& param) const {
    if(param[ID] == nullptr) { assert(ID==0); return; } //input layer
    else param[ID]->save(std::to_string(ID));
  };

  virtual void restart(const std::vector<Params*>& param) const {
    if(param[ID] == nullptr) { assert(ID==0); return; } //input layer
    else param[ID]->save(std::to_string(ID));
  };

  Activation* allocateActivation(const unsigned batchSize) {
    return new Activation(batchSize, size);
  }
  virtual Params* allocate_params() const = 0;
};

template<int nOutputs>
struct Input_Layer: public Layer
{
  Input_Layer() : Layer(nOutputs, 0) {
    printf("(%d) Input Layer of sizes Output:%d\n", ID, nOutputs);
  }

  Params* allocate_params() const override {
    // non linear activation layers have no parameters:
    return nullptr;
  }

  void forward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param) const override {}

  void bckward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param,
               const std::vector<Params*>& grad)  const override {}

  void init(std::mt19937& gen, const std::vector<Params*>& param) const override {}
};

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
    //array of outputs from previous layer
    const Real*const inputs = act[ID-1]->output; //size is batchSize * nInputs

    //weight matrix and bias vector:
    const Real*const weight = param[ID]->weights; //size is nInputs * nOutputs
    const Real*const bias   = param[ID]->biases; //size is nOutputs

    //return matrix that contains layer's output
    Real*const output = act[ID]->output; //size is batchSize * nOutputs

    // TODO : reset layer's output with the bias

    // TODO : perform the forward step with gemm
  }


  void bckward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param,
               const std::vector<Params*>& grad)  const override
  {
    // At this point, act[ID]->dError_dOutput contins derivative of error
    // with respect to the outputs of the network.
    const Real* const deltas = act[ID]->dError_dOutput; // batchSize * nOutputs
    const Real* const inputs = act[ID-1]->output; // batchSize * nInputs
    const Real* const weight = param[ID]->weights; // nInputs * nOutputs
    const int batchSize = act[ID]->batchSize;

    // TODO: Implement BackProp to compute bias gradient: dError / dBias
    {
      Real* const grad_B = grad[ID]->biases; // size nOutputs
    }

    // TODO: Implement BackProp to compute weight gradient: dError / dWeights
    {
      Real* const grad_W = grad[ID]->weights; // size nInputs * nOutputs
    }

    // TODO: Implement BackProp to compute dEdO of prev layer
    {
      Real* const errinp = act[ID-1]->dError_dOutput; // batchSize * nInputs
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
    //array of outputs from previous layer:
    const Real*const inputs = act[ID-1]->output; //size is batchSize * size
    //return matrix that contains layer's output
    Real*const output = act[ID]->output; //size is batchSize * size

    // TODO : Compute output
  }

  void bckward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param,
               const std::vector<Params*>& grad)  const override
  {
    const int batchSize = act[ID]->batchSize;

    const Real* const inputs = act[ID-1]->output; //size is batchSize * size
    const Real* const output = act[ID]->output; //size is batchSize * size

    // this matrix already contains dError / dOutput for this layer:
    const Real* const deltas = act[ID]->dError_dOutput; // batchSize * size

    //return matrix that contains dError / dOutput for previous layer:
    Real* const errinp = act[ID-1]->dError_dOutput; //size is batchSize * size

    // TODO : compute dError / dOutput for previous layer
  }

  // no parameters to initialize;
  void init(std::mt19937& gen, const std::vector<Params*>& param) const override {}

};
