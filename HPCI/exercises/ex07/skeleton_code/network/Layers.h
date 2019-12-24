/*
 *  Layers.h
 *
 *  Created by Guido Novati on 29.10.18.
 *  Copyright 2018 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Activations.h"

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

  virtual void init(std::mt19937& G, const std::vector<Params*>& P) const = 0;

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

  void init(std::mt19937& G, const std::vector<Params*>& P) const override {}
};
