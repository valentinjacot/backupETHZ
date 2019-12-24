/*
 *  Network_buildFunctions.h
 *
 *  Created by Guido Novati on 30.10.18.
 *  Copyright 2018 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "Layer_Conv2D.h"
#include "Layer_Im2Mat.h"
#include "Layer_Functions.h"
#include "Layer_Linear.h"

template<int size>
void Network::addInput()
{
  if(size<=0) { printf("Requested empty layer. Aborting.\n"); abort(); }
  if(layers.size() not_eq 0) {
    printf("Multiple input layers. Aborting.\n");
    abort();
  }
  assert(nInputs == 0);

  Layer * l = new Input_Layer<size>();
  assert(params.size() == layers.size());
  assert(grads.size() == layers.size());

  nInputs = size;
  layers.push_back(l);

  // input layer has no parameters and therefore no gradient of parameters:
  params.push_back(nullptr);
  grads.push_back(nullptr);
}

template<int nInputs, int size>
void Network::addLinear(const std::string fname)
{
  if(layers.size() == 0) {
    printf("Missing input layer. Aborting.\n");
    abort();
  }
  if(size <= 0) {
    printf("Requested empty layer. Aborting.\n");
    abort();
  }
  if(layers.back()->size not_eq nInputs) {
    printf("Mismatch between input size (%d) and previous layer size (%d). Aborting\n",
        nInputs, layers.back()->size);
    abort();
  }

  auto l = new LinearLayer<size, nInputs>(layers.size());
  assert(params.size() == layers.size());
  assert(grads.size() == layers.size());

  layers.push_back(l);
  nOutputs = l->size;

  // allocate also parameters and their gradient (which by definition has same size)
  params.push_back(l->allocate_params());
  grads.push_back(l->allocate_params());

  l->init(gen, params);
}

template<int size>
void Network::addSoftMax()
{
  if(layers.size() == 0) {
    printf("Missing input layer. Aborting.\n");
    abort();
  }
  if(layers.back()->size not_eq size) {
    printf("Mismatch between SoftMax size (%d) and previous layer size (%d). Aborting\n",
        size, layers.back()->size);
    abort();
  }

  auto l = new SoftMaxLayer<size>(layers.size());
  assert(params.size() == layers.size());
  assert(grads.size() == layers.size());

  layers.push_back(l);
  nOutputs = l->size;
  params.push_back(nullptr);
  grads.push_back(nullptr);
}

template<int size>
void Network::addLReLu()
{
  if(layers.size() == 0) {
    printf("Missing input layer. Aborting.\n");
    abort();
  }
  if(layers.back()->size not_eq size) {
    printf("Mismatch between LReLu size (%d) and previous layer size (%d). Aborting\n",
        size, layers.back()->size);
    abort();
  }

  auto l = new LReLuLayer<size>(layers.size());
  assert(params.size() == layers.size());
  assert(grads.size() == layers.size());

  layers.push_back(l);
  nOutputs = l->size;
  params.push_back(nullptr);
  grads.push_back(nullptr);
}


template < int InX, int InY, int InC, int KnX, int KnY, int KnC,
           int  Sx, int  Sy, int  Px, int  Py, int OpX, int OpY >
void Network::addConv2D(const std::string fname)
{
  if(layers.size() == 0) {
    printf("Missing input layer. Aborting.\n");
    abort();
  }
  if(OpX * OpY * KnC <= 0) {
    printf("Requested empty layer. Aborting.\n");
    abort();
  }
  if(layers.back()->size not_eq InX * InY * InC) {
    printf("Mismatch between input size (%d) and previous layer size (%d). Aborting\n",
        InX * InY * InC, layers.back()->size);
    abort();
  }

  {
    auto l = new Im2MatLayer<InX,InY,InC, KnX,KnY,KnC, Sx,Sy, Px,Py, OpX,OpY>(
      layers.size() );
    layers.push_back(l);
    params.push_back(nullptr);
    grads.push_back(nullptr);
  }
  {
    auto l = new Conv2DLayer<InX,InY,InC, KnX,KnY,KnC, OpX,OpY>(
      layers.size());

    layers.push_back(l);
    nOutputs = l->size;
    params.push_back(l->allocate_params());
    grads.push_back(l->allocate_params());

    l->init(gen, params);
  }
}
