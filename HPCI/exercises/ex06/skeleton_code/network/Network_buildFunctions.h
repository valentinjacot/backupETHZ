/*
 *  Network_buildFunctions.h
 *
 *  Created by Guido Novati on 30.10.18.
 *  Copyright 2018 ETH Zurich. All rights reserved.
 *
 */

#pragma once

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

  if(fname not_eq std::string()) params.back()->restart(fname);
  else l->init(gen, params);
}

template<int size>
void Network::addTanh()
{
  if(layers.size() == 0) {
    printf("Missing input layer. Aborting.\n");
    abort();
  }
  if(layers.back()->size not_eq size) {
    printf("Mismatch between Tanh size (%d) and previous layer size (%d). Aborting\n",
        size, layers.back()->size);
    abort();
  }

  auto l = new TanhLayer<size>(layers.size());
  assert(params.size() == layers.size());
  assert(grads.size() == layers.size());

  layers.push_back(l);
  nOutputs = l->size;
  params.push_back(nullptr);
  grads.push_back(nullptr);
}
