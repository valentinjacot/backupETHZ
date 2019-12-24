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

#define CHECK_NOEMPTY(SIZE) do { if(SIZE <= 0) { \
  printf("Requested empty layer. Aborting.\n"); abort(); } } while (0)

#define CHECK_NOINPUT() do { if(layers.size() == 0 || nInputs == 0) { \
  printf("Missing input layer. Aborting.\n"); abort(); } } while (0)

#define CHECK_INPOUT(NINP) do { if(layers.back()->size not_eq NINP) {       \
  printf("Mismatch: input size (%d) and prev. layer size (%d). Aborting\n", \
  NINP, layers.back()->size); abort(); } } while (0)

#define CHECKOUT_NOPARAM() do {                                              \
    /* check that layer/weight/grad counters are correct so far */           \
    assert(params.size() == layers.size() && grads.size() == layers.size()); \
    layers.push_back(l);                                                     \
    params.push_back(nullptr); /* layer does not need params */              \
    grads.push_back(nullptr);  /* layer does not need params' grads */       \
  } while(0)

#define CHECKOUT_ALLOCPARAM() do {                                           \
    /* check that layer/weight/grad counters are correct so far */           \
    assert(params.size() == layers.size() && grads.size() == layers.size()); \
    layers.push_back(l);                                                     \
    params.push_back(l->allocate_params());                                  \
    grads.push_back(l->allocate_params()); /* grads same size as params */   \
    l->init(gen, params); /* initialize params' values */                    \
  } while(0)


template<int size>
void Network::addInput()
{
  CHECK_NOEMPTY(size);
  if(layers.size() != 0) {
    printf("Multiple input layers. Aborting.\n"); abort();
  }
  assert(nInputs == 0);

  Layer * l = new Input_Layer<size>();
  nInputs = l->size;
  // input layer has no parameters and therefore no gradient of parameters:
  CHECKOUT_NOPARAM();
}

template<int inpSize, int size>
void Network::addLinear(const std::string fname)
{
  CHECK_NOINPUT();
  CHECK_NOEMPTY(size);
  CHECK_INPOUT(inpSize);

  auto l = new LinearLayer<size, inpSize>(layers.size());
  nOutputs = l->size;
  CHECKOUT_ALLOCPARAM();
}

template<int size>
void Network::addSoftMax()
{
  CHECK_NOINPUT();
  CHECK_NOEMPTY(size);
  CHECK_INPOUT(size);

  auto l = new SoftMaxLayer<size>(layers.size());
  nOutputs = l->size;
  CHECKOUT_NOPARAM();
}

template<int size>
void Network::addLReLu()
{
  CHECK_NOINPUT();
  CHECK_NOEMPTY(size);
  CHECK_INPOUT(size);

  auto l = new LReLuLayer<size>(layers.size());
  nOutputs = l->size;
  CHECKOUT_NOPARAM();
}

template<int size>
void Network::addTanh()
{
  CHECK_NOINPUT();
  CHECK_NOEMPTY(size);
  CHECK_INPOUT(size);

  auto l = new TanhLayer<size>(layers.size());
  nOutputs = l->size;
  CHECKOUT_NOPARAM();
}


template < int InX, int InY, int InC, int KnX, int KnY, int KnC,
           int  Sx, int  Sy, int  Px, int  Py, int OpX, int OpY >
void Network::addConv2D(const std::string fname)
{
  CHECK_NOINPUT();
  CHECK_NOEMPTY(OpX * OpY * KnC);
  CHECK_INPOUT(InX * InY * InC);

  {
    auto l = new Im2MatLayer<InX,InY,InC, KnX,KnY,KnC, Sx,Sy, Px,Py, OpX,OpY>(
      layers.size() );

    CHECKOUT_NOPARAM();
  }
  {
    auto l = new Conv2DLayer<InX,InY,InC, KnX,KnY,KnC, OpX,OpY>(
      layers.size());
    nOutputs = l->size;
    CHECKOUT_ALLOCPARAM();
  }
}
