/*
 *  Network.h
 *
 *  Created by Guido Novati on 30.10.18.
 *  Copyright 2018 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Layers.h"

struct Network
{
  std::mt19937 gen;
  // Vector of layers, each defines a forward and bckward operation:
  std::vector<Layer*>  layers;
  // Vector of parameters of each layer (two vectors must have the same size)
  // Each Params contains the matrices of parameters needed by the corresp layer
  std::vector<Params*> params;
  // Vector of grads for each parameter. By definition they have the same size
  std::vector<Params*>  grads;
  // Memory space where each layer can compute its output and gradient:
  std::vector<Activation*> workspace;
  // Number of inputs to the network:
  int nInputs = 0;
  // Number of network outputs:
  int nOutputs = 0;

  Network(const int seed = 0) : gen(seed) {};

  std::vector<std::vector<Real>> forward(
        // one vector of input for each element in the mini-batch:
        const std::vector<std::vector<Real>> I,
        // layer ID at which to start forward operation:
        const size_t layerStart = 0 // (zero means compute from input to output)
    )
  {
    if(params.size()==0 || grads.size()==0 || layers.size()==0) {
      printf("Attempted to access uninitialized network. Aborting\n");
      abort();
    }

    // input is a minibatch of datapoints: one vector for each datapoint:
    const size_t batchSize = I.size();

    // allocate workspaces where we can write output of each layer
    clearWorkspace();
    workspace = allocateActivation(batchSize);

    // User can overwrite the output of any upper layer (marked by layerStart)
    // in order to see what happens if layer layerStart has a predefined output.
    // ( this allows visualizing PCA components! )
    const int inputLayerSize = workspace[layerStart]->layersSize;

    //copy input onto output of input layer:
    for (size_t b=0; b<batchSize; b++)
    {
      assert(I[b].size() == (size_t) inputLayerSize );
      // Input to the network is the output of input layer.
      // Respective workspace is a matrix of size [batchSize]x[nInputs]
      // Here we use row-major ordering: nInputs is the number of columns.
      Real* const input_b = workspace[layerStart]->output + b * inputLayerSize;
      // copy from function argument to workspace:
      std::copy(I[b].begin(), I[b].end(), input_b);
    }

    // Start from layer after input. E.g. Input layer is 0. No need to backprop
    // input layer has it has no parameters.
    for (size_t j=layerStart+1; j<layers.size(); j++)
      layers[j]->forward(workspace, params);

    // copy output into vector of vectors: one vector for each element of batch
    std::vector<std::vector<Real>> O(batchSize, std::vector<Real>(nOutputs, 0));

    for (size_t b=0; b<batchSize; b++)
    {
      assert(nOutputs == workspace.back()->layersSize);
      // network output is the output of last layer.
      // Respective workspace is a matrix of size [batchSize]x[nOutputs]
      // Here we use row-major ordering: nOutputs is the number of columns.
      Real* const output_b = workspace.back()->output + b * nOutputs;
      // copy from function argument to workspace:
      std::copy(output_b, output_b + nOutputs, O[b].begin());
    }

    return O;
  }

  void bckward(
    // vector of size of mini-batch of gradients of error wrt to network output
    const std::vector<std::vector<Real>> E,
    // layer ID at which forward operation was started:
    const size_t layerStart=0 // (zero means compute from input to output)
  ) const
  {
    //this function assumes that we already called forward propagation
    if(params.size()==0 || grads.size()==0 || layers.size()==0)
    {
      printf("Attempted to access uninitialized network. Aborting\n");
      abort();
    }
    // input is a minibatch of datapoints: one vector for each datapoint:
    const size_t batchSize = E.size();
    assert( (size_t) workspace.back()->batchSize == batchSize);

    // first, clear memory over which we'll write gradients:
    for(auto& p : workspace) p->clearErrors();

    //copy input onto output of input layer:
    for (size_t b=0; b<batchSize; b++)
    {
      assert(E[b].size() == (size_t) nOutputs);
      // Write d Err / d Out onto last layer of the network.
      // Respective workspace is a matrix of size [batchSize]x[nOutputs]
      // Here we use row-major ordering: nOutputs is the number of columns.
      Real* const errors_b = workspace.back()->dError_dOutput + b * nOutputs;
      // copy from function argument to workspace:
      std::copy(E[b].begin(), E[b].end(), errors_b);
    }

    // Backprop starts at the last layer, which computes gradient of error wrt
    // to its parameters and gradient of error wrt to it's input.
    // Last layer to backprop is the one above input layer. Eg. if layerStart=0
    // Then input layer was 0, which has no parametes and has no inputs to
    // backprp the error grad to, last layer to backprop is layer 1.
    for (size_t i = layers.size()-1; i >= layerStart + 1; i--)
      layers[i]->bckward(workspace, params, grads);
  }

  // Helper function for forward with batchsize = 1
  std::vector<Real> forward(const std::vector<Real>I, const size_t layerStart=0)
  {
    std::vector<std::vector<Real>> vecI (1, I);
    std::vector<std::vector<Real>> vecO = forward(vecI, layerStart);
    return vecO[0];
  }

  // Helper function for forward with batchsize = 1)
  void bckward(const std::vector<Real> E, const size_t layerStart = 0) const
  {
    std::vector<std::vector<Real>> vecE (1, E);
    bckward(vecE, layerStart);
  }

  ~Network() {
    for(auto& p : grads)      _dispose_object(p);
    for(auto& p : params)     _dispose_object(p);
    for(auto& p : layers)     _dispose_object(p);
    for(auto& p : workspace)  _dispose_object(p);
  }

  inline void clearWorkspace() {
    for(auto& p : workspace) _dispose_object(p);
    workspace.clear();
  }

  // Function to loop over layers and allocate workspace for network operations:
  inline std::vector<Activation*> allocateActivation(size_t batchSize) const
  {
    std::vector<Activation*> ret(layers.size(), nullptr);
    for(size_t j=0; j<layers.size(); j++)
      ret[j] = layers[j]->allocateActivation(batchSize);
    return ret;
  }

  // Function to loop over layers and allocate memory space for parameter grads:
  inline std::vector<Params*> allocateGrad() const
  {
    std::vector<Params*> ret(layers.size(), nullptr);
    for(size_t j=0; j<layers.size(); j++)
      ret[j] = layers[j]->allocate_params();
    return ret;
  }

  //////////////////////////////////////////////////////////////////////////////
  /// Functions to build the network are defined in Network_buildFunctions.h ///
  //////////////////////////////////////////////////////////////////////////////

  template<int size>
  void addInput();

  template<int nInputs, int size>
  void addLinear(const std::string fname = std::string());

  template<int nInputs>
  void addSoftMax();

  template<int nInputs>
  void addLReLu();

  template
  <
    int InX, int InY, int InC, //input image: x:width, y:height, c:channels
    int KnX, int KnY, int KnC, //filter:      x:width, y:height, c:channels
    int Sx=1, // (Stride in x) Defaults to 1: advance one pixel at the time.
    int Sy=1, // (Stride in y)
    int Px=(KnX -1)/2, // (Padding in x) Defaults to value that makes output
    int Py=(KnY -1)/2, // (Padding in y) image of the same size as input image.
    int OpX=(InX -KnX +2*Px)/Sx+1, //Out image: same number of channels as KnC.
    int OpY=(InY -KnY +2*Py)/Sy+1 //Default: uniform padding in all directions.
  >
  void addConv2D(const std::string fname = std::string());

};

#include "Network_buildFunctions.h"
