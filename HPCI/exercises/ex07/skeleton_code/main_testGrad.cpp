/*
 *  main_testGrad.cpp
 *
 *  Compare finite differences and analytical backprop.
 *
 *  Created by Guido Novati on 30.10.18.
 *  Copyright 2018 ETH Zurich. All rights reserved.
 *
 */

#include "network/Network.h"

int main (int argc, char * argv[])
{
  printf("Checking gradients\n");
  static constexpr int nOutputs = 1;
  static constexpr int nInputs  = 36;

  const Real incr = std::cbrt( std::numeric_limits<Real>::epsilon() );
  const Real tol = incr;

  Network NET;

  // prepare the network
  if(argc not_eq 2) {
    printf("Requires one arg to specify test.\n Options: lrelu, linear, conv, softmax \n");
    abort();
  }

  if(strcmp ("lrelu", argv[1]) == 0)
  {
    NET.addInput<nInputs>();
    const int nHidden  = 32;
    NET.addLinear<nInputs, nHidden>();
    NET.addLReLu<nHidden>();

    NET.addLinear<nHidden, nOutputs>();
  }
  else if (strcmp ("conv", argv[1]) == 0)
  {
    NET.addInput<nInputs>();
    NET.addConv2D<6,6,1, 3,3,3>();
    NET.addLinear<6*6*3, nOutputs>();
  }
  else if (strcmp ("linear", argv[1]) == 0)
  {
    NET.addInput<nInputs>();
    NET.addLinear<nInputs, nOutputs>();
  }
  else if (strcmp ("softmax", argv[1]) == 0)
  {
    NET.addInput<nInputs>();
    NET.addSoftMax<nInputs>();
    NET.addLinear<nInputs, nOutputs>();
  }
  else
  {
    printf("Argument not recognized.\n Options: lrelu, linear, conv, softmax\n");
    abort();
  }

  // fetch and init network gradients
  const std::vector<Params*>& grads = NET.grads;
  for(auto& p: grads) if(p not_eq nullptr) { p->clearBias(); p->clearWeight(); }

  // prepare some input
  std::vector<Real> input(nInputs);
  {
    std::normal_distribution<Real> dis_inp(0, 1);
    for(int j=0; j<nInputs; j++) input[j] = dis_inp(NET.gen);
  }

  // compute network output for given input:
  const std::vector<Real> output = NET.forward(input);

  // Compute gradients for dErr / dOut = 1 . This is equivalent to saying
  // compute the gradient of the output wrt to the network parameters.
  std::vector<Real> dErrdOut(nOutputs, 1);
  NET.bckward(dErrdOut);

  // fetch vector of layers from network:
  const std::vector<Layer *>& layers = NET.layers;
  // fetch vector of parameters arrays from network:
  const std::vector<Params*>& params = NET.params;

  // define function to perform finite differences:
  auto finDiff = [&](int outputID, int paramID, Real*paramArray, Real*gradArray)
  {
    const Real backup = paramArray[paramID];
    //1) compute output after increasing network parameter by incr
    paramArray[paramID] = backup + incr;
    const std::vector<Real> resP = NET.forward(input);

    //2) compute output after decreasing network parameter by incr
    paramArray[paramID] = backup - incr;
    const std::vector<Real> resM = NET.forward(input);

    //0) restore parameter to initial value
    paramArray[paramID] = backup;

    // Compute finite differences gradient:
    const Real diff = (resP[outputID] - resM[outputID]) / (2*incr);
    const Real err = std::fabs(gradArray[paramID]-diff); // absolute error
    if (err > tol)
      printf("Param %d - absolute error is %f (dO:%f G:%f, out:%f)\n",
          paramID, err, diff, gradArray[paramID], resP[0]);
    return err;
  };


  long double meanerr = 0, squarederr = 0, cnterr = 0;
  for (int o=0; o < nOutputs; o++) // loop over all network outputs
  {
    // loop over all layers:
    for (size_t j=0; j < layers.size(); j++)
    {
      if(params[j] == nullptr) {
        printf("Layer %lu has no parameters ... \n", j);
        continue;
      } else printf("Testing Layer %lu ... \n", j);
      // loop over all parameters of the layer:
      for (int i=0; i < params[j]->nWeights + params[j]->nBiases; i++)
      {
        int paramID;
        Real* gradArray;
        Real* paramArray;

        // if loop counter >= num weights of layer then we consider biases
        if(i < params[j]->nWeights) {
          paramArray = params[j]->weights;
          gradArray = grads[j]->weights;
          paramID = i; // goes from 0 to nWeights-1
        } else {
          paramArray = params[j]->biases;
          gradArray = grads[j]->biases;
          paramID = i - params[j]->nWeights; // goes from 0 to nBiases-1
        }

        //printf("W:%f G:%f\n", paramArray[paramID], gradArray[paramID]);
        const Real err = finDiff(o, paramID, paramArray, gradArray);

        cnterr += 1;
        meanerr += err;
        squarederr += err*err;
      }
    }
  }

  // Compute mean and stdef of errors:
  const Real stdef= std::sqrt((squarederr -meanerr*meanerr/cnterr)/cnterr);
  const Real mean = meanerr/cnterr;

  printf("Mean err:%g std:%g count:%LG.\n", mean, stdef, cnterr);
  if(mean > tol) {
    printf("Test FAILED!\n");
  } else {
    printf("Test PASSED!\n");
  }
  return mean > tol;
}
