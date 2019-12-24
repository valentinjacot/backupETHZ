/*
 *  Optimizer.h
 *
 *  Created by Guido Novati on 30.10.18.
 *  Copyright 2018 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include <fstream>
#include "Network.h"

struct Adam
{
  const Real eta, fac, beta1, beta2, lambda;
  static constexpr Real EPS = 1e-8;

  Adam(const Real _eta, const int batchSize, const Real _lambda,
    const Real _b1, const Real _b2, const Real _b1t, const Real _b2t) :
    eta(_eta * std::sqrt(1-_b2t)/(1-_b1t)), fac(1.0/batchSize),
    beta1(_b1), beta2(_b2), lambda(_lambda) {}

  // perform gradient update for a parameter array:
  inline void step (
        const int size,     // parameter array's size
        Real* const __restrict__ param,  //param. array
        Real* const __restrict__ grad,   //param. array gradient
        Real* const __restrict__ mom1st, //param. array gradient 1st moment
        Real* const __restrict__ mom2nd  //param. array gradient 2nd moment
      ) const
  {
    #pragma omp for schedule(dynamic, 64 / sizeof(Real)) nowait
    for (int i = 0; i < size; i++)
    {
      // grad has two components: minimize loss function and L2 penalization:
      const Real G = fac * grad[i] + lambda * param[i];
      mom1st[i] = beta1 * mom1st[i] + (1-beta1) * G;
      mom2nd[i] = beta2 * mom2nd[i] + (1-beta2) * G * G;
      param[i] = param[i] - eta * mom1st[i] / ( std::sqrt(mom2nd[i]) + EPS );
    }
  }
};

template<typename Algorithm>
struct Optimizer
{
  Network& NET;
  const Real eta, beta_1, beta_2, lambda;
  Real beta_1t = beta_1;
  Real beta_2t = beta_2;
  // grab the reference to network weights and parameters
  std::vector<Params*> & parms = NET.params;
  std::vector<Params*> & grads = NET.grads;

  // allocate space to store first (and if needed second) moment of the grad
  // which will allow us to learn with momentum:
  std::vector<Params*> momentum_1st = NET.allocateGrad();
  std::vector<Params*> momentum_2nd = NET.allocateGrad();

  // counter of gradient step:
  size_t step = 0;

  // Constructor:
  Optimizer(Network& NN, Real LR = .001, // Learning rate. Should be in range [1e-5 to 1e-2]
      Real L2penal = 0, // L2 penalization coefficient. Found by exploration.
      Real B1 = .900, // Momentum coefficient. Should be in range [.5 to .9]
      Real B2 = .999   // Second moment coefficient. Currently not in use.
      ) :
      NET(NN), eta(LR), beta_1(B1), beta_2(B2), lambda(L2penal) {
  }

  virtual ~Optimizer() {
    for (auto& p : momentum_1st) _dispose_object(p);
    for (auto& p : momentum_2nd) _dispose_object(p);
  }

  virtual void update(const int batchSize)
  {
    assert(parms.size() == grads.size());
    assert(parms.size() == momentum_1st.size());
    assert(parms.size() == momentum_2nd.size());

    // Given some learning algorithm..
    const Algorithm algo(eta,batchSize,lambda, beta_1,beta_2,beta_1t,beta_2t);

    // ... loop over all parameter arrays and compute the update:
    #pragma omp parallel
    for (size_t j = 0; j < parms.size(); j++)
    {
      if (parms[j] == nullptr) continue; //layer does not have parameters

      if (parms[j]->nWeights > 0)
      {
        algo.step(parms[j]->nWeights,
                  parms[j]->weights, grads[j]->weights,
                  momentum_1st[j]->weights, momentum_2nd[j]->weights);
      }

      if (parms[j]->nBiases > 0)
      {
        algo.step(parms[j]->nBiases,
                  parms[j]->biases, grads[j]->biases,
                  momentum_1st[j]->biases, momentum_2nd[j]->biases);
      }
    }

    step++;
    beta_1t *= beta_1t; if(beta_1t<NNEPS) beta_1t = 0; // prevent underflow
    beta_2t *= beta_2t; if(beta_2t<NNEPS) beta_2t = 0; // prevent underflow
  }
};
