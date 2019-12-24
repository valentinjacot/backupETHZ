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

struct MomentumSGD
{
  const Real eta;
  const Real normalization; // 1/batchSize
  const Real beta;
  const Real lambda;

  MomentumSGD(const Real _eta,     // Learning rate
      const int batchSize,
      const Real _beta1,
      const Real _beta2,
      const Real _lambda
  ) : eta(_eta), normalization(1./batchSize), beta(_beta1), lambda(_lambda) {}

  // perform gradient update for a parameter array:
  inline void step (
        const int size,     // parameter array's size
        Real* const param,  // parameter array
        Real* const grad,   // parameter array gradient
        Real* const mom1st, // parameter array gradient 1st moment
        Real* const mom2nd  // parameter array gradient 2nd moment (unused)
      ) const
  {
#pragma omp for schedule (dynamic, 64/sizeof(Real)) nowait
    for (int i = 0; i < size; i++)
    {
      // grad has two components: minimize loss function and L2 penalization:
      const Real G = normalization * grad[i] + lambda * param[i];
      mom1st[i] = beta * mom1st[i] - eta * G;
      param[i] = param[i] + mom1st[i];
    }
  }
};

template<typename Algorithm>
struct Optimizer
{
  Network& NET;
  const Real eta, beta_1, beta_2, lambda;
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
    for (auto& p : momentum_1st)
      _dispose_object(p);
    for (auto& p : momentum_2nd)
      _dispose_object(p);
  }

  virtual void update(const int batchSize)
  {
    assert(parms.size() == grads.size());
    assert(parms.size() == momentum_1st.size());
    assert(parms.size() == momentum_2nd.size());

    // Given some learning algorithm..
    const Algorithm algo(eta, batchSize, beta_1, beta_2, lambda);

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

    for (size_t j = 0; j < parms.size(); j++)
      if ( parms[j] not_eq nullptr ) {
        grads[j]->clearBias();
        grads[j]->clearWeight();
      }
    step++;
  }
};
