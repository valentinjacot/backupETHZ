/*
 *  Layers.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Layers.h"

template
<
  int InX, int InY, int InC, //input image: x:width, y:height, c:color channels
  int KnX, int KnY, int KnC, //filter:      x:width, y:height, c:color channels
  int OpX, int OpY //output img: x:width, y:height, same color channels as KnC
>
struct Conv2DLayer: public Layer
{
  Params* allocate_params() const override {
    //number of kernel parameters:
    // 2d kernel size * number of inp channels * number of out channels
    const int nParams = KnY * KnX * InC * KnC;
    const int nBiases = KnC;
    return new Params(nParams, nBiases);
  }

  Conv2DLayer(const int _ID) : Layer(OpX * OpY * KnC, _ID) {
    static_assert(InX>0 && InY>0 && InC>0, "Invalid input");
    static_assert(KnX>0 && KnY>0 && KnC>0, "Invalid kernel");
    static_assert(OpX>0 && OpY>0, "Invalid outpus");
    print();
  }

  void print() {
    printf("(%d) Conv: In:[%d %d %d %d %d] F:[%d %d %d %d] Out:[%d %d %d]\n",
      ID, OpY,OpX,KnY,KnX,InC, KnY,KnX,InC,KnC, OpX,OpY,KnC);
  }

  void forward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param) const override
  {
    assert(act[ID]->layersSize   == OpY * OpX *                   KnC);
    assert(act[ID-1]->layersSize == OpY * OpX * KnY * KnX * InC      );
    assert(param[ID]->nWeights   ==             KnY * KnX * InC * KnC);
    assert(param[ID]->nBiases    ==                               KnC);

    const int batchSize = act[ID]->batchSize;

    {
            Real* const __restrict__ OUT = act[ID]->output;
      const Real* const __restrict__ B = param[ID]->biases;
      #pragma omp parallel for schedule(static)
      for (int i=0; i<batchSize * OpY * OpX * KnC; i++) OUT[i] = B[i % KnC];
    }
    {
      const int mm_outRow = batchSize * OpY * OpX;
      const int mm_nInner = KnY * KnX * InC;
      const int mm_outCol = KnC;
      gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        mm_outRow, mm_outCol, mm_nInner,
    		(Real) 1.0, act[ID-1]->output, mm_nInner,
                    param[ID]->weights, mm_outCol,
    		(Real) 1.0, act[ID]->output, mm_outCol);
    }
  }

  void bckward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param,
               const std::vector<Params*>& grad) const override
  {
    const int batchSize = act[ID]->batchSize;
    {
      const Real* const __restrict__ dEdO = act[ID]->dError_dOutput;
            Real* const __restrict__ B = grad[ID]->biases;
      std::fill(B, B+KnC, 0);
      #pragma omp parallel for schedule(static) reduction(+ : B[:KnC])
      for (int i=0; i<batchSize * OpY * OpX * KnC; i++) B[i % KnC] += dEdO[i];
    }
    {
      // Compute gradient of error wrt to kernel parameters:
      // [KnY*KnX*InC, KnC] = [BS*OpY*OpX, KnY*KnX*InC]^T [BS*OpX*OpY, KnC]
      const int mm_outRow = KnY * KnX * InC;
      const int mm_nInner = batchSize * OpY * OpX;
      const int mm_outCol = KnC;
      gemm(CblasRowMajor, CblasTrans, CblasNoTrans,
          mm_outRow, mm_outCol, mm_nInner,
      		(Real) 1.0, act[ID-1]->output,       mm_outRow,
                      act[ID]->dError_dOutput, mm_outCol,
      		(Real) 0.0, grad[ID]->weights,       mm_outCol);
    }
    {
      // Compute gradient of error wrt to output of previous layer:
      //[BS*OpY*OpX, KnY*KnX*InC] = [BS*OpY*OpX, KnC] [KnY*KnX*InC, KnC]^T
      const int mm_outRow = batchSize * OpY * OpX;
      const int mm_nInner = KnC;
      const int mm_outCol = KnY * KnX * InC;
      gemm(CblasRowMajor, CblasNoTrans, CblasTrans,
          mm_outRow, mm_outCol, mm_nInner,
      		(Real) 1.0, act[ID]->dError_dOutput,   mm_nInner,
                      param[ID]->weights,        mm_nInner,
      		(Real) 0.0, act[ID-1]->dError_dOutput, mm_outCol);
    }
  }

  void init(std::mt19937& gen, const std::vector<Params*>& param) const override
  {
    // get pointers to layer's weights and bias
    Real *const W = param[ID]->weights, *const B = param[ID]->biases;
    // initialize weights with Xavier initialization
    const int nAdded = KnX * KnY * InC, nW = param[ID]->nWeights;
    const Real scale = std::sqrt(6.0 / (nAdded + KnC));
    std::uniform_real_distribution < Real > dis(-scale, scale);
    std::generate(W, W + nW, [&]() {return dis( gen );});
    std::fill(B, B + KnC, 0);
  }
};
