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

// Im2MatLayer gets as input an image of sizes InX * InY * InC
// and prepares the output for convolution with a filter of size KnY * KnX * KnC
// and output an image of size OpY * OpX * KnC
template
<
  int InX, int InY, int InC, //input image: x:width, y:height, c:color channels
  int KnX, int KnY, int KnC, //filter:      x:width, y:height, c:color channels
  int Sx, int Sy, // stride  x/y
  int Px, int Py, // padding x/y
  int OpX, int OpY //output img: x:width, y:height, same color channels as KnC
>
struct Im2MatLayer: public Layer
{
  //Im2ColLayer has no parameters:
  Params* allocate_params() const override { return nullptr; }

  Im2MatLayer(const int _ID) : Layer(OpY*OpX*KnY*KnX*InC, _ID) {
    static_assert(Sx> 0 && Sy> 0, "Invalid stride");
    static_assert(Px>=0 && Py>=0, "Invalid kernel");
    print();
  }

  void print() {
    printf("(%d) Im2Col transform Img:[%d %d %d] to Mat:[%d %d %d %d %d] ",
          ID, InY,InX,InC, OpY,OpX,KnY,KnX,InC);
    printf("with Stride:[%d %d] and Padding:[%d %d]\n",Sx,Sy,Px,Py);
  }

  void forward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param) const override
  {
    const int batchSize = act[ID]->batchSize;

    assert(act[ID-1]->layersSize == InX * InY * InC);
    assert(act[ID]->layersSize == OpY * OpX * KnY * KnX * InC);
    Im2Mat(batchSize, act[ID-1]->output, act[ID]->output);
  }

  void bckward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param,
               const std::vector<Params*>& grad) const override
  {
    const int batchSize = act[ID]->batchSize;

    assert(act[ID-1]->layersSize == InX * InY * InC);
    assert(act[ID]->layersSize == OpY * OpX * KnY * KnX * InC);
    Mat2Im(batchSize, act[ID]->dError_dOutput, act[ID-1]->dError_dOutput);
  }

  void Im2Mat(const int BS, const Real*const lin_inp, Real*const lin_out) const
  {
    using InputImages    = Real[][InY][InX][InC];
    using OutputMatrices = Real[][OpY][OpX][KnY][KnX][InC];

    // Convert pointers to a reference to multi dim arrays for easy access:
    // 1) INP is a reference: i'm not creating new data
    // 2) The type of INP is an array of sizes [???][InY][InX][InC]
    // 3) The first dimension is the batchsize and is not known at compile time
    // 4) Because it's the slowest index the compiler does not complain
    // 5) The conversion should be read from right to left: (A) convert lin_inp
    // to pointer to a static multi-array of size [???][InY][InX][InC]
    // (B) Return the reference of the memory space pointed at by a.
    const InputImages & INP = * (InputImages*) lin_inp;
    //                       (B)(     A      )
    OutputMatrices & OUT = * (OutputMatrices*) lin_out;

    // clean up memory space of lin_out. Why? Because padding, that's why.
#if 1
    for (int bc=0; bc<BS; bc++)
      for (int oy = 0; oy < OpY; oy++)
      for (int ox = 0; ox < OpX; ox++)
        for (int fy = 0; fy < KnY; fy++)
        for (int fx = 0; fx < KnX; fx++)
          for (int ic = 0; ic < InC; ic++)
            OUT[bc][oy][ox][fy][fx][ic] = 0;
#else
    memset(lin_out, 0, BS * OpY * OpX * KnY * KnX * InC * sizeof(Real) );
#endif

    printf("TODO: Im2MatLayer::Im2Mat\n");
    abort();
  }

  void Mat2Im(const int BS, const Real*const lin_inp, Real*const lin_out) const
  {
    using InputImages    = Real[][InY][InX][InC];
    using OutputMatrices = Real[][OpY][OpX][KnY][KnX][InC];
    // Output is d Loss d Input, same size as INP before:
    InputImages & dLdINP = * (InputImages*) lin_out;
    // Input is d Loss d Output, same size as OUT before:
    const OutputMatrices & dLdOUT = * (OutputMatrices*) lin_inp;

    // Mat2Im accesses memory with plus equal: reset field
#if 1
    for (int bc=0; bc<BS; bc++)
      for (int iy = 0; iy < InY; iy++)
      for (int ix = 0; ix < InX; ix++)
        for (int ic = 0; ic < InC; ic++)
          dLdINP[bc][iy][ix][ic] = 0;
#else
    memset(lin_out, 0, BS * InY * InX * InC * sizeof(Real) );
#endif

    printf("TODO: Im2MatLayer::Mat2Im\n");
    abort();
  }

  void init(std::mt19937& G, const std::vector<Params*>& P) const override {  }
};
