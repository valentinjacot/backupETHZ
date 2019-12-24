/*
 *  main_backprop.cpp
 *
 *  1) Train autoencoder on MNIST dataset.
 *  2) Write to file the principal components.
 *
 *  Created by Guido Novati on 30.10.18.
 *  Copyright 2018 ETH Zurich. All rights reserved.
 *
 */


#include "network/Network.h"
#include "network/Optimizer.h"
#include "mnist/mnist_reader.hpp"
#include <chrono>

static void prepare_input(const std::vector<int>& image, std::vector<Real>& input)
{
  static const Real fac = 1/(Real)255;
  assert(image.size() == input.size());
  for (size_t j = 0; j < input.size(); j++) input[j] = image[j]*fac;
}

static Real compute_error(const std::vector<Real>& output, std::vector<Real>& input)
{
  Real l2err = 0;
  assert(output.size() == input.size());
  for (size_t j = 0; j < input.size(); j++) {
    l2err += std::pow(input[j] - output[j], 2);
    // gradient of l2err/2 wrt to output[j]:
    input[j] = output[j] - input[j];
  }
  return l2err / 2;
}

int main (int argc, char** argv)
{
  std::cout << "MNIST data directory: ./" << std::endl;

  // Load MNIST data"
  mnist::MNIST_dataset<std::vector, std::vector<int>, uint8_t> dataset =
  mnist::read_dataset<std::vector, std::vector, int, uint8_t>("./");
  assert(dataset.training_labels.size() == dataset.training_images.size());
  assert(dataset.test_labels.size() == dataset.test_images.size());
  const int n_train_samp = dataset.training_images.size();
  const int n_test_samp = dataset.test_images.size();

  // Training parameters:
  const int nepoch = 30, batchsize = 32;
  const Real learn_rate = 1e-4;
  // Compression parameter:
  const int Z = 10;

  // Create Network:
  Network net;
  // layer 0: input
  net.addInput<28*28*1>();
  // layer 1: linear encoder
  net.addLinear<28*28*1, Z>();
  // layer 2: linear decoder
  net.addLinear<Z, 28*28*1>();

  const size_t compressionID = 1; // ID of layer whose size is Z

  //Create optimizer:
  Optimizer<MomentumSGD> opt(net, learn_rate);

  const int steps_in_epoch = n_train_samp / batchsize;
  assert(steps_in_epoch > 0);

  for (int iepoch = 0; iepoch < nepoch; iepoch++)
  {
    std::vector<std::vector<Real>> INP(batchsize, std::vector<Real>(28*28));
    std::vector<std::vector<Real>> OUT(batchsize, std::vector<Real>(28*28));
    std::vector<int> sample_ids(n_train_samp);
    //fill array: 0, 1, ..., n_train_samp-1
    std::iota(sample_ids.begin(), sample_ids.end(), 0);

    //shuffle dataset in order to sample random mini batches:
    std::shuffle(sample_ids.begin(), sample_ids.end(), net.gen );

    Real epoch_mse = 0;
    const double t0 = omp_get_wtime();
    for (int step = 0; step < steps_in_epoch; step++)
    {
      // Put `batchsize` samples in the inputs vector-of-vectors. Start from
      // the end because it's easier to remove entries from a vectors's end.

  #pragma omp parallel for schedule(static)
      for (int i = 0; i < batchsize; i++)
      {
        const int sample = sample_ids[sample_ids.size() - 1 - i];
        prepare_input(dataset.training_images[sample], INP[i]);
      }

      net.forward(OUT, INP);

      // Compute the error = 1/2 \Sum (OUT - INP) ^ 2
#pragma omp parallel for schedule(static) reduction(+ : epoch_mse)
      for (int i = 0; i < batchsize; i++)
      {
        // For simplicity here we overwrite INP with the gradient of the error
        // With respect to the Network's outputs = OUT - INP. OUT and INP have
        // the same size and that's the size of the net's output
        const Real error = compute_error(OUT[i], INP[i]); //now INP contains ERR
        epoch_mse += error;
      }

      net.bckward(INP);

      opt.update(batchsize);

      // Erase last batchsize randomly shuffled dataset samples:
      sample_ids.erase(sample_ids.end()-batchsize, sample_ids.end());
    }
    const double elapsed = omp_get_wtime() - t0;

    if(iepoch % 1 == 0)
    {
      const int steps_in_test = n_test_samp / batchsize;

      Real test_mse = 0;
      for (int step = 0; step < steps_in_test; step++)
      {

#pragma omp parallel for schedule(static)
        for (int i = 0; i < batchsize; i++)
        {
          const int sample = i + batchsize * step;
          prepare_input(dataset.test_images[sample], INP[i]);
        }

        net.forward(OUT, INP);

#pragma omp parallel for schedule(static) reduction(+ : test_mse)
        for (int i = 0; i < batchsize; i++)
          test_mse += compute_error(OUT[i], INP[i]); //now input contains err
      }
      printf("Training set MSE:%f, Test set MSE:%f, wclock %f\n",
        epoch_mse/steps_in_epoch/batchsize, test_mse/steps_in_test/batchsize, elapsed);
    }
  }

  //extract features:
  // WARNING: if you change the shape of the net in any way, this will fail.
  // If you add layers, edit the `compressionID` variable accordingly.
  for (int z = 0; z < Z; z++)
  {
    // initialize layer output of all zeros:
    std::vector<Real> z_vec(Z, 0);
    // turn on only one component in the compression layer
    z_vec[z] = 1;

    const std::vector<Real> OUT = net.forward(z_vec, compressionID);
    std::vector<float> OUT_float(OUT.size());

    std::copy(OUT.begin(), OUT.end(), OUT_float.begin());

    FILE* pFile = fopen(("component_"+std::to_string(z)+".raw").c_str(),"wb");
    fwrite(OUT_float.data(), sizeof(float), 28*28, pFile);
    fflush(pFile);
    fclose(pFile);
  }

  return 0;
}
