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
  for (size_t j = 0; j < 28*28; j++) input[j] = image[j]*fac;
}

static inline uint8_t max_index(const std::vector<Real>& O) {
  return std::distance(O.begin(), std::max_element(O.begin(), O.end()));
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
  const int nepoch = 100, batchsize = 512;
  const Real learn_rate = 1e-5;

  // Create Network:
  Network net;
  // layer 0: input
  net.addInput<28*28*1>();

 #if 1
  net.addConv2D< 28, 28,  1,   8,   8,   4,   2,2,    0,0>();
  net.addLReLu< 11 * 11 * 4 >();
  net.addConv2D< 11, 11,  4,   6,   6,   8,   1,1,    0,0>();
  net.addLReLu< 6 * 6 * 8 >();
  net.addConv2D<  6,  6,  8,   4,   4,  16,   1,1,    0,0>();
  net.addLReLu< 3 * 3 * 16 >();
  net.addConv2D<  3,  3, 16,   3,   3,  10,   1,1,    0,0>();
#else
  //            (input img)  (conv filter)  (stride)
  //             nx, ny, nc  nfx, nfy, nfc         (padding)
  net.addConv2D< 28, 28,  1,   8,   8,  16,   2,2,    0,0>();
  // output of first conv has sizes (28 -8 +2*0)/2+1 = 11
  net.addLReLu< 11 * 11 * 16>();

  net.addConv2D< 11, 11, 16,   6,   6,  32,   1,1,    0,0>();
  // output of first conv has sizes (11 -6 +2*0)/1+1 = 6
  net.addLReLu< 6 * 6 * 32>();

  net.addConv2D<  6,  6, 32,   4,   4,  64,   1,1,    0,0>();
  // output of first conv has sizes ( 6 -4 +2*0)/1+1 = 3
  net.addLReLu< 3 * 3 * 64 >();

  net.addLinear<3 * 3 * 64, 96>();
  net.addTanh<96>();
  net.addLinear<96, 10>();
#endif

  net.addSoftMax<10>();

  //Create optimizer:
  Optimizer<Adam> opt(net, learn_rate, 1e-6);

  const int steps_in_epoch = n_train_samp / batchsize;
  assert(steps_in_epoch > 0);

  for (int iepoch = 0; iepoch < nepoch; iepoch++)
  {
    std::vector<std::vector<Real>> INP(batchsize, std::vector<Real>(28*28));
    std::vector<std::vector<Real>> OUT(batchsize, std::vector<Real>(10));

    std::vector<int> sample_ids(n_train_samp);
    //fill array: 0, 1, ..., n_train_samp-1
    std::iota(sample_ids.begin(), sample_ids.end(), 0);

    //shuffle dataset in order to sample random mini batches:
    std::shuffle(sample_ids.begin(), sample_ids.end(), net.gen );

    Real epoch_mse  = 0, epoch_prec = 0;
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
#pragma omp parallel for reduction(+ : epoch_mse, epoch_prec) schedule(static)
      for (int i = 0; i < batchsize; i++)
      {
        // For simplicity here we overwrite OUT with the gradient of the error
        const int sample = sample_ids[sample_ids.size() - 1 - i];
        const uint8_t label = dataset.training_labels[sample];
        assert(label < 10 and OUT[i].size() == 10);
        std::vector<Real> ret(10 , 0);
        // predicted label is output with higher probability
        const uint8_t predicted_label = max_index(OUT[i]);
        // error is cross-entropy = - sum P(label) * log ( P_predicted (label) )
        // P(label) == 1 only for the correct label, 0 otherwise
        ret[label] = - 1 / OUT[i][label]; // - 1 * d/d_output * log(output)
        epoch_mse -= std::log(OUT[i][label]);
        epoch_prec += (predicted_label == label);
        OUT[i] = ret; // overwrite output with grad of err wrt to output
      }

      net.bckward(OUT);

      opt.update(batchsize);

      // Erase last batchsize randomly shuffled dataset samples:
      sample_ids.erase(sample_ids.end()-batchsize, sample_ids.end());
    }
    const double elapsed = omp_get_wtime() - t0;

    if(iepoch % 1 == 0)
    {
      const int steps_in_test = n_test_samp / batchsize;

      Real test_mse = 0, test_prec = 0;
      for (int step = 0; step < steps_in_test; step++)
      {

#pragma omp parallel for schedule(static)
        for (int i = 0; i < batchsize; i++)
        {
          const int sample = i + batchsize * step;
          prepare_input(dataset.test_images[sample], INP[i]);
        }

        net.forward(OUT, INP);

#pragma omp parallel for reduction(+ : test_mse, test_prec) schedule(static)
        for (int i = 0; i < batchsize; i++) {
          const int sample = i + batchsize * step;
          const uint8_t label = dataset.test_labels[sample];
          const uint8_t predicted_label = max_index(OUT[i]);
          assert(label < 10 and OUT[i].size() == 10);
          test_mse -= std::log(OUT[i][label]);
          test_prec += (predicted_label == label);
        }
      }
      printf("Training set mean cross-entropy:%f precision:%f, Test set mean "
        "cross-entropy:%f precision:%f. Elapsed wtime:%f\n",
        epoch_mse/steps_in_epoch/batchsize, epoch_prec/steps_in_epoch/batchsize,
        test_mse/steps_in_test/batchsize, test_prec/steps_in_test/batchsize,
        elapsed );
    }
  }

  return 0;
}
