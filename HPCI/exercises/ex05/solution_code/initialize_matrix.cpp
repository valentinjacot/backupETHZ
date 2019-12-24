// File       : initialize_matrix.cpp
// Created    : Thu Nov 08 2018 05:42:12 PM (+0100)
// Description: Symmetric random initial matrix
// Copyright 2018 ETH Zurich. All Rights Reserved.
#include <random>

// alpha: Diagonal scaling factor
// A:     N x N Matrix
// N:     Matrix dimension
void initialize_matrix(const double alpha, double* const A, const int N)
{
   // random generator with seed 0
   std::default_random_engine g(0);
   // uniform distribution in [0, 1]
   std::uniform_real_distribution<double> u;

    // matrix initialization
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            const double rand = u(g);
            A[i*N + j] = rand;
            A[j*N + i] = rand;
        }
        A[i*N + i] = (i+1.0) * alpha;
    }
}
