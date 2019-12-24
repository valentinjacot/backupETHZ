// File       : power_manual.cpp
// Created    : Thu Nov 08 2018 05:40:07 PM (+0100)
// Description: Standalone Power Method
// Copyright 2018 ETH Zurich. All Rights Reserved.
#include "common.h"

// fills a vector y: multiplication of matrix A and vector x
inline void _gemv(const int m, const int n, const double* const A, const double* const x, double* const y)
{
    for (int i = 0; i < m; ++i)
    {
        double sum = 0.0;
        for (int j = 0; j < n; ++j)
            sum += A[i*n + j] * x[j];
        y[i] = sum;
    }
}

// overwrites vector y with a unit vector in the same direction
inline void _norm(const int n, double* const y)
{
    double sum2 = 0.0;
    for (int i = 0; i < n; ++i)
        sum2 += y[i] * y[i];
    sum2 = 1.0 / std::sqrt(sum2);
    for (int i = 0; i < n; ++i)
        y[i] *= sum2;
}


int main(int argc, char *argv[])
{
    if(argc!=3) {
        std::cout << "\nUsage: ./power_manual <N> <alpha>\n" << std::endl;
        exit(1);
    }

    const int N = atoi(argv[1]);
    const double alpha = atof(argv[2]);

    // matrix initialization
    double* A = new double[N*N];
    initialize_matrix(alpha, A, N);

    // power method
    double* q0 = new double[N];
    double* q1 = new double[N];

    // initial guess
    for (int i = 0; i < N; i++)
        q0[i] = 0.0;
    q0[0] = 1.0;

    const double tol = 1.0e-12;
    double lambda0 = 1.0e12;
    double lambda1 = 0.0;
    size_t iter = 0;

    auto tstart = std::chrono::steady_clock::now();
    // Main algorithm:
    // You can re-structure the Power Method algorithm such that you only
    // perform one GEMV operation in the iteration loop.  By doing so, you have
    // to use a pointer swap at the end of the loop (which is very cheap to
    // do).
    _gemv(N, N, A, q0, q1);
    while (true)
    {
        _norm(N, q1);
        _gemv(N, N, A, q1, q0);
        lambda1 = 0.0;
        for (int i = 0; i < N; ++i)
            // estimate eigenvalue with Rayleigh quotient
            lambda1 += q0[i] * q1[i];
        ++iter;
        if (std::abs(lambda1 - lambda0) < tol)
            break;
        lambda0 = lambda1;
        std::swap(q0, q1);
    }
    // end of algorithm
    auto tend = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count();

    std::cout << "  Largest eigenvalue = " << lambda1 << ",  iterations = " << iter << std::endl;
    std::cout << "  Eigenvalue computation took " << time << " milliseconds." << std::endl;


    // Write results to file
    std::ofstream ofs;
    ofs.open ("manual.txt", std::ofstream::out | std::ofstream::app);
    ofs << alpha << " " << lambda1 << " " << iter << " " << time << "\n";
    ofs.close();


    delete [] A;
    delete [] q0;
    delete [] q1;

    return 0;
}
