// File       : power_cblas.cpp
// Created    : Thu Nov 08 2018 05:40:51 PM (+0100)
// Description: Power Method with reference to BLAS
// Copyright 2018 ETH Zurich. All Rights Reserved.
#include <cblas.h>
#include "common.h"


int main(int argc, char *argv[])
{
    if(argc!=3) {
        std::cout << "\nUsage: ./power_cblas <N> <alpha>\n" << std::endl;
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
    cblas_dsymv(CblasRowMajor, CblasUpper , N, 1.0, A, N, q0, 1, 0.0, q1, 1);
    while (true)
    {
        const double norm = cblas_dnrm2(N, q1, 1);
        cblas_dscal(N, 1./norm, q1, 1);
        cblas_dsymv(CblasRowMajor, CblasUpper , N, 1.0, A, N, q1, 1, 0.0, q0, 1);
        lambda1 = cblas_ddot(N, q0, 1, q1, 1);
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
    ofs.open ("cblas.txt", std::ofstream::out | std::ofstream::app);
    ofs << alpha << " " << lambda1 << " " << iter << " " << time << "\n";
    ofs.close();


    delete [] A;
    delete [] q0;
    delete [] q1;

    return 0;
}
