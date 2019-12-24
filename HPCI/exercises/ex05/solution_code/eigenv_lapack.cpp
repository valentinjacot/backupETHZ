// File       : eigenv_lapack.cpp
// Created    : Thu Nov 08 2018 05:39:30 PM (+0100)
// Description: Full eigenvalue sover for symmetric problems
// Copyright 2018 ETH Zurich. All Rights Reserved.
#include <mkl_lapacke.h>
#include "common.h"


int main(int argc, char *argv[])
{
    if(argc!=3) {
        std::cout << "\nUsage: ./eigenv_lapack <N> <alpha>\n" << std::endl;
        exit(1);
    }

    const int N = atoi(argv[1]);
    const double alpha = atof(argv[2]);

    // matrix initialization
    double* A = new double[N*N];
    initialize_matrix(alpha, A, N);

    // Study the function signature for the LAPACKE_dsyev routine:
    // https://software.intel.com/en-us/mkl-developer-reference-c-syev

    // required parameters for the LAPACKE_dsyev routine:
    char jobz = 'N'; // we are only interested in the eigenvalues
    char uplo = 'U'; // we have only initialized the upper-triangular part
    double* lambdas = new double[N]; // placeholder for eigenvalues

    auto tstart = std::chrono::steady_clock::now();

    int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, jobz, uplo, N, A, N, lambdas);

    auto tend = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count();


    if (info!=0) {
        std::cout << "\nThe LAPACKE_dsyev function did not complete successfully !\n" << std::endl;
        exit(1);
    }

    std::cout << "  1st Largest eigenvalue = " << lambdas[N-1] << std::endl;
    std::cout << "  2nd Largest eigenvalue = " << lambdas[N-2] << std::endl;
    std::cout << "  Eigenvalue computation took " << time << " milliseconds." << std::endl;


    // Write results to file
    std::ofstream ofs;
    ofs.open ("lapack.txt", std::ofstream::out | std::ofstream::app);
    ofs << alpha << " " << lambdas[N-1] << " " << lambdas[N-2] << " " << time << "\n";
    ofs.close();


    delete [] A;
    delete [] lambdas;

    return 0;

}
