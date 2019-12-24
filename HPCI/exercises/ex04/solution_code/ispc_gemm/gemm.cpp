// File       : gemm.cpp
// Created    : Tue Oct 16 2018 10:45:45 AM (+0200)
// Description: General Matrix-Matrix multiplication with ISPC
// Copyright 2018 ETH Zurich. All Rights Reserved.
#include <cassert>
#include <iostream>
#include <chrono>
#include <string>
#include <random>
#include <cassert>
#include <cstring>
#include <cmath>
using namespace std;

#include "common.h"
#ifdef _USE_ISPC_
#include "gemm_sse2.h" // kernel signature for the ISPC SSE2 code
#include "gemm_avx2.h" // kernel signature for the ISPC AVX2 code
using namespace ispc;
#endif /* _USE_ISPC_ */

#ifdef _WITH_EIGEN_
#include <Eigen/Core>

template <typename T>
using EMat = Eigen::Map< Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> >;

// compiled in separate compilation unit with optimization flags tuned for
// Haswell architecture, see the gemm_eigen.o target in the Makefile
void gemm_eigen(const EMat<Real>&, const EMat<Real>&, EMat<Real>&);
#endif /* _WITH_EIGEN_ */

/**
 * @brief General matrix-matrix multiplication kernel (GEMM). Computes C = AB.
 * Naive implementation.
 *
 * @tparam T Real type parameter
 * @param A Matrix dimension p x r
 * @param B Matrix dimension r x q
 * @param C Matrix dimension p x q
 * @param p Dimensional parameter
 * @param r Dimensional parameter
 * @param q Dimensional parameter
 */
template <typename T>
static void gemm_serial(const T* const A, const T* const B, T* const C,
        const int p, const int r, const int q)
{
    for (int i = 0; i < p; ++i)
        for (int j = 0; j < q; ++j)
        {
            T sum = 0.0;
            for (int k = 0; k < r; ++k)
                sum += A[i*r + k] * B[k*q + j]; // Note: The access pattern
                                                // into matrix B is bad!
            C[i*q + j] = sum;
        }
}

/**
 * @brief General matrix-matrix multiplication kernel (GEMM). Computes C = AB.
 * Implementation with optimized memory access.
 *
 * @tparam T Real type parameter
 * @param A Matrix dimension p x r
 * @param B Matrix dimension r x q
 * @param C Matrix dimension p x q
 * @param p Dimensional parameter
 * @param r Dimensional parameter
 * @param q Dimensional parameter
 */
template <typename T>
static void gemm_serial_tile(const T* const A, const T* const B, T* const C,
        const int p, const int r, const int q)
{
    assert(q%_HTILE_ == 0);
    assert(p%_VTILE_ == 0);
    T tile[_VTILE_][_HTILE_]; // tmp storage to exploit temporal locality.
                              // This must fit into L1 cache, otherwise
                              // performance will degrade

    // outer loops over dimension of matrix C.  Note the necessary stride on
    // the loop counters because of our cached data structure tile (above).
    for (int i = 0; i < p; i += _VTILE_)
        for (int j = 0; j < q; j += _HTILE_)
        {
            // initialize tile to zero
            for (int tv = 0; tv < _VTILE_; ++tv)
                for (int th = 0; th < _HTILE_; ++th)
                    tile[tv][th] = (T)0.0;

            // start of inner product
            for (int k = 0; k < r; ++k)
            {
                for (int tv = 0; tv < _VTILE_; ++tv)
                {
                    const T Aik = A[(i+tv)*r + k];
                    for (int th = 0; th < _HTILE_; ++th)
                        tile[tv][th] += Aik * B[k*q + j + th]; // Optimized memory access
                                                               // for matrix B!
                }
            }

            for (int tv = 0; tv < _VTILE_; ++tv)
                for (int th = 0; th < _HTILE_; ++th)
                    C[(i+tv)*q + j + th] = tile[tv][th]; // Optimized writes
                                                         // into C also
        }
}

/**
 * @brief Compute the Frobenius norm between the difference of two input
 * matrices
 *
 * @tparam T Real type parameter
 * @param M Test matrix dimension p x q
 * @param truth Reference matrix dimension p x q
 * @param p Dimensional parameter
 * @param q Dimensional parameter
 *
 * @return
 */
template <typename T>
static T validate(const T* const M, const T* const truth,
        const int p, const int q)
{
    T res = 0.0;
    for (int i = 0; i < p; ++i)
        for (int j = 0; j < q; ++j)
        {
            const T diff = M[i*q + j] - truth[i*q + j];
            res += diff * diff;
        }
    return std::sqrt(res/(p*q));
}

/**
 * @brief Initialize matrices A and B to uniform random values
 *
 * @tparam T Real type parameter
 * @param A Matrix dimension p x r
 * @param B Matrix dimension r x q
 * @param p Dimensional parameter
 * @param r Dimensional parameter
 * @param q Dimensional parameter
 * @param seed Used for random number generator
 */
template <typename T>
static void initialize(T* const A, T* const B,
        const int p, const int r, const int q, const int seed=101)
{
    default_random_engine gen(seed);
    uniform_real_distribution<T> dist(0.0, 1.0);
    for (int i = 0; i < p; ++i)
        for (int k = 0; k < r; ++k)
            A[i*r + k] = dist(gen);

    for (int k = 0; k < r; ++k)
        for (int j = 0; j < q; ++j)
            B[k*q + j] = dist(gen);
}

/**
 * @brief Benchmark a test kernel versus a baseline kernel
 *
 * @tparam T Real type parameter
 * @tparam MEMORY_OPTIMIZED Flag to switch between naive and memory optimized
 * base kernel
 * @tparam WITH_EIGEN Flag to enable Eigen kernel for benchmarking
 * @param p Dimensional parameter
 * @param r Dimensional parameter
 * @param q Dimensional parameter
 * @param func Function pointer to test kernel
 * @param test_name String to describe additional output
 */
template <typename T, bool MEMORY_OPTIMIZED=false, bool WITH_EIGEN=false>
void benchmark(const int p, const int r, const int q,
        void (*func)(const T* const, const T* const, T* const, const int, const int, const int),
        const string test_name)
{
    T *A, *B, *C, *truth;
    posix_memalign((void**)&A, 32, p*r*sizeof(T));
    posix_memalign((void**)&B, 32, r*q*sizeof(T));
    posix_memalign((void**)&C, 32, p*q*sizeof(T));
    posix_memalign((void**)&truth, 32, p*q*sizeof(T));
    typedef chrono::steady_clock Clock;

    // initialize random matrices
    initialize(A, B, p, r, q);
    memset(C, 0, p*q*sizeof(T));
    memset(truth, 0, p*q*sizeof(T));

    // compute truth
    auto t1 = Clock::now();
    if (WITH_EIGEN)
    {
        (*func)(A, B, truth, p, r, q);
    }
    else
    {
        if (MEMORY_OPTIMIZED)
            gemm_serial_tile(A, B, truth, p, r, q);
        else
            gemm_serial(A, B, truth, p, r, q);
    }
    auto t2 = Clock::now();
    const double t_gold = chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
    const T norm_truth = validate(truth, C, p, q);

    // test kernel
#ifdef _WITH_EIGEN_
    // Matrix wrappers
    const EMat<T> Ap(A, p, r);
    const EMat<T> Bp(B, r, q);
    EMat<T> Cp(C, p, q);
#endif /* _WITH_EIGEN_ */

    auto tt1 = Clock::now();
#ifdef _WITH_EIGEN_
    if (WITH_EIGEN)
        gemm_eigen(Ap, Bp, Cp);
    else
#endif /* _WITH_EIGEN_ */
        (*func)(A, B, C, p, r, q);
    auto tt2 = Clock::now();
    const double t = chrono::duration_cast<chrono::nanoseconds>(tt2 - tt1).count();

    // validate
    const T diff = validate(C, truth, p, q);

    cout << test_name << ":" << endl;
    cout << "  Data type size:     " << sizeof(T) << " byte" << endl;
    cout << "  Number of elements: A=" << p*r << "; B=" << r*q << "; C=" << p*q << endl;
    cout << "  Norm of truth:      " << norm_truth << endl;
    cout << "  Error:              " << diff << endl;
    cout << "  Time reference:     " << t_gold*1.0e-6 << " millisec" << endl;
    cout << "  Time test kernel:   " << t*1.0e-6 << " millisec" << endl;
    cout << "  Speedup:            " << t_gold/t << endl;

    // clean up
    free(A);
    free(B);
    free(C);
    free(truth);
}


int main(void)
{
    // problem size:
    // Matrix $A \in\mathbb{R}^{ p \times r }$
    // Matrix $B \in\mathbb{R}^{ r \times q }$
    // Matrix $C \in\mathbb{R}^{ p \times q }$
    //
    // We solve C = A * B
    constexpr int p = 512;
    constexpr int r = 1024;
    constexpr int q = 1024;

    // benchmark serial case naive vs. memory optimized
    benchmark<Real>(p, r, q, gemm_serial_tile, "GEMM serial access optimized");

#ifdef _USE_ISPC_
    // to be fair, we must use our best serial implementation to benchmark
    // against the vectorized implementations --> use memory optimized version
    // of serial GEMM
    constexpr bool MEMORY_OPTIMIZED = true;
    benchmark<Real,MEMORY_OPTIMIZED>(p, r, q, gemm_sse2, "GEMM ISPC SSE2");
    benchmark<Real,MEMORY_OPTIMIZED>(p, r, q, gemm_avx2, "GEMM ISPC AVX2");
#endif /* _USE_ISPC_ */

#ifdef _WITH_EIGEN_
    // benchmark against the Eigen library
    constexpr bool USE_EIGEN = true;
    benchmark<Real,true,USE_EIGEN>(p, r, q, gemm_avx2, "GEMM Eigen vs. ISPC AVX2");
#endif /* _WITH_EIGEN_ */

    return 0;
}
