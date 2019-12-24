// File       : gemm.ispc
// Created    : Wed Oct 17 2018 09:43:19 AM (+0200)
// Description: ISPC GEMM body
// Copyright 2018 ETH Zurich. All Rights Reserved.
#include "common.h"

/**
 * @brief General matrix-matrix multiplication kernel (GEMM). Computes C = AB.
 * ISPC implementation used for SSE2 and AVX2 extended ISA
 *
 * @param A Matrix dimension p x r
 * @param B Matrix dimension r x q
 * @param C Matrix dimension p x q
 * @param p Dimensional parameter
 * @param r Dimensional parameter
 * @param q Dimensional parameter
 */
export
#ifdef _ISPC_SSE2_
void gemm_sse2(
#else
void gemm_avx2(
#endif /* _ISPC_SSE2_ */
        const uniform Real* uniform const A,
        const uniform Real* uniform const B,
        uniform Real* uniform const C,
        const uniform int p,
        const uniform int r,
        const uniform int q)
{
    // we use the same tiling strategy for our vectorization to exploit cache
    // memories
    assert(q%_HTILE_ == 0); // for simplicity
    assert(p%_VTILE_ == 0); // for simplicity
    uniform Real tile[_VTILE_][_HTILE_];
    uniform Real Aik[_VTILE_];

    // begin GEMM loops:
    // Note: All loop indices should be declared uniform because all program
    // instances will follow the same execution path.  This allows the compiler
    // to generate better code.  Points in the program where we enter SPMD
    // execution flow are marked with comments starting with'SPMD:'
    for (uniform int i = 0; i < p; i += _VTILE_)
    {
        for (uniform int j = 0; j < q; j += _HTILE_)
        {
            // 1.) INITIALIZE TILE TO ZERO (WILL CONTAIN SUMMATION FOR INNER
            // PRODUCTS).
            //
            // we run this loop first to favor the memory access pattern into
            // the horizontal tile dimnension (_HTILE_)
            for (uniform int tv = 0; tv < _VTILE_; ++tv)
            {
                // each program instance (SIMD lane) executes this loop with
                // its own index th.  The gang size determines how many program
                // instances will run in parallel
                foreach (th = 0 ... _HTILE_) // SPMD: initialize tile to zero
                    tile[tv][th] = (Real)0.0;
            }

            // 2.) INNER PRODUCT
            for (uniform int k = 0; k < r; ++k)
            {
                // fetch matrix A values (uniform among all program instances,
                // each SIMD lane has the same value here)
                for (uniform int tv = 0; tv < _VTILE_; ++tv)
                    Aik[tv] = A[(i+tv)*r + k];

                // we run this loop first to favor the memory access pattern
                // into the horizontal tile dimnension (_HTILE_)
                for (uniform int tv = 0; tv < _VTILE_; ++tv)
                {
                    foreach (th = 0 ... _HTILE_) // SPMD: horizontal on B matrix
                        tile[tv][th] += Aik[tv] * B[k*q + j + th]; // Note: all coalesced
                                                                   // vector loads into B
                }
            }

            // 3.) WRITE RESULT BACK INTO MATRIX C
            //
            // we run this loop first to favor the memory access pattern into
            // the horizontal tile dimnension (_HTILE_)
            for (uniform int tv = 0; tv < _VTILE_; ++tv)
            {
                foreach (th = 0 ... _HTILE_) // SPMD: write to C
                    C[(i+tv)*q + j + th] = tile[tv][th]; // Note: again, all
                                                         // coalesced vector
                                                         // stores!
            }
        }
    }
}
