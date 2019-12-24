// File       : gemm_eigen.cpp
// Created    : Tue 23 Oct 2018 02:57:27 PM CEST
// Description: GEMM Eigen implementation
// Copyright 2018 ETH Zurich. All Rights Reserved.

#include <Eigen/Core>
#include "common.h"

// Wrapper to map pointer to Eigen matrix type (needed in benchmark)
template <typename T>
using EMat = Eigen::Map< Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> >;

/**
 * @brief General matrix-matrix multiplication kernel (GEMM). Computes C = AB.
 * Eigen library implementation
 *
 * @param A Eigen matrix dimension p x r
 * @param B Eigen matrix dimension r x q
 * @param C Eigen matrix dimension p x q
 */
void gemm_eigen(const EMat<Real>& A, const EMat<Real>& B, EMat<Real>& C)
{
    C.noalias() += A * B;
}
