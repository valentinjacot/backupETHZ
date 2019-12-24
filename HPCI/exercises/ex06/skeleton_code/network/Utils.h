/*
 *  Utils.h
 *
 *  Created by Guido Novati on 29.10.18.
 *  Copyright 2018 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include <random>
#include <vector>
#include <cassert>
#include <sstream>
#include <cstring>
#include <limits>
#include <cmath>
#include <algorithm>
#include <omp.h>

#if 1
  typedef double Real;
  #define gemv cblas_dgemv
  #define gemm cblas_dgemm
#else
  typedef float Real;
  #define gemv cblas_sgemv
  #define gemm cblas_sgemm
#endif

static constexpr int ALIGNBYTES = 32;
static constexpr Real NNEPS = std::numeric_limits<Real>::epsilon();

inline void _myfree(Real * const & ptr)
{
  if(ptr == nullptr) return;
  free(ptr);
}

inline Real * _myalloc(const int size)
{
  Real * ret = nullptr;
  if(size > 0)
  {
    const int SSIMD = std::ceil(size*sizeof(Real)/(Real)ALIGNBYTES)*ALIGNBYTES;
    posix_memalign((void **) &ret, ALIGNBYTES, SSIMD);
  }
  // else if size = 0 no need to allocate. If code is correct will never be
  // accessed. if code is wrong and memory accessed will cause seg fault.
  return ret;
}

template <typename T>
void _dispose_object(T *& ptr)
{
  if(ptr == nullptr) return;
  delete ptr;
  ptr=nullptr;
}

template <typename T>
void _dispose_object(T *const& ptr)
{
  if(ptr == nullptr) return;
  delete ptr;
}
