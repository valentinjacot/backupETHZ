// File       : benchmark_1Ddiffusion.cpp
// Created    : Thu Oct 12 2017 04:04:24 PM (+0200)
// Description: Benchmark 1D diffusion test
// Copyright 2017 ETH Zurich. All Rights Reserved.
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <chrono>

#define Ntimesteps 1000
#define FLUSH
#define BUFSIZE 1<<26   // ~60 MB


inline void _flush_cache(volatile unsigned char* buf)
{
    for(unsigned int i = 0; i < BUFSIZE; ++i)
    {
        buf[i] += i;
    }
}

// update the solution based on diffusion scheme
inline void _sweep(const double * const __restrict in, double * const __restrict out, const long long int N, const double c)
{
    out[0] = in[0] + c*(in[N-1] - 2.0*in[0] + in[1]);
    for (long long int i = 1; i < N-1; ++i)
        out[i] = in[i] + c*(in[i-1] - 2.0*in[i] + in[i+1]);
    out[N-1] = in[N-1] + c*(in[N-2] - 2.0*in[N-1] + in[0]);
}

int main()
{
    // parameter
    const long long int N = 1<<20;
    const double alpha = 1.0e-4;
    const double L = 1000.0;

    // allocate memory
    std::vector<double> v0(N); // 8MB
    std::vector<double> v1(N); // 8MB
    double* uold = v0.data();
    double* unew = v1.data();
#ifdef FLUSH
    volatile unsigned char* buf = new volatile unsigned char[BUFSIZE];
#endif /* FLUSH */

    // constants
    const double dx = L/(N-1); // grid spacing
    const double dt = 0.5*dx*dx/alpha; // time step
    const double c  = alpha*dt/(dx*dx);

    // initialize data
    auto u0 = [L](const double x) { return std::sin(2.0*M_PI/L*x); };
    for (long long int i = 0; i < N; ++i)
        uold[i] = u0( i*dx );

    // run benchmark
    std::chrono::duration<double> tsum(0);
    for (int i = 0; i < Ntimesteps; ++i)
    {
#ifdef FLUSH
        _flush_cache(buf); // ensure the cache is cold
#endif /* FLUSH */

        const auto t0 = std::chrono::steady_clock::now();
        _sweep(uold, unew, N, c); // here we perform the sweeps over the grid
        const auto t1 = std::chrono::steady_clock::now();
        tsum += t1-t0;

        std::swap(uold, unew);
    }
#ifdef FLUSH
    delete[] buf; // deallocate memory
#endif /* FLUSH */

    // report
    const double flop = 5; // 5 flop per grid point
    const double mem  = 2; // assuming finite size cache, 1 read + 1 write
    const double OI   = flop/(mem*sizeof(double));
    const double bandwidth = 68.3; // GB/s (per socket, Intel Xeon E5-2680v3)
    const double peakFlop  = 3.3 * 4 * 2 * 1; // 3.3GHz (Max Turbo  Frequency) * 256bit=4double SIMD (AVX-2) * 2 FMA/cycle * 1 core; [peakFlop]=Gflop/s
    const double nominalPerf = std::min(OI*bandwidth, peakFlop);
    const double Nflop= flop * N * Ntimesteps;
    const double fperf= Nflop/tsum.count() * 1.0e-9; // Gflop/s
    std::cout << "Last value in the last timestep:     " << uold[N-1] << std::endl;
    std::cout << "Precision:                           " << sizeof(double) << " byte" << std::endl;
    std::cout << "Number of timesteps:                 " << Ntimesteps << std::endl;
    std::cout << "Operational intensity:               " << OI << " flop/byte" << std::endl;
    std::cout << "Nominal floating point performance:  " << nominalPerf << " Gflop/s" << std::endl;
    std::cout << "Measured floating point performance: " << fperf << " Gflop/s" << std::endl;
    std::cout << "Measured peak performance:           " << fperf/nominalPerf * 100.0 << " %" << std::endl;

    return 0;
}
