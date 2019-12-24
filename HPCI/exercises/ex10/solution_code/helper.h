// File       : helper.h
// Created    : Tue Nov 27 2018 05:52:10 PM (+0100)
// Description: Helper functions
// Copyright 2018 ETH Zurich. All Rights Reserved.
#ifndef HELPER_H_QDLXC62J
#define HELPER_H_QDLXC62J

#include <cassert>
#include <chrono>
#include <vector>
#include <iostream>
#include <string>
#include <cstring>
#include <cmath>
#include <limits>
#include <zlib.h>
#include <mpi.h>

/**
 * @brief Read horizontal tile of gzip compressed input file
 *
 * @tparam T Data precision
 * @param filename Name of input file
 * @param row_offset Read from this row
 * @param tile_width Width of tile (number of columns)
 * @param tile_height Height of tile (number of rows)
 *
 * @return  Pointer to read data
 */
template <typename T>
T* read_gzfile_tile(const std::string& filename,
        size_t row_offset,
        size_t tile_width,
        size_t tile_height)
{
    gzFile fp = gzopen(filename.c_str(), "rb");
    if (fp == nullptr)
    {
        std::cout << "Gzip: Can not open file '" << filename << "'" << std::endl;
        std::exit(1);
    }

    // set file pointer based on row_offset
    gzseek(fp, row_offset*tile_width*sizeof(T), SEEK_SET);

    // read compressed file
    T* A   = new T[tile_width*tile_height];
    T* row = new T[tile_width];
    assert(A != nullptr);
    assert(row != nullptr);

    const size_t row_bytes = tile_width*sizeof(T);
    for (size_t i = 0; i < tile_height; ++i)
    {
        gzread(fp, row, row_bytes);
        std::memcpy(&A[i*tile_width], row, row_bytes);
    }

    // clean up
    gzclose(fp);
    delete[] row;

    return A;
}

/**
 * @brief Write gzip compressed file (not optimized)
 *
 * @tparam T Data precision
 * @param data Data to be written
 * @param filename Output filename
 * @param tile_width Width of tile (number of columns)
 * @param tile_height Height of tile (number of rows)
 */
template <typename T>
void write_gzfile_tile(T* const data,
        const std::string& filename,
        size_t tile_width,
        size_t tile_height)
{
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const bool isroot = (0 == rank);

    T* buf;
    const size_t tile_size = tile_height * tile_width;
    std::vector<MPI_Request> requests;
    if (isroot)
    {
        buf = new T[tile_width * tile_width];
        for (int i = 1; i < size; ++i)
        {
            MPI_Request req;
            MPI_Irecv(&buf[i*tile_size], tile_size, (sizeof(T)==4)? MPI_FLOAT : MPI_DOUBLE, i, 42, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
        std::memcpy(buf, data, tile_size*sizeof(T));
    }
    else
        MPI_Send(data, tile_size, (sizeof(T)==4)? MPI_FLOAT : MPI_DOUBLE, 0, 42, MPI_COMM_WORLD);

    if (isroot)
    {
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

        gzFile fp = gzopen(filename.c_str(), "wb");
        if (fp == nullptr)
        {
            std::cout << "Gzip: Can not create file '" << filename << "'" << std::endl;
            std::exit(1);
        }

        // write data
        gzwrite(fp, buf, tile_width*tile_width*sizeof(T));

        // clean up
        gzclose(fp);
        delete[] buf;
    }
}

using Clock = std::chrono::steady_clock;
using Time  = std::chrono::time_point<Clock>;

/**
 * @brief Report elapsed time
 *
 * @param t1 Start time
 * @param t2 End time
 * @param what Hint
 */
void report_time(Time& t1, Time& t2, std::string what)
{
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const bool isroot = (0 == rank);

    const double t_elap = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() * 1.0e-6;

    double gavg, gmin, gmax;
    MPI_Reduce(&t_elap, &gavg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_elap, &gmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_elap, &gmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (isroot)
    {
        std::cout << std::setw(32) << std::setfill('.') << std::left << what;
        std::cout << ": avg:" << std::scientific << gavg/size << " sec; ";
        std::cout << " min:" << std::scientific << gmin << " sec ; ";
        std::cout << " max:" << std::scientific << gmax << " sec" << std::endl;
    }
}

/**
 * @brief Compute compression statistics
 *
 * @tparam T Data precision
 * @param ref Lossless data
 * @param test Lossy data
 * @param N Data dimension
 * @param myrows Local number of MPI rows
 * @param compressed_bytes Number of bytes for compressed data
 */
template <typename T>
void show_stat(const T* const ref, const T* const test,
               const size_t N, const size_t myrows,
               const size_t compressed_bytes)
{
    double lL1 = 0.;
    double lL2 = 0.;
    double lLinf= std::numeric_limits<double>::min();
    double lmin = std::numeric_limits<double>::max();
    double lmax = std::numeric_limits<double>::min();
    for (size_t i = 0; i < N*myrows; ++i)
    {
        const double v = std::abs( ref[i] );
        lmin = (v < lmin) ? v : lmin;
        lmax = (v > lmax) ? v : lmax;

        const double diff = std::abs( static_cast<double>(test[i]) - static_cast<double>(ref[i]) );
        lL1 += diff;
        lL2 += diff*diff;
        lLinf = (diff > lLinf) ? diff : lLinf;
    }

    double L1 = 0.;
    double L2 = 0.;
    double Linf = std::numeric_limits<double>::min();
    double gmin = std::numeric_limits<double>::max();
    double gmax = std::numeric_limits<double>::min();
    MPI_Reduce(&lL1, &L1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&lL2, &L2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&lLinf, &Linf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&lmin, &gmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&lmax, &gmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double lcbytes = static_cast<double>(compressed_bytes);
    double cbytes = 0.;
    MPI_Reduce(&lcbytes, &cbytes, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (0 == rank)
    {
        const double n0 = N*N;
        const double MSE = L2 / n0;
        const double ubytes = N*N*sizeof(T); // uncompressed bytes
        const double compression_rate = ubytes / cbytes;
        L1 /= n0;
        L2 = std::sqrt( MSE );
        const double psnr = 20. * std::log10( (gmax-gmin) / L2 );
        const double bitrate = cbytes * 8. / n0;
        std::cout << std::setw(32) << std::setfill('.') << std::left << "Uncompressed size" << ": " << std::scientific << ubytes / 1024. /1024. << " MB" << std::endl;
        std::cout << std::setw(32) << std::setfill('.') << std::left << "Compressed size"   << ": " << std::scientific << cbytes / 1024. /1024. << " MB" << std::endl;
        std::cout << std::setw(32) << std::setfill('.') << std::left << "Compression rate"  << ": " << std::scientific << compression_rate << '\n';
        std::cout << std::setw(32) << std::setfill('.') << std::left << "L1 error"          << ": " << std::scientific << L1 << '\n';
        std::cout << std::setw(32) << std::setfill('.') << std::left << "L2 error"          << ": " << std::scientific << L2 << '\n';
        std::cout << std::setw(32) << std::setfill('.') << std::left << "Linf error"        << ": " << std::scientific << Linf << '\n';
        std::cout << std::setw(32) << std::setfill('.') << std::left << "PSNR"              << ": " << std::scientific << psnr << " dB\n";
        std::cout << std::setw(32) << std::setfill('.') << std::left << "BITRATE"           << ": " << std::scientific << bitrate << " bps" << std::endl;
    }
}

#endif /* HELPER_H_QDLXC62J */
