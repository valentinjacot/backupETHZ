// File       : compressor.h
// Created    : Tue Nov 27 2018 09:58:35 AM (+0100)
// Description: Floating point compression class using ZFP
//              https://computation.llnl.gov/projects/floating-point-compression
// Copyright 2018 ETH Zurich. All Rights Reserved.
#ifndef COMPRESSOR_H_IHMYXR2F
#define COMPRESSOR_H_IHMYXR2F

#include <cassert>
#include <iostream>
#include <cstdlib>

#include "zfp.h"

template <typename T>
class ZFPCompressor2D
{
public:
    ZFPCompressor2D(T* const data, const size_t Nx, const size_t Ny, const double tol) :
        m_tol(tol), m_buf(nullptr)
    {
        m_type  = (4 == sizeof(T)) ? zfp_type_float : zfp_type_double;
        m_field = zfp_field_2d(data, m_type, Nx, Ny);
    }

    ~ZFPCompressor2D()
    {
        if (m_buf)
        {
            delete[] m_buf;
            m_buf = nullptr;
        }
        zfp_field_free(m_field);
    }

    ZFPCompressor2D(const ZFPCompressor2D& c) = delete;
    ZFPCompressor2D& operator=(const ZFPCompressor2D& c) = delete;

    // interface
    unsigned char* compress(size_t& bufsize, size_t& compressed_bytes)
    {
        zfp_stream* zfp = zfp_stream_open(nullptr);
        zfp_stream_set_accuracy(zfp, m_tol);

        bufsize = zfp_stream_maximum_size(zfp, m_field);
        assert(bufsize > 0);
        if (m_buf) delete[] m_buf;          // clear previous compression
        m_buf = new unsigned char[bufsize]; // allocate compression buffer

        // associate compression buffer with bitstream
        bitstream* stream = stream_open(m_buf, bufsize);
        zfp_stream_set_bit_stream(zfp, stream);
        zfp_stream_rewind(zfp);

        // compress
        compressed_bytes = zfp_compress(zfp, m_field);
        assert(compressed_bytes > 0);

        // clean up
        zfp_stream_close(zfp);
        stream_close(stream);

        return m_buf;
    }

    T* decompress(unsigned char* buf, const size_t bufsize)
    {
        zfp_stream* zfp = zfp_stream_open(nullptr);
        zfp_stream_set_accuracy(zfp, m_tol);

        const size_t local_bufsize = zfp_stream_maximum_size(zfp, m_field);
        if ( !(bufsize == local_bufsize) )
            std::cerr << "Compression buffer size mismatch!" << std::endl;

        // associate compression buffer with bitstream
        bitstream* stream = stream_open(buf, bufsize);
        zfp_stream_set_bit_stream(zfp, stream);
        zfp_stream_rewind(zfp);

        // decompress
        if (!zfp_decompress(zfp, m_field))
            std::cerr << "Decompression failed!" << std::endl;

        // clean up
        zfp_stream_close(zfp);
        stream_close(stream);

        return static_cast<T*>( m_field->data );
    }

private:
    // compression tolerance
    double m_tol;

    // compression buffer and size
    unsigned char* m_buf;

    // meta data of input data
    zfp_type m_type;
    zfp_field* m_field;
};

#endif /* COMPRESSOR_H_IHMYXR2F */
