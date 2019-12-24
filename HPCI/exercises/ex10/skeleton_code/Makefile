# File       : Makefile
# Created    : Tue Oct 16 2018 10:53:24 AM (+0200)
# Description: Compile targets
# Copyright 2018 ETH Zurich. All Rights Reserved.
CXX = mpic++
CXXFLAGS = -Wall -Wextra -pedantic -std=c++11

CXXFLAGS += -Izfp-0.5.4/include -Lzfp-0.5.4/lib
LIBS = -lzfp -lm -lz

# debug=true --> use assertions for debugging
debug ?= false

# disable GZIP output of lossy data
gzipout ?= true

.PHONY: clean cleandata install_zfp

ifeq ("$(debug)", "false")
CXXFLAGS += -O2 -DNDEBUG
else
CXXFLAGS += -O0 -g
endif

ifneq ("$(gzipout)", "true")
CXXFLAGS += -D_NO_GZIP_OUT_
endif

mpi_float_compression: mpi_float_compression.cpp compressor.h helper.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)

setup: zfp-0.5.4 cyclone

zfp-0.5.4:
	curl https://computation.llnl.gov/projects/floating-point-compression/download/$@.tar.gz > $@.tar.gz
	tar -xzf $@.tar.gz
	make -C $@

cyclone:
	curl https://polybox.ethz.ch/index.php/s/BUYTtdsGIx9P4LK/download > $@.bin.gz

clean:
	rm -f mpi_float_compression

cleandata:
	rm -f *_t*.zfp *_t*.bin.gz *_t*.png
