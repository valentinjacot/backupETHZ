# File       : Makefile
# Created    : Thu Nov 08 2018 05:56:11 PM (+0100)
# Description: Compile targets.  Before compiling on Euler, do:
#              module load openblas mkl
# Copyright 2018 ETH Zurich. All Rights Reserved.
CXX ?= g++
CXXFLAGS = -Wall -Wextra -pedantic -std=c++11

# debug=true --> use assertions for debugging
debug ?= false

ifeq ("$(debug)", "false")
CXXFLAGS += -O3 -DNDEBUG
else
CXXFLAGS += -O0 -g
endif

# Common linker flags
LDFLAGS=-lm -ldl

# MKL only flags
MKL_FLAGS=-DMKL_ILP64 -m64 -I$(MKLROOT)/include
MKL_LDFLAGS=-Wl,--start-group \
			$(MKLROOT)/lib/intel64/libmkl_intel_ilp64.a \
			$(MKLROOT)/lib/intel64/libmkl_gnu_thread.a \
			$(MKLROOT)/lib/intel64/libmkl_core.a \
			-Wl,--end-group -lgomp -lpthread


.PHONY: all clean clear

all: power_manual power_cblas eigenv_lapack

power_manual: power_manual.cpp initialization.o
	$(CXX) $(CXXFLAGS) -o $@ $^

power_cblas: power_cblas.cpp initialization.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lopenblas $(LDFLAGS)

eigenv_lapack: eigenv_lapack.cpp initialization.o
	$(CXX) $(CXXFLAGS) $(MK_LFLAGS) -o $@ $^ $(MKL_LDFLAGS) $(LDFLAGS)

initialization.o: initialize_matrix.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f power_manual power_cblas eigenv_lapack *.o

clear:
	rm -f manual.txt cblas.txt lapack.txt
