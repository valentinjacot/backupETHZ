UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CXX=g++
endif
ifeq ($(UNAME_S),Darwin)
	CXX=g++-7
endif


perf?=0
ifeq "$(perf)" "1"
	CPPFLAGS += -D_PERF_
endif


# these are for running on Euler!
CPPFLAGS+=-Wall -O3 -std=c++11
CPPFLAGS_THREADS=$(CPPFLAGS) -fopenmp


all: diffusion2d_adi_openmp


diffusion2d_adi_openmp: diffusion2d_adi_openmp.cpp
	$(CXX) $(CPPFLAGS_THREADS) -o $@ $<


clean:
	rm -f diffusion2d_adi_openmp diagnostics.dat
