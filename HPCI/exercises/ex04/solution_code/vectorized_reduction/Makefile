# File       : Makefile
# Created    : Tue Oct 16 2018 10:53:24 AM (+0200)
# Description: Compile targets
# Copyright 2018 ETH Zurich. All Rights Reserved.
CXX ?= g++
CXXFLAGS = -Wall -Wextra -pedantic -std=c++11 -fopenmp

# debug=true --> use assertions for debugging
debug ?= false

.PHONY: clean measurement

ifeq ("$(debug)", "false")
CXXFLAGS += -O2 -msse -msse2 -DNDEBUG
else
CXXFLAGS += -O0 -g
endif

# be sure that the compiler generated executable is named 'vec_red'
vec_red: vectorized_reduction.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# You can submit a job on euler using this target.  It depends on the 'vec_red'
# target, that is, it will compile the code if there are changes prior to
# submitting the job.  It will then submit the contents of the script
# measure_speedup.sh
measurement: vec_red
	bsub -W 00:15 \
		-n 24 -R fullnode -R 'rusage[scratch=16]' < measure_speedup.sh

clean:
	rm -f vec_red
