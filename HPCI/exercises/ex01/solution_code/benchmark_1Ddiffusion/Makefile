ifeq ($(shell uname -s),Darwin)  # Mac.
CXX = g++
# To use actual GCC on Mac, please first install it.
# CXX = g++-7
else  # Euler.
CXX = g++
endif

CPPFLAGS = -std=c++11 -O3 -march=native -ffast-math -funroll-loops \
		 -Wall -Wextra \

.PHONY: clean

all: benchmark_1Ddiffusion.cpp
	$(CXX) $(CPPFLAGS) -o benchmark_1Ddiffusion benchmark_1Ddiffusion.cpp

clean:
	rm -f benchmark_1Ddiffusion
