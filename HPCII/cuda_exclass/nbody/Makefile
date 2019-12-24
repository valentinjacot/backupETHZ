CXXFLAGS = -O3 -g -std=c++11 -Wall -Wfatal-errors
NVXFLAGS = -O3 -g --use_fast_math -std=c++11
BINARIES = nbody_cpu nbody_gpu

.SECONDARY:
.PHONY: all
all: $(BINARIES)

nbody_cpu : nbody_cpu.o
	$(CXX) -o $@ $^

nbody_gpu : nbody_gpu.o 
	nvcc -o $@ $^ 

nbody_cpu.o: nbody_cpu.cpp
	$(CXX) -c $(CXXFLAGS) $<	

nbody_gpu.o: nbody_gpu.cu
	nvcc -c $(NVXFLAGS) $<

.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
