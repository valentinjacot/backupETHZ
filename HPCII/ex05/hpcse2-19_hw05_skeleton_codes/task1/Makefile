CXXFLAGS = -O3 -g -std=c++11 -Wall -Wfatal-errors
NVXFLAGS = -O3 -g --use_fast_math -std=c++11
BINARIES = heat2d_cpu heat2d_gpu

.SECONDARY:
.PHONY: all
all: $(BINARIES)

heat2d_cpu : heat2d_cpu.o
	$(CXX) -o $@ $^

heat2d_gpu : heat2d_gpu.o 
	nvcc -o $@ $^ 

heat2d_cpu.o: heat2d_cpu.cpp
	$(CXX) -c $(CXXFLAGS) $<	

heat2d_gpu.o: heat2d_gpu.cu
	nvcc -c $(NVXFLAGS) $<

.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
