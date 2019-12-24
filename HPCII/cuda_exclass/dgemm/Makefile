CXXFLAGS = -O3 -g -std=c++11 -Wall -Wfatal-errors
NVXFLAGS = -O3 -g --use_fast_math -std=c++11
BINARIES = dgemm_cpu dgemm_gpu

.SECONDARY:
.PHONY: all
all: $(BINARIES)

dgemm_cpu : dgemm_cpu.o
	$(CXX) -o $@ $^

dgemm_gpu : dgemm_gpu.o 
	nvcc -o $@ $^ 

dgemm_cpu.o: dgemm_cpu.cpp
	$(CXX) -c $(CXXFLAGS) $<	

dgemm_gpu.o: dgemm_gpu.cu
	nvcc -c $(NVXFLAGS) $<

.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
