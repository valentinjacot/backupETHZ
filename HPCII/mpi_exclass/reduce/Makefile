CXXFLAGS = -O3 -g -std=c++11 -Wall -Wfatal-errors
NVXFLAGS = -O3 -g -std=c++11
BINARIES = reduce_cpu reduce_gpu

.SECONDARY:
.PHONY: all
all: $(BINARIES)

reduce_cpu : reduce_cpu.o 
	$(CXX) -o $@ $^ 

reduce_cpu.o: reduce_cpu.cpp
	$(CXX) -c $(CXXFLAGS) $<

reduce_gpu : reduce_gpu.o 
	nvcc -o $@ $^ 

reduce_gpu.o: reduce_gpu.cu
	nvcc -c $(NVXFLAGS) $<

.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
