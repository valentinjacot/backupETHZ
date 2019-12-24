CXXFLAGS = -O3 -g -std=c++11 -Wall -Wfatal-errors
NVXFLAGS = -O3 -g --use_fast_math -std=c++11
BINARIES = dgemm dgemm_opt

.SECONDARY:
.PHONY: all
all: $(BINARIES)

dgemm : dgemm.o 
	nvcc -o $@ $^ 

dgemm.o: dgemm.cu
	nvcc -c $(NVXFLAGS) $<

dgemm_opt : dgemm_opt.o 
	nvcc -o $@ $^ 

dgemm_opt.o: dgemm_opt.cu
	nvcc -c $(NVXFLAGS) $<

.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
