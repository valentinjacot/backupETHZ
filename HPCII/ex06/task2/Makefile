CXXFLAGS = -O3 -g -std=c++11 -Wall -Wfatal-errors
BINARIES = jacobi jacobi_opt
MPICXX = mpicxx

.SECONDARY:
.PHONY: all
all: $(BINARIES)

jacobi: jacobi.o
	$(MPICXX) -o $@ $^ -fopenmp

jacobi.o: jacobi.cpp
	$(MPICXX) -c $(CXXFLAGS) -fopenmp $<	

jacobi_opt: jacobi_opt.o
	$(MPICXX) -o $@ $^ -fopenmp

jacobi_opt.o: jacobi_opt.cpp
	$(MPICXX) -c $(CXXFLAGS) -fopenmp $<	

.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
