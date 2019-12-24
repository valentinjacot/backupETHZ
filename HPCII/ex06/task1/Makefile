CXXFLAGS = -O3 -g -std=c++11 -Wall -Wfatal-errors
BINARIES = heat2d_cpu heat2d_mpi
MPICXX = mpic++

.SECONDARY:
.PHONY: all
all: $(BINARIES)

heat2d_cpu : heat2d_cpu.o
	$(CXX) -o $@ $^

heat2d_mpi : heat2d_mpi.o 
	$(MPICXX) -o $@ $^ 

heat2d_cpu.o: heat2d_cpu.cpp
	$(CXX) -c $(CXXFLAGS) $<	

heat2d_mpi.o: heat2d_mpi.cpp
	$(MPICXX) -c $(CXXFLAGS) $<

.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
