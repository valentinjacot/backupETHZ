CXX=upcxx
LDFLAGS  += -L../korali/lib -lkorali -lgsl -lgslcblas 
CXXFLAGS += -I../korali/include -I../korali/include/gsl -O3 -g -std=c++17 -Wall -Wfatal-errors
BINARIES = task1_n4
OBJECTS  = model/heat2d.o conduits/single.o conduits/upcxx.o

.SECONDARY:
.PHONY: all
all: $(BINARIES)

$(BINARIES) : % : %.o $(OBJECTS) 
	upcxx -o $@ $^ $(LDFLAGS) 

model/%.o:
	$(MAKE) -C model all

conduits/%.o:
	$(MAKE) -C conduits all

%.o: %.cpp
	upcxx -c $(CXXFLAGS) $<

.PHONY: clean
clean:
	$(MAKE) -C model clean
	$(MAKE) -C conduits clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
