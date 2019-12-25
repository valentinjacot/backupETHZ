LDFLAGS  += -L../korali/lib -lkorali
CXXFLAGS += -I../korali/include -I../korali/include/gsl -O3 -g -std=c++17 -Wall -Wfatal-errors
BINARIES = minimize_function sample_function maximize_likelihood sample_likelihood maximize_posterior sample_posterior

.SECONDARY:
.PHONY: all
all: $(BINARIES)

$(BINARIES) : % : %.o $(OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $<

.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
