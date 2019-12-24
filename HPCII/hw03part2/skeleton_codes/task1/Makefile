LDFLAGS  += -L../korali/lib -lkorali
CXXFLAGS += -I../korali/include -I../korali/include/gsl -O3 -g -std=c++17 -Wall -Wfatal-errors
BINARIES = task1a task1c task1d
OBJECTS = model/grass.o

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
