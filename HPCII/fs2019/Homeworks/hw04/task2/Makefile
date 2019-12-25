CXX=upcxx
LDFLAGS  += 
CXXFLAGS += -O3 -g -std=c++17 -Wall -Wfatal-errors
BINARIES = single task2a task2b
OBJECTS = sampler/sampler.o

.SECONDARY:
.PHONY: all
all: $(BINARIES)

$(BINARIES) : % : %.o $(OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
