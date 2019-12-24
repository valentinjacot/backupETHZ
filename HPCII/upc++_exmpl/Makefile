CXX = upcxx
CXXFLAGS += -O3 -g -std=c++17

SOURCES =  $(shell ls *.cpp)
BINARIES = $(SOURCES:.cpp=)

all: $(BINARIES)

clean:; rm -rf *.a *.o $(BINARIES)

.PHONY: clean all

