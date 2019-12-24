all: single.o upcxx.o

CXXFLAGS = -march=native -D NOALIAS -O3 -g -std=c++17 -Wall -Wfatal-errors -I../../korali/include

single.o: single.cpp
	$(CXX) -c -o $@ $^ $(CXXFLAGS)

upcxx.o: upcxx.cpp
	upcxx -c -o $@ $^ $(CXXFLAGS)

clean:; rm -rf $(TARGET) *.o

.PHONY: clean all
