CXX=mpic++
CXXFLAGS+=-std=c++11 -Wall -Wpedantic -O3

main: main.o
	$(CXX) $(CXXFLAGS) main.o -o main

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o main
