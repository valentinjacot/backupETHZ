CXX ?= g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++11 -fopenmp

all: main

main: main.cpp Makefile
	$(CXX) $(CXXFLAGS) main.cpp -o main

run: 
	./varym $(N)

plot:
	./plot

runplot: run plot

clean:
	rm -rf main out results.png

.PHONY: all clean run plot runplot
