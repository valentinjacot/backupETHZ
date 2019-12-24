CXX=g++
CXXFLAGS=-O3 -Wall -Wextra -std=c++11

.PHONY: all clean run plot

all: associativity

clean:
	rm -f associativity results.txt results.png

run:
	./associativity | tee results.txt
	# Results stored into 'results.txt', run 'make plot'.

plot:
	./plot.sh
	# Plot stored into 'results.png'.

associativity: associativity.cpp Makefile
	$(CXX) $(CXXFLAGS) associativity.cpp -o associativity

