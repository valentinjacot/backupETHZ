CXX ?= g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++11 -fopenmp

.PHONY: all clean run plot submit

all: alock

clean:
	rm -f alock results.txt results-rearranged.txt results.png

run:
	./alock | tee results.txt
	# Results stored into 'results.txt', run 'make plot'.

plot:
	python2 plot.py

submit:
	OMP_NUM_THREADS=12 bsub -R "span[ptile=12]" -n 12 make run

alock: alock.cpp Makefile
	$(CXX) $(CXXFLAGS) alock.cpp -o alock

