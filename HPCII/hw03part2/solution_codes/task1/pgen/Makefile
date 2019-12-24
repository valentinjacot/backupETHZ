all: pgen

CXXFLAGS = -march=native -DNOALIAS -O3 -g -std=c++17 -Wextra -Wall -Wfatal-errors `gsl-config --cflags`
LDFLAGS = `gsl-config --libs` 
clean:; rm -rf pgen *.o

.PHONY: clean all
