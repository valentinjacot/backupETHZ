CXX = mpic++
CXXFLAGS = -O3 -Wall -Wextra -std=c++11

all: main

main: main.cpp Makefile *.h
	$(CXX) $(CXXFLAGS) main.cpp -o main

run: 
	mpirun -n 4 --oversubscribe ./main

plot:
	./plot u0
	./plot u1

runplot: run plot

clean:
	rm -vf main u0.png u1.png u0 u1

.PHONY: all clean run plot runplot
