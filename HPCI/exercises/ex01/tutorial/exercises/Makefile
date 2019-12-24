# for more info on Makefiles, see e. g. http://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/
CC=g++
CFLAGS=-O3 -Wall -Wextra -Wpedantic -std=c++11

all: ex1 ex5 ex1_sol ex5_sol

ex1.o: skeletons/ex1.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

ex5.o: skeletons/ex5.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

ex1_sol.o: solutions/ex1_sol.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

ex5_sol.o: solutions/ex5_sol.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

ex1: ex1.o
	$(CC) -o $@ $< $(CFLAGS)

ex5: ex5.o
	$(CC) -o $@ $< $(CFLAGS)

ex1_sol: ex1_sol.o
	$(CC) -o $@ $< $(CFLAGS)

ex5_sol: ex5_sol.o
	$(CC) -o $@ $< $(CFLAGS)

clean:
	rm -f *.o *~ ex1 ex5 ex1_sol ex5_sol

.PHONY: all
.PHONY: clean
