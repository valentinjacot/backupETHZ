CC=g++
CFLAGS=-O3 -Wall -Wextra -Wpedantic

all: basic

basic.o: basic.cpp
	$(CC) -c -o basic.o basic.cpp $(CFLAGS)

basic: basic.o
	$(CC) -o basic basic.o

clean:
	rm -f *.o *~ basic

.PHONY: all
.PHONY: clean
