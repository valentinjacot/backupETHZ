CFLAGS = -O2
#CFLAGS = -O0 -g
LFLAGS = 

BINARIES = pgen

.SECONDARY: 
.PHONY: all 
all: $(BINARIES)

heat2d: heat2d.o
	$(CC) $(CFLAGS) $(LFLAGS) -o $@ $^

%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -Wno-unknown-pragmas

.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti *.optrpt 


	
