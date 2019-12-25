CFLAGS = -O2
#CFLAGS = -O0 -g
LFLAGS = 

.SECONDARY: 
.PHONY: all 
all: auxiliar.o

%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -Wno-unknown-pragmas

.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti *.optrpt 


	
