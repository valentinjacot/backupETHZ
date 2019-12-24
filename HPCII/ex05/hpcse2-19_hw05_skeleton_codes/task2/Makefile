NVCCFLAGS = -O3 -g --use_fast_math -std=c++11
BINARIES = nbody_naive nbody_opt

.SECONDARY:
.PHONY: all
all: $(BINARIES)

$(BINARIES) :: 
	nvcc -o $@ $@.cu $(NVCCFLAGS)

.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
