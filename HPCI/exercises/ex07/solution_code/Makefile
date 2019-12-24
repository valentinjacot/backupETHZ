config ?= prod
blas ?= openblas

ifeq ($(shell uname -s), Darwin)
CXX=g++-8
LIBS += -L/usr/local/opt/openblas/lib/ -lopenblas
CXXFLAGS += -I/usr/local/opt/openblas/include/
else
CXX=g++
ifeq "$(blas)" "mkl"
CXXFLAGS+= -m64 -DUSE_MKL
LIBS+= -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
else
LIBS+= -lopenblas
endif
endif

CXXFLAGS+= -std=c++14 -fopenmp

ifeq "$(config)" "debug"
CXXFLAGS += -g -O0
# addressing errors generally involve accessing/writing beyond the bounds of
# the allocated memory space (i.e. seg fault)
#CXXFLAGS += -fsanitize=address
# undefined behaviors range from operation outcomes depending on uninitialized
# data to wron memory alignment:
#CXXFLAGS += -fsanitize=undefined
endif

ifeq "$(config)" "prod"
CXXFLAGS += -DNDEBUG -O3 -ffast-math
endif


CXXFLAGS+= -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wpedantic
CXXFLAGS+= -Wmissing-declarations -Wredundant-decls -Wshadow -Wwrite-strings
CXXFLAGS+= -Woverloaded-virtual -Wno-unused-parameter -Wno-unused-variable


exec_testGrad: main_testGrad.o
	$(CXX) $(CXXFLAGS) $(LIBS) main_testGrad.o -o $@
	rm main_testGrad.o

exec_classify: main_classify.o
	$(CXX) $(CXXFLAGS) $(LIBS) main_classify.o -o $@
	rm main_classify.o

all: exec_testGrad exec_classify
.DEFAULT_GOAL := all

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o *.dSYM *.s *.d exec_*
