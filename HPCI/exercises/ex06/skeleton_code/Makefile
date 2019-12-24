config ?= debug

ifeq ($(shell uname -s), Darwin)
CXX=g++-8
LIBS += -L/usr/local/opt/openblas/lib/ -lopenblas
CXXFLAGS += -I/usr/local/opt/openblas/include/
else
CXX=g++ -lopenblas
endif

CXXFLAGS+= -std=c++11 -fopenmp

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

exec_linear: main_linear.o
	$(CXX) $(CXXFLAGS) $(LIBS) main_linear.o -o $@
	rm main_linear.o

exec_nonlinear: main_nonlinear.o
	$(CXX) $(CXXFLAGS) $(LIBS) main_nonlinear.o -o $@
	rm main_nonlinear.o

all: exec_testGrad exec_linear exec_nonlinear
.DEFAULT_GOAL := all

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o *.dSYM *.s *.d exec_*
