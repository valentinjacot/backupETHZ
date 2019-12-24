CXX=g++
CXXFLAGS=-O3 -Wall -Wextra -std=c++11

.PHONY: all clean run plot

all: associativity

clean:
	rm -f associativity results.txt results.png

run:
	-lscpu
	-grep . /sys/devices/system/cpu/cpu0/cache/index*/*
	./associativity | tee results.txt
	# Results stored into 'results.txt', run 'make plot'.

plot:
	./plot.sh
	# Plot stored into 'results.png'.

plotall:
	./plot.sh results-euler4
	./plot.sh results-euler3
	./plot.sh results-euler2

submit:
	bsub -R "select[model==XeonGold_6150]" -R fullnode -n 36 -W 00:30 make run

submitall:
	bsub -R "select[model==XeonGold_6150]" -R fullnode -n 36 -W 00:30 'lscpu && ./associativity | tee results-euler4.txt'
	bsub -R "select[model==XeonE3_1585Lv5]"            -n 4  -W 00:30 'lscpu && ./associativity | tee results-euler3.txt'
	bsub -R "select[model==XeonE5_2680v3]" -R fullnode -n 24 -W 00:30 'lscpu && ./associativity | tee results-euler2.txt'

associativity: associativity.cpp Makefile
	$(CXX) $(CXXFLAGS) associativity.cpp -o associativity
