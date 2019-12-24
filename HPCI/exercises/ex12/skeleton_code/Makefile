CXX=g++ -O3 -Wall -Wextra -std=c++11

.PHONY: clean run plot

pse2d: pse2d.cpp
	$(CXX) $< -o $@

clean:
	rm -rf output
	rm -f pse2d movie2d.mp4

run: pse2d
	mkdir -p output
	rm -f output/*
	./pse2d

plot:
	./plot.py
