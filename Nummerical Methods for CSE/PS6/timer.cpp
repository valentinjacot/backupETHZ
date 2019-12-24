#include <chrono>
#include <iostream>
#include "timer.hpp"

void Timer::start(){
	Timer::start_now = std::chrono::high_resolution_clock::now();
};

void Timer::stop(){	
	Timer::stop_now = std::chrono::high_resolution_clock::now();
}
double Timer::duration(){
	double tmp = static_cast<std::chrono::duration<double> >(Timer::stop_now-Timer::start_now).count();
	return tmp;
};
