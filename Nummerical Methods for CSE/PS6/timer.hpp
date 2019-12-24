#ifndef TIMER_HPP
#define TIMER_HPP
#include <chrono>

//class Timer;

class Timer {
public:

	//Timer();// constructor
	//~Timer();//destructor 
	
	void start();
	void stop();
	double duration(); 	
private: 
	typedef std::chrono::high_resolution_clock::time_point my_time ; 
	my_time start_now;
	my_time stop_now;
};


#endif

