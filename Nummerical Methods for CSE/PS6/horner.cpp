/// problem Sheet 6, problem 3, about the horner scheme and poynomial evaluation 
#include <iostream>
#include <utility>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include "timer.hpp"



template <typename CoeffVec>
std::pair<double,double> evaldp(const CoeffVec & c, double x){
	size_t n = c.size(); 
	double p_x=0;
	double dp_x=0;
	for (size_t i = 0; i < n-1; i++){
		p_x+=c[i];
		p_x*=x;
		dp_x+=c[i+1]*(n-i-1);
		(i<n-2)?dp_x*=x:0;
	}
	p_x+=c[n-1];
	std::pair<double,double> pair(p_x, dp_x);
	return pair	;
}
template <typename CoeffVec>
std::pair<double,double> evaldp_naive(const CoeffVec & c, double x){
	size_t n = c.size(); 
	double p_x=0;
	double dp_x=0;
	for (size_t i = 0; i < n-1; i++){
		p_x+= c[i]*std::pow(x,n-i-1);
		dp_x+=(n-i-1)*c[i+1]*std::pow(x,n-i-2);
	}
	p_x+= c[n-1];
	std::pair<double,double> pair(p_x, dp_x);
	return pair	;

}

int main () {
		
	std::vector<int> c(10000000,2);
	double y=.87;
	// !!! Not effcient, but doubles the running time ==> makes it easier to see the difference
	Timer t1;
	Timer t2;
	
	t1.start();
	long double p_first = evaldp_naive(c,y).first;
	long double dp_first = evaldp_naive(c,y).second;
	t1.stop();
	
	t2.start();
	long double p_second = evaldp(c,y).first;
	long double dp_second = evaldp(c,y).second;
	t2.stop();
	
	//std::cout << evaldp_naive(c,x).first << "        " << evaldp_naive(c,x).second << std::endl;
	std::cout << std::setprecision(20)<< p_first << "        " << dp_first << "     in : " << t1.duration() << std::endl;
	std::cout << p_second << "        " << dp_second << "     in : " << t2.duration() << std::endl;
		
	return 0;
}

///inf        inf     in : 18.292796312000000114
///inf        inf     in : 10.330089042999999194
/// ==> we do indeed have a much longer execution time with the second version. 
///15.384615384615383249        118.34319526627217556     in : 9.2452059020000003642
///15.38461538461537792        118.34319526627220398     in : 0.62949855600000004241
/// We have a MUCH larger difference with x<1
/// Probably a lost of precision due to the use of the pow ()  function. In the "non-naive"method we don't have this problem


