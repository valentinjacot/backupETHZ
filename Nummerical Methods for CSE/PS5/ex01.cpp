#include <utility>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>


//#include "header_ex01.h"


template <typename arg, class func, class jac>
void mod_newt_step(const arg & x, arg & x_next, func&& f, jac&& df){
	arg y;
	y=x+f(x)/df(x);
	x_next = y - f(y)/df(x);
}


void mod_newt_step_exec(){
	
	const double a = 0.123;
	auto f = [& a] (double x)
	{ return std::atan(x)-a;};
	auto df = [ /* captured vars. */ ] (double x)
	{ return 1./(x*x+1.);};
	double x_star = std::tan(a); std::cout << "x_star is : " << x_star<< std::endl;
	auto eval_err= [& x_star] (double x_eval)
	{ return std::abs(x_star - x_eval);};

	
	double x = 5;  double x_next = 0;
	double tol = 1.e-15;
	
	std::cout << "initial guess error : " <<eval_err(x)<< std::endl;
	
	int i=0;

	while (eval_err(x)>tol){
		i++;
		mod_newt_step(x, x_next,f,df );
		std::cout << i << "th interation,    x_next is " << x_next << "        and x_star - x_next : " << eval_err(x_next) << std::endl;
		if (i> 15){
			std::cout << " Broke " << std::endl;
			break;}
			x= x_next;
	}

}

int main () {
	mod_newt_step_exec();
	const double a = 0.123;
	std::cout << "tan(a) : " << std::tan(a) << std::endl;
	return 0;
	
}
