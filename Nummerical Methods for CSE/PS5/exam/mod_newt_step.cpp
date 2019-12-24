#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>

template <typename arg, class func, class jac>
void mod_newt_step(const arg & x, arg & x_next, func&& f, jac&& df){
	arg y= x+f(x)/df(x);
	x_next=y-f(y)/df(x);	
}




int main(){
	double a=0.123;
	auto f= [&a](double x)
		{return atan(x)-a;};
	auto df=[](double x){
		return 1./(x*x+1.);};
	double x=5.,x_next;
	double exact_x=tan(a);
	std::cout << std::setprecision(15);

	std::cout << exact_x << std::endl;
	int i=0;
	std::cout << "i: \t step: \t \terror: \t\t cvg: " << std::endl;
	while(true){
		mod_newt_step(x,x_next,f,df);
		i++;
		if(std::abs(x-x_next)<1e-18||std::abs(x - exact_x)<1e-16||i>200){
			break;}
		
		std::cout << i << "\t" << x_next << "\t" << std::abs(x - exact_x)<<"\t"<< log2(x/x_next) <<std::endl;
		x=x_next;
	}
}

