#include <iostream>
#include <limits>

template < class R, class U, class T> // min<double>(1,3.141); specify the output type
R const& min(U const& x, T const& y)
{
	return (x < y ? static_cast<R>(x) : static_cast<R>(y));
}


int main() {
	std::cout << min<double> (1,3.14) <<std::endl;
	
	
	
	
	

	
	
}
