#include <iostream>
#include <cmath>
using namespace std ;
double epsilon(){
	for( int i=1; ; i++){
	double b = pow(10,-i);
	double c = 1 - b;
	if(c==1){
	return b;}
	}
	
	}
	
int main ()
{cout << epsilon() << endl;
cout << 1-(epsilon()*100) << endl;
	return 0;

	}

