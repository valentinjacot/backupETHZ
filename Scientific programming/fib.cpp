#include<iostream>
using namespace std;
int main () {
	unsigned int n; 
	cout << "state n" << endl;
	cin >> n;
	double a=0; double b=1;
	for (unsigned int i= 0;i<n-1; i++){
		double t = b;
		b = b + a;
		a = t;
		}
		
	cout << b << endl;
	return 0;
}
	
