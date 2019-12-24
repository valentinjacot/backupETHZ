#include<iostream>
#include<Eigen/Dense>
using namespace Eigen;

using namespace std;
int main () {
	unsigned int n; 
	cout << "state n" << endl;
	cin >> n;
	Matrix2d m;
	m << 0,1,1,1;
	Vector2d v;
	v << 0, 1;
	//cout << v << endl << " " << m << endl;
	for (unsigned int i= 0;i<n-1; i++){
		v=m*v;
		}
		
	cout << "The " << n << "th Fibonacci number is "<< v.row(1) << endl;
	return 0;
}
	
