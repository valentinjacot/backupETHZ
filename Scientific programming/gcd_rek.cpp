#include<iostream>
using namespace std;
int gcd(int a, int b) {
	if (a<b){int t = a;	a = b;	b=t;}
	if (b==0){return a;}	
	gcd((a%b),b);
}


int main () {
	int a; int b;
	cout << "state a and b" << endl;
	cin >> a >> b;
	a = gcd(a,b);
	cout << a << endl;
	return 0;
}
	

