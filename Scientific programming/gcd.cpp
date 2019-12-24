#include<iostream>
using namespace std;
int gcd(int a, int b) {
	if (a<b){int t = a;	a = b;	b=t;}
	while (b!=0){
		int t =b;
		b = (a % b);
		a = t;
		}

return a;

}


int main () {
	int a; int b;
	cout << "state a and b" << endl;
	cin >> a >> b;
	a = gcd(a,b);
	cout << a << endl;
	return 0;
}
	
