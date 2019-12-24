#include <chrono>
#include <iostream>

using namespace std;


int main (){
	//start point
	chrono::time_point<chrono::high_resolution_clock> t1 = chrono::time::high_resolution_clock::now();
	int a;
	
	for (int i=1;1<10000;i++){
		a=2^i;
	}
	return a;
	
	//end point
	chrono::time_point<chrono::high_resolution_clock> t2 = chrono::time::high_resolution_clock::now();
	cout << a << endl << t1 << endl << t2 << endl;
	return 0;
}
