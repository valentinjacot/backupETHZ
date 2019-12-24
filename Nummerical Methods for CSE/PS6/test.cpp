#include <iostream> 
#include <vector>

int main(){
	std::vector<double> v;
	
	std::pair<double,int>* my_vector;
	int n=5;
	my_vector = new std::pair<double,int>[n];
	
	my_vector[1].first=1;
	my_vector[2].first=2;
	my_vector[3].first=3;
	my_vector[4].first=4.;
	my_vector[5].first=5.;
	my_vector[1].second=0;
	my_vector[2].second=0;
	my_vector[3].second=0;
	my_vector[4].second=0;
	my_vector[5].second=0;
	for (int i=1; i<n+1; i++){
	std::cout << my_vector[i].first << "  ";
	std::cout << my_vector[i].second << std::endl;
	std::cout << sizeof(my_vector) << std::endl;
}
	return 0;
}


