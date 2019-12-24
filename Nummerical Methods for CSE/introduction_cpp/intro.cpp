void function(MatrixXd A); //very expensive
void function(MartixXd & A); // better
void function(const MartixXd & A); // a passed by constant reference

int Id(int a){std::cout <<"int"; return a;}
int Id(double a){std::cout <<"double"; return a;}
int Id(){std::cout <<"void"; return 1;}
Id(1);
Id(1.);
Id();// return int, double, void

//BETTER: overloading
bool operator<(complex a, complex b) {
	return a.real() < b.real();
}
//can construct completely new operators


