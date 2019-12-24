//Namespace
//--------------------------------------------------------
//EXPL: Eigen or std are namespaces

namespace Myspace {
	int i=9;
}

std::cout << i; //error: 'i' was not declared

std::cout << Myspace::i; //ok

//or
using namespace Myspace;
int i = 7;
std::cout << i; //ok i= 7
