/// test 
#include <vector>
#include <cassert>

#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>

int main(){
Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(100, 0, 1);
std::cout << x << std::endl;
std::cout <<1e4<< std::endl;
}
