#include <iostream>
#include <vector>
#include "calculator.hpp"

int main()
{
    // define on one place what type is used
    // --> change here and the whole code is updated
    typedef double numeric;

    // this is a pointer in order to demonstrate runtime polymorphism
    Function<numeric>* f;
    
    // using the example child class
    f = new ConstantFunction<numeric>(5);
    std::cout << "Constant function:" << std::endl;
    std::cout << "Value at x = 2: " << (*f)(2) << std::endl;
    std::cout << "Derivative at x = 2: " << f->derivative(2) << std::endl;
    delete f;
    
    std::cout << std::endl;
    
    // using the polynomial class
    const std::vector<numeric> coeffs({1., 1., 1.});  // a quadratic polynomial
    
    f = new Polynomial<numeric, 3>(coeffs);
    std::cout << "Polynomial:" << std::endl;
    std::cout << "Value at x = 2: " << (*f)(2) << std::endl;
    std::cout << "Derivative at x = 2: " << f->derivative(2) << std::endl;
    delete f;
    
    return 0;
}
