// TODO: add include guards
#include <vector>
#include <cmath>

template <typename T>
class Function
{
public:
    virtual ~Function() {};
    
    // calculates the value of the function represented by this class at the point x
    virtual T operator()(const T& x) = 0;
    // calculates the value of the function represented by this class at the point x (assuming the function is in C^(R))
    virtual T derivative(const T& x) = 0;
};

// an example child class
template <typename T>
class ConstantFunction : public Function<T>
{
public:
    ConstantFunction(const T& c) : c_(c) {}
    
    ConstantFunction() : ConstantFunction(0) {}
    
    T operator()(const T& x)
    {
        return c_;
    }
    
    T derivative(const T& x)
    {
        return 0;
    }
    
private:
    T c_;
};

// YOUR TASK: implement the missing parts
template <typename T, int D>
class Polynomial : public Function<T>
{
public:
    // PRE: coeffs must be of size D+1 and contain the coefficients in the following order:
    // coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ... + coeffs[D]*x^D
    // TODO: implement the constructor
    Polynomial(const std::vector<T>& coeffs) {};
    
    T operator()(const T& x)
    {
        // TODO: implement point evaluation
    }
    
    T derivative(const T& x)
    {
        // TODO: implement point evaluation of the derivative
    }
    
private:
    const std::vector<T> coeffs_;
};
// TODO: end include guards
