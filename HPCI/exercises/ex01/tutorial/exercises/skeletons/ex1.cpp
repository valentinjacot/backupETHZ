#include <iostream>
#include <iomanip>

int main()
{
    // TODO
    // WRITE YOUR CODE HERE! (remove the line below)
    int N = 0;
    
    // print the header
    std::cout << "Calculating the factorial for n = 0, ..., N:" << std::endl;
    std::cout << "n" << std::setw(12) << "n!" << std::endl;
    
    int product = 1;
    for(int n = 0; n <= N; ++n)
    {
        if (n != 0)
        {
            // calculate the next next factorial
            product *= n;
        }
        
        // output the factorial
        std::cout << n << std::setw(12) << product << std::endl;
    }
    
    // indicate that no error has occurred
    return 0;
}
