/**********************************************************************
 * Code: Simple Lambda Function Example
 * Author: Sergio Martin
 * ETH Zuerich - HPCSE II (Spring 2019)
 **********************************************************************/

#include <stdio.h>

int main(int argc, char* argv[])
{
 const double scale = 2.0;

 // Square Lambda returns the square of a number times a scale factor
 // scale is taken from the local scope via the capture list []
 // The number to square is provided via the parameter (double x)
 // The function returns a double, hence -> double
 auto squareLambda = [scale](double x) -> double { return scale*x*x; };
 for (double x = 0.0; x < 5.0; x += 1.0)
  printf("X = %f, %f*X^2 = %f\n", (double)x, scale, squareLambda(x));
}

