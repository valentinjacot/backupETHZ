#include <iostream> 
#include <math.h>

float Q_rsqrt( float number )
{
	//Fast inverse square root, source: wikipedia

  long i;
  float x2, y;
  const float threehalfs = 1.5F;

  x2 = number * 0.5F;
  y  = number;
  i  = * ( long * ) &y;
  i  = 0x5f3759df - ( i >> 1 );
  y  = * ( float * ) &i;
  y  = y * ( threehalfs - ( x2 * y * y ) );
  y  = y * ( threehalfs - ( x2 * y * y ) );
  y  = y * ( threehalfs - ( x2 * y * y ) );
  return y;
};

int main( int argc, char * argv[] )
{
	
	std::cout << 1./sqrt(26) << std::endl;
	std::cout << Q_rsqrt(26) << std::endl;
	return 0;
}
