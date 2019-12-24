#include <math.h>

double getGrassHeight(double xn, // X-axis position of the point to be evaluated. Range: [0.0-5.0] meters.
		                  double yn, // Y-axis position of the point to be evaluated. Range: [0.0-5.0] meters.
											double phn, // pH of the soil
											double mmn) // mm of rain of the previous month.
{
	double x = (xn - 2.8352)*0.8;
	double y = (yn - 2.4976)*0.8;
	double ph = phn/6.0;
	double mm = mmn + 20;
  double s = mm*pow(y-x*x, 2) + ph*pow(x-1., 2);
  return -s;
}
