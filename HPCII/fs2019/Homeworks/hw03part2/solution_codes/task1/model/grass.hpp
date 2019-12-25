#ifndef _GRASS_H_
#define _GRASS_H_

double getGrassHeight(double x, // X-axis position of the point to be evaluated. Range: [0.0-5.0] meters.
		                  double y, // Y-axis position of the point to be evaluated. Range: [0.0-5.0] meters.
											double ph, // pH of the soil
											double mm); // mm of rain of the previous month.

#endif // _GRASS_H_
