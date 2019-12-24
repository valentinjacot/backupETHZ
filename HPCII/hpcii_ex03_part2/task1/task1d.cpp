#include "model/grass.hpp"
#include "korali.h"

// Grass Height at different spots, as measured by Herr Kueheli.
size_t  nSpots;
double* xPos;
double* yPos;
double* heights;


int main(int argc, char* argv[])
{
	// Loading grass height data

	FILE* dataFile = fopen("grass.in", "r");

	fscanf(dataFile, "%lu", &nSpots);
	xPos     = (double*) calloc (sizeof(double), nSpots);
	yPos     = (double*) calloc (sizeof(double), nSpots);
	heights  = (double*) calloc (sizeof(double), nSpots);

	for (int i = 0; i < nSpots; i++)
	{
		fscanf(dataFile, "%le ", &xPos[i]);
		fscanf(dataFile, "%le ", &yPos[i]);
		fscanf(dataFile, "%le ", &heights[i]);
	}


	return 0;
}
