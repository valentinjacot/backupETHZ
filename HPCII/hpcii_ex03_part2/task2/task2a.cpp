#include "model/heat2d.hpp"
#include "korali.h"

int main(int argc, char* argv[])
{
 // Loading temperature measurement data

 FILE* dataFile = fopen("data.in", "r");
 fscanf(dataFile, "%lu", &p.nPoints);

 p.xPos    = (double*) calloc (sizeof(double), p.nPoints);
 p.yPos    = (double*) calloc (sizeof(double), p.nPoints);
 p.refTemp = (double*) calloc (sizeof(double), p.nPoints);

 for (int i = 0; i < p.nPoints; i++)
 {
  fscanf(dataFile, "%le ", &p.xPos[i]);
  fscanf(dataFile, "%le ", &p.yPos[i]);
  fscanf(dataFile, "%le ", &p.refTemp[i]);
 }

 // How many candles will we simulate?
 // p.nCandles = 1; // 1-Candle Model - Requires 2 parameters (PosX, PosY)
 // p.nCandles = 2; // 2-Candle Model - Requires 4 parameters (PosX, PosY)x2
 // p.nCandles = 3; // 3-Candle Model - Requires 6 parameters (PosX, PosY)x3

 // Start configuring the Problem and the Korali Engine

 return 0;
}
